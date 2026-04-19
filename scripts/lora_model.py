"""
lora_model.py — LoRA fine-tuning and inference (Model 3).

Fine-tunes Stable Diffusion v1-5 with Low-Rank Adaptation (LoRA) on
z-filtered cochlear implant CT slices. Uses img2img inference so the
model refines a real CT rather than generating from pure noise.

References:
  - LoRA: https://arxiv.org/abs/2106.09685
  - PEFT library: https://github.com/huggingface/peft
  - Diffusers: https://github.com/huggingface/diffusers
"""

import os
import gc

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import DataLoader

from config import (
    BASE_MODEL_ID, LORA_RANK, LORA_ALPHA, LORA_TARGET_MODS, LORA_DROPOUT,
    LORA_TRAIN_STEPS, LORA_LR, LORA_BATCH_SIZE, LORA_IMG_SIZE, LORA_SAVE_PATH,
    TRAIN_PROMPT, NEGATIVE_PROMPT,
    INFERENCE_STEPS, INFERENCE_GUIDANCE, INFERENCE_STRENGTH, INFERENCE_SEED,
    RESULTS_DIR, WINDOWS,
)
from cochlearart_dataset import CTSliceDataset


def clear_gpu_memory(*model_names: str) -> None:
    """
    Delete named model variables from the global scope and free VRAM.

    Args:
        model_names: Variable names (strings) of models to delete.
    """
    import builtins
    for name in model_names:
        if name in dir(builtins):
            del builtins.__dict__[name]
    torch.cuda.empty_cache()
    gc.collect()


def load_sd_components(base_model_id: str = BASE_MODEL_ID, device: str = "cuda"):
    """
    Load all Stable Diffusion components and apply LoRA to the UNet.

    Freezes VAE and text encoder — only LoRA adapter weights are trainable.

    Args:
        base_model_id: HuggingFace model ID for the base SD model.
        device: Target device ("cuda" or "cpu").

    Returns:
        Tuple of (tokenizer, text_encoder, vae, unet, scheduler, lora_config).
    """
    from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTokenizer, CLIPTextModel

    print(f"Loading {base_model_id}...")
    weight_dtype = torch.float16

    lora_config = LoraConfig(
        r              = LORA_RANK,
        lora_alpha     = LORA_ALPHA,
        target_modules = LORA_TARGET_MODS,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
    )

    tokenizer    = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    unet         = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    scheduler    = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    vae.to(device,          dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device,         dtype=torch.float32)

    print("Model loaded and LoRA applied")
    return tokenizer, text_encoder, vae, unet, scheduler, lora_config


def train_lora(
    train_implant_paths: list,
    val_implant_paths: list,
    tokenizer,
    text_encoder,
    vae,
    unet,
    scheduler,
    lora_config,
    device: str      = "cuda",
    train_steps: int = LORA_TRAIN_STEPS,
    lr: float        = LORA_LR,
    batch_size: int  = LORA_BATCH_SIZE,
    results_dir: str = RESULTS_DIR,
) -> tuple[list, list]:
    """
    Run the LoRA fine-tuning loop on z-filtered implant slices.

    Logs training and validation loss every 10 steps. Saves the loss curve
    to results_dir.

    Args:
        train_implant_paths: Paths to training PNG slices (implant only).
        val_implant_paths: Paths to validation PNG slices (implant only).
        tokenizer: CLIPTokenizer.
        text_encoder: Frozen CLIPTextModel.
        vae: Frozen AutoencoderKL.
        unet: LoRA-wrapped UNet2DConditionModel.
        scheduler: DDPMScheduler.
        lora_config: LoraConfig used when wrapping the UNet.
        device: "cuda" or "cpu".
        train_steps: Number of gradient steps.
        lr: AdamW learning rate.
        batch_size: Training batch size.
        results_dir: Directory for saving the loss curve figure.

    Returns:
        Tuple of (train_losses, val_losses) as lists of floats.
    """
    from diffusers.optimization import get_cosine_schedule_with_warmup

    if len(train_implant_paths) < 5:
        raise RuntimeError(
            f"Only {len(train_implant_paths)} implant slices after z-filter. "
            "Lower Z_BUFFER_MM or METAL_HU_THRESHOLD and re-run."
        )

    weight_dtype = torch.float16

    dataset    = CTSliceDataset(train_implant_paths, TRAIN_PROMPT, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_dataset    = CTSliceDataset(val_implant_paths, TRAIN_PROMPT, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=lr, weight_decay=1e-2,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=train_steps
    )

    unet.train()
    train_losses, val_losses = [], []
    data_iter = iter(dataloader)

    print(f"Training LoRA on {len(train_implant_paths)} slices ({train_steps} steps)...")

    for step in range(train_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch     = next(data_iter)

        pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
        input_ids    = batch["input_ids"].to(device)

        with torch.no_grad():
            latents               = vae.encode(pixel_values).latent_dist.sample()
            latents               = latents * vae.config.scaling_factor
            encoder_hidden_states = text_encoder(input_ids)[0]

        noise         = torch.randn_like(latents)
        timesteps     = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device
        ).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        noise_pred = unet(
            noisy_latents.to(torch.float32),
            timesteps,
            encoder_hidden_states.to(torch.float32),
        ).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise.to(torch.float32))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        train_losses.append(loss.item())

        if (step + 1) % 10 == 0:
            unet.eval()
            with torch.no_grad():
                vl_batch = next(iter(val_dataloader))
                vl_pix   = vl_batch["pixel_values"].to(device, dtype=weight_dtype)
                vl_ids   = vl_batch["input_ids"].to(device)
                vl_lat   = vae.encode(vl_pix).latent_dist.sample() * vae.config.scaling_factor
                vl_hs    = text_encoder(vl_ids)[0]
                vl_noise = torch.randn_like(vl_lat)
                vl_t     = torch.randint(
                    0, scheduler.config.num_train_timesteps, (1,), device=device
                ).long()
                vl_noisy = scheduler.add_noise(vl_lat, vl_noise, vl_t)
                vl_pred  = unet(vl_noisy.float(), vl_t, vl_hs.float()).sample
                vl_loss  = torch.nn.functional.mse_loss(
                    vl_pred, vl_noise.float()
                ).item()
            val_losses.append(vl_loss)
            unet.train()
            print(f"  Step {step+1:4d}/{train_steps}  "
                  f"train={loss.item():.4f}  val={vl_loss:.4f}")

    print(f"Training done — final loss: {train_losses[-1]:.4f}")

    plt.figure(figsize=(10, 3))
    plt.plot(train_losses, label="train")
    plt.plot(
        [i * 10 for i in range(len(val_losses))], val_losses,
        label="val", linestyle="--"
    )
    plt.xlabel("Step"); plt.ylabel("MSE Loss")
    plt.title(f"LoRA Loss Curve (rank={lora_config.r}, {train_steps} steps)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "lora_loss_curve.png"), dpi=150)
    plt.show()

    return train_losses, val_losses


def save_lora_weights(unet, save_path: str = LORA_SAVE_PATH) -> None:
    """
    Save the LoRA adapter weights to disk (adapter only, not full SD model).

    Args:
        unet: Trained LoRA-wrapped UNet.
        save_path: Directory to save adapter files.
    """
    unet.save_pretrained(save_path)
    size_mb = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(save_path)
        for f in files
    ) / 1e6
    print(f"LoRA weights saved to {save_path}  ({size_mb:.1f} MB)")


def run_lora_inference(
    unet,
    slices: list,
    implant_idx_set: set,
    implant_scores: list,
    base_model_id: str  = BASE_MODEL_ID,
    device: str         = "cuda",
    results_dir: str    = RESULTS_DIR,
    train_steps: int    = LORA_TRAIN_STEPS,
    inference_seed: int = INFERENCE_SEED,
) -> np.ndarray:
    """
    Run img2img inference using the fine-tuned LoRA model.

    Uses the most metal-dense real implant slice as the starting image
    so anatomical structure is preserved.

    Args:
        unet: Trained LoRA-wrapped UNet.
        slices: Full DICOM slice list.
        implant_idx_set: Z-filtered set of implant slice indices.
        implant_scores: List of (index, voxel_count) from detection.
        base_model_id: HuggingFace model ID for the base pipeline.
        device: "cuda" or "cpu".
        results_dir: Directory to save the inference output figure.
        train_steps: Used only for the plot title.
        inference_seed: Generator seed for reproducibility.

    Returns:
        uint8 numpy array of the generated grayscale image.
    """
    from diffusers import StableDiffusionImg2ImgPipeline
    from dicom_utils import dicom_to_hu, apply_window

    unet.eval()
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model_id, unet=unet,
        torch_dtype=torch.float16, safety_checker=None,
    ).to(device)

    best_ref_idx = max(implant_idx_set, key=lambda i: dict(implant_scores)[i])
    ref_hu       = dicom_to_hu(slices[best_ref_idx])
    ref_windowed = apply_window(ref_hu, *WINDOWS["bone"])
    ref_pil      = (
        Image.fromarray(ref_windowed)
        .convert("RGB")
        .resize((LORA_IMG_SIZE, LORA_IMG_SIZE))
    )

    with torch.autocast("cuda"):
        result = pipe(
            prompt              = TRAIN_PROMPT,
            negative_prompt     = NEGATIVE_PROMPT,
            image               = ref_pil,
            strength            = INFERENCE_STRENGTH,
            num_inference_steps = INFERENCE_STEPS,
            guidance_scale      = INFERENCE_GUIDANCE,
            generator           = torch.Generator(device).manual_seed(inference_seed),
        )

    generated_gray = np.array(result.images[0].convert("L"))
    ref_disp       = apply_window(ref_hu, *WINDOWS["bone"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Model 3: LoRA Generation — {train_steps} steps", fontsize=11)
    axes[0].imshow(ref_disp,       cmap="gray")
    axes[0].set_title("Real CT (reference)"); axes[0].axis("off")
    axes[1].imshow(generated_gray, cmap="gray")
    axes[1].set_title(f"LoRA Generated ({train_steps} steps)"); axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "lora_generation_test.png"), dpi=150)
    plt.show()

    return generated_gray


if __name__ == "__main__":
    print("Import this module — do not run directly.")
