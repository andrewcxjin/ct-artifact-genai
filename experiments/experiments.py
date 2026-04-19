"""
experiments.py — Sensitivity sweep and rank ablation experiments.

Experiment 1: Grid search over DL n_atoms × n_nonzero to find the sweet spot
              between reconstruction quality and training time.

Experiment 2: LoRA rank ablation — trains ranks {4, 8, 16} under identical
              conditions and compares val-loss curves.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from config import (
    ATOM_GRID, NONZERO_GRID, SWEEP_MAX_IMGS, SWEEP_MAX_PATCHES, SWEEP_N_ITER,
    ABLATION_RANKS, LORA_TRAIN_STEPS, LORA_LR, LORA_BATCH_SIZE,
    LORA_ALPHA, LORA_TARGET_MODS, LORA_DROPOUT,
    BASE_MODEL_ID, TRAIN_PROMPT, RESULTS_DIR, SEED,
    IMG_SIZE, DL_PATCH_SIZE, DL_RECON_STRIDE,
)


def _smooth(values: list, window: int = 5) -> np.ndarray:
    """Simple moving-average smoother."""
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def run_dl_sensitivity_sweep(
    train_paths: list,
    val_paths: list,
    atom_grid: list    = ATOM_GRID,
    nonzero_grid: list = NONZERO_GRID,
    max_imgs: int      = SWEEP_MAX_IMGS,
    max_patches: int   = SWEEP_MAX_PATCHES,
    n_iter: int        = SWEEP_N_ITER,
    results_dir: str   = RESULTS_DIR,
) -> dict:
    """
    Grid search over n_atoms × n_nonzero for MiniBatchDictionaryLearning.

    Args:
        train_paths: Paths to training bone-window PNG slices.
        val_paths: Paths to validation bone-window PNG slices.
        atom_grid: List of n_atoms values to sweep.
        nonzero_grid: List of n_nonzero_coefs values to sweep.
        max_imgs: Maximum training images to load per configuration.
        max_patches: Maximum patches to sample per configuration.
        n_iter: MiniBatchDictionaryLearning iterations per configuration.
        results_dir: Directory to save the heatmap figure.

    Returns:
        Dict mapping (n_atoms, n_nonzero) → ssim_score.
    """
    from dict_learning import DictionaryLearningGenerator

    train_imgs = [
        np.array(Image.open(p).convert("L")) for p in train_paths[:max_imgs]
    ]
    val_img   = np.array(Image.open(val_paths[5]).convert("L"))
    val_small = np.array(Image.fromarray(val_img).resize((128, 128)))
    val_disp  = np.array(Image.fromarray(val_small).resize((IMG_SIZE, IMG_SIZE)))

    results = {}
    print(f"DL sensitivity sweep: {len(atom_grid)} × {len(nonzero_grid)} = "
          f"{len(atom_grid) * len(nonzero_grid)} configurations")

    for n_atoms in atom_grid:
        for n_nz in nonzero_grid:
            print(f"  n_atoms={n_atoms}, n_nonzero={n_nz} ...", end=" ", flush=True)
            gen = DictionaryLearningGenerator(
                n_atoms=n_atoms, n_nonzero=n_nz, n_iter=n_iter, random_state=SEED
            )
            gen.fit(train_imgs, max_patches=max_patches)
            recon_sm   = gen.reconstruct(val_small, stride=DL_RECON_STRIDE)
            recon_disp = np.array(Image.fromarray(recon_sm).resize((IMG_SIZE, IMG_SIZE)))
            score      = ssim(val_disp, recon_disp, data_range=255)
            results[(n_atoms, n_nz)] = score
            print(f"SSIM={score:.3f}")

    grid = np.array([[results[(a, nz)] for nz in nonzero_grid] for a in atom_grid])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, cmap="viridis", aspect="auto",
                   vmin=grid.min() - 0.01, vmax=grid.max() + 0.01)
    ax.set_xticks(range(len(nonzero_grid))); ax.set_xticklabels(nonzero_grid)
    ax.set_yticks(range(len(atom_grid)));   ax.set_yticklabels(atom_grid)
    ax.set_xlabel("n_nonzero_coefs"); ax.set_ylabel("n_atoms")
    ax.set_title("DL Sensitivity: SSIM vs n_atoms × n_nonzero")

    for i, a in enumerate(atom_grid):
        for j, nz in enumerate(nonzero_grid):
            ax.text(j, i, f"{results[(a, nz)]:.3f}",
                    ha="center", va="center", fontsize=8, color="white")

    plt.colorbar(im, ax=ax, label="SSIM")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "dl_sensitivity_heatmap.png"), dpi=150)
    plt.show()

    best_k = max(results, key=results.get)
    print(f"\nBest config: n_atoms={best_k[0]}, n_nonzero={best_k[1]}, "
          f"SSIM={results[best_k]:.4f}")
    return results


def run_lora_rank_ablation(
    train_implant_paths: list,
    val_implant_paths: list,
    ranks: list      = ABLATION_RANKS,
    train_steps: int = LORA_TRAIN_STEPS,
    device: str      = "cuda",
    results_dir: str = RESULTS_DIR,
) -> dict:
    """
    Train LoRA adapters at multiple ranks and compare validation loss curves.

    Args:
        train_implant_paths: Paths to training PNG slices (implant, z-filtered).
        val_implant_paths: Paths to validation PNG slices (implant, z-filtered).
        ranks: List of LoRA ranks to train.
        train_steps: Number of gradient steps per rank.
        device: "cuda" or "cpu".
        results_dir: Directory to save loss-curve figure.

    Returns:
        Dict mapping rank (int) → {"train": list[float], "val": list[float]}.
    """
    import torch
    import gc
    from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTokenizer, CLIPTextModel
    from lora_model import train_lora

    weight_dtype = torch.float16
    all_losses   = {}

    for rank in ranks:
        print(f"\n{'='*50}\n  LoRA rank ablation: rank = {rank}\n{'='*50}")

        tokenizer    = CLIPTokenizer.from_pretrained(BASE_MODEL_ID, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_ID, subfolder="text_encoder")
        vae          = AutoencoderKL.from_pretrained(BASE_MODEL_ID, subfolder="vae")
        unet         = UNet2DConditionModel.from_pretrained(BASE_MODEL_ID, subfolder="unet")
        scheduler    = DDPMScheduler.from_pretrained(BASE_MODEL_ID, subfolder="scheduler")

        lora_cfg = LoraConfig(
            r=rank, lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODS,
            lora_dropout=LORA_DROPOUT, bias="none",
        )

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)
        unet = get_peft_model(unet, lora_cfg)
        unet.print_trainable_parameters()

        vae.to(device,          dtype=weight_dtype)
        text_encoder.to(device, dtype=weight_dtype)
        unet.to(device,         dtype=torch.float32)

        train_losses, val_losses = train_lora(
            train_implant_paths=train_implant_paths,
            val_implant_paths=val_implant_paths,
            tokenizer=tokenizer, text_encoder=text_encoder,
            vae=vae, unet=unet, scheduler=scheduler, lora_config=lora_cfg,
            device=device, train_steps=train_steps, results_dir=results_dir,
        )
        all_losses[rank] = {"train": train_losses, "val": val_losses}

        del unet, vae, text_encoder, tokenizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    _plot_rank_ablation(all_losses, train_steps, results_dir)
    return all_losses


def _plot_rank_ablation(all_losses: dict, train_steps: int, results_dir: str) -> None:
    """Plot smoothed train/val loss curves for each rank."""
    colors = {4: "salmon", 8: "cornflowerblue", 16: "limegreen"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("LoRA Rank Ablation — Train & Val Loss", fontsize=13)

    for rank, losses in all_losses.items():
        col = colors.get(rank, "gray")
        ax1.plot(_smooth(losses["train"]), color=col, alpha=0.85, label=f"rank={rank}")
        ax2.plot(
            [i * 10 for i in range(len(losses["val"]))], losses["val"],
            color=col, linestyle="--", marker="o", markersize=3, label=f"rank={rank}",
        )

    for ax, title in [(ax1, "Smoothed Train Loss"), (ax2, "Val Loss (every 10 steps)")]:
        ax.set_xlabel("Step"); ax.set_ylabel("MSE Loss")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "rank_ablation_loss.png"), dpi=150)
    plt.show()

    print("\n" + "=" * 50)
    print(f"{'Rank':<10} {'Final Train':>12} {'Final Val':>10}")
    print("=" * 50)
    for rank, losses in sorted(all_losses.items()):
        print(f"{rank:<10} {losses['train'][-1]:>12.4f} "
              f"{losses['val'][-1] if losses['val'] else float('nan'):>10.4f}")
    print("=" * 50)


if __name__ == "__main__":
    print("Import this module — do not run directly.")
