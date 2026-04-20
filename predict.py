"""
predict.py — Cog predictor for CochleArt on Replicate.

Loaded once at container start (setup), then called for each request (predict).
All heavy models are cached in instance variables so subsequent predictions
are fast (no re-loading).
"""

import contextlib
import pathlib
import sys
import tempfile

import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path

sys.path.insert(0, "scripts")


class Predictor(BasePredictor):

    def setup(self) -> None:
        """
        Load everything expensive once when the container starts:
          - DICOM series + implant detection (same pipeline as training)
          - Dictionary learning atoms
          - Stable Diffusion v1.5 + LoRA adapter
        """
        import torch
        from dicom_utils import (
            load_dicom_series,
            detect_implant_slices,
            filter_implants_by_z,
        )

        # ── DICOM data ────────────────────────────────────────────────────────
        slices = load_dicom_series("data/raw")
        implant_slices, implant_idx_set, implant_scores = detect_implant_slices(slices)
        z_filtered_idx = filter_implants_by_z(slices, implant_slices)
        if not z_filtered_idx:
            z_filtered_idx = set(range(len(slices)))

        self.slices          = slices
        self.implant_idx_set = implant_idx_set
        self.z_filtered_idx  = z_filtered_idx
        self.implant_scores  = implant_scores
        print(f"DICOM ready: {len(slices)} slices, "
              f"{len(implant_idx_set)} implant, {len(z_filtered_idx)} z-filtered")

        # ── Dictionary atoms ─────────────────────────────────────────────────
        self.atoms = np.load("models/dict_learning/atoms.npy")
        print(f"Dictionary atoms loaded: {self.atoms.shape}")

        # ── LoRA pipeline ─────────────────────────────────────────────────────
        from diffusers import StableDiffusionImg2ImgPipeline
        from peft import PeftModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.unet = PeftModel.from_pretrained(pipe.unet, "models/lora")
        pipe      = pipe.to(device)
        pipe.enable_attention_slicing()

        self.lora_pipe = pipe
        self.device    = device
        print(f"LoRA pipeline ready on {device}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_slice(self, seed: int, prefer_implant: bool) -> np.ndarray:
        """Return a bone-windowed CT slice at the cochlear implant level."""
        from dicom_utils import dicom_to_hu, apply_window
        from config import WINDOWS

        if prefer_implant and self.implant_scores:
            ranked = sorted(self.implant_scores, key=lambda x: x[1], reverse=True)
            pool   = [self.slices[i] for i, _ in ranked[:max(1, len(ranked) // 2)]]
        else:
            clean  = self.z_filtered_idx - self.implant_idx_set or self.z_filtered_idx
            pool   = [self.slices[i] for i in sorted(clean)]

        ds       = pool[seed % len(pool)]
        hu       = dicom_to_hu(ds)
        windowed = apply_window(hu, *WINDOWS["bone"])
        return np.array(Image.fromarray(windowed).resize((256, 256)))

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        model: str = Input(
            description="Generation model",
            choices=["naive", "dict", "lora"],
            default="naive",
        ),
        prompt: str = Input(
            description="Text prompt (LoRA only — describes desired artifact characteristics)",
            default=(
                "axial head CT scan at skull base level, petrous temporal bone, "
                "cochlear implant with electrode array, bone window, "
                "bright white hyperdense metal artifact, beam-hardening streak artifacts, "
                "high-resolution grayscale HRCT"
            ),
        ),
        seed: int = Input(
            description="Random seed (0–9999) — change to get different slice / artifact placement",
            default=42, ge=0, le=9999,
        ),
        streak_intensity: float = Input(
            description="Artifact streak intensity (Naive only)",
            default=0.35, ge=0.1, le=0.8,
        ),
        noise_std: float = Input(
            description="Photon-starvation noise level (Naive only)",
            default=0.04, ge=0.01, le=0.15,
        ),
        n_angles: int = Input(
            description="Number of streak directions (Naive only)",
            default=24, ge=8, le=48,
        ),
    ) -> Path:

        import torch

        base_img = self._get_slice(seed=seed % 97, prefer_implant=(model == "lora"))

        # ── Model 1: Physics simulation ───────────────────────────────────────
        if model == "naive":
            from naive_baseline import NaiveArtifactSimulator
            result = NaiveArtifactSimulator(
                streak_intensity=streak_intensity,
                n_angles=n_angles,
                streak_width=3,
                noise_std=noise_std,
                seed=seed,
            ).simulate(base_img)

        # ── Model 2: Dictionary learning ──────────────────────────────────────
        elif model == "dict":
            from sklearn.decomposition import sparse_encode
            ps, stride, n_nz = 16, 8, 6
            small  = np.array(Image.fromarray(base_img).resize((128, 128)))
            h, w   = small.shape
            output = np.zeros((h, w), np.float32)
            count  = np.zeros((h, w), np.float32)
            for row in range(0, h - ps + 1, stride):
                for col in range(0, w - ps + 1, stride):
                    patch = small[row:row+ps, col:col+ps].ravel().astype(np.float32) / 255.0
                    code  = sparse_encode(
                        patch.reshape(1, -1), self.atoms,
                        algorithm="omp", n_nonzero_coefs=n_nz,
                    )
                    recon = (code @ self.atoms).reshape(ps, ps)
                    output[row:row+ps, col:col+ps] += recon
                    count[row:row+ps,  col:col+ps] += 1
            count  = np.where(count == 0, 1, count)
            result = np.array(
                Image.fromarray(
                    np.clip(output / count * 255, 0, 255).astype(np.uint8)
                ).resize((256, 256))
            )

        # ── Model 3: LoRA diffusion ───────────────────────────────────────────
        elif model == "lora":
            from config import (
                INFERENCE_GUIDANCE, INFERENCE_STEPS, INFERENCE_STRENGTH,
                NEGATIVE_PROMPT, TRAIN_PROMPT,
            )
            if not prompt.strip():
                prompt = TRAIN_PROMPT

            pil_in = Image.fromarray(base_img).convert("RGB").resize((256, 256))
            ctx    = (torch.autocast(self.device)
                      if self.device == "cuda" else contextlib.nullcontext())
            with ctx:
                out = self.lora_pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=pil_in,
                    strength=INFERENCE_STRENGTH,
                    num_inference_steps=INFERENCE_STEPS,
                    guidance_scale=INFERENCE_GUIDANCE,
                    generator=torch.Generator(self.device).manual_seed(seed),
                )
            result = np.array(out.images[0].convert("L"))

        else:
            raise ValueError(f"Unknown model: {model}")

        # Save and return
        out_path = pathlib.Path(tempfile.mkdtemp()) / "output.png"
        Image.fromarray(result).save(str(out_path))
        return Path(out_path)
