"""
predict.py — Model training, inference, and evaluation.

Orchestrates all three model pipelines and optional downstream tasks:

  Default (no flags): train all three models and run evaluation
  --eval-only:        evaluate from cached .npy outputs (skip training)
  --experiments:      run DL sensitivity sweep + LoRA rank ablation
  --error-analysis:   generate 5 LoRA misprediction case studies

Requires build_features.py to have been run first (PNGs must exist).

Usage:
    python predict.py
    python predict.py --eval-only
    python predict.py --experiments
    python predict.py --error-analysis
"""

import os
import sys
import argparse

import numpy as np

from config import (
    DICOM_ROOT, PROCESSED_DIR, OUTPUT_DIR, RESULTS_DIR,
    MODELS_DIR, LORA_SAVE_PATH, TARGET_SERIES,
    METAL_HU_THRESHOLD, MIN_METAL_VOXELS, Z_BUFFER_MM, SEED, IMG_SIZE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    for d in [OUTPUT_DIR, RESULTS_DIR, MODELS_DIR, LORA_SAVE_PATH]:
        os.makedirs(d, exist_ok=True)


def _load_dicom(dicom_root: str) -> tuple:
    """Load and z-filter DICOM slices. Returns (slices, implant_idx_set, implant_scores)."""
    from dicom_utils import (
        load_dicom_series, filter_target_series, anonymize_series,
        detect_implant_slices, filter_implants_by_z,
    )
    slices = load_dicom_series(dicom_root)
    slices = filter_target_series(slices, TARGET_SERIES)
    slices = anonymize_series(slices)
    implant_slices, _, implant_scores = detect_implant_slices(
        slices, hu_threshold=METAL_HU_THRESHOLD, min_voxels=MIN_METAL_VOXELS,
    )
    implant_idx_set = filter_implants_by_z(slices, implant_slices, Z_BUFFER_MM)
    return slices, implant_idx_set, implant_scores


def _rebuild_split(bone_dir: str, implant_idx_set: set) -> dict:
    from preprocessing import build_dataset_split
    return build_dataset_split(bone_dir, implant_idx_set, seed=SEED)


def _cache_outputs(naive: np.ndarray, dl: np.ndarray, lora: np.ndarray) -> None:
    np.save(os.path.join(OUTPUT_DIR, "naive_output.npy"), naive)
    np.save(os.path.join(OUTPUT_DIR, "dl_output.npy"),   dl)
    np.save(os.path.join(OUTPUT_DIR, "lora_output.npy"), lora)
    print("  Cached model outputs → data/output/*.npy")


def _load_cached() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = [os.path.join(OUTPUT_DIR, f)
             for f in ("naive_output.npy", "dl_output.npy", "lora_output.npy")]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit("Missing cached outputs — run without --eval-only first.\n"
                 + "\n".join(f"  {p}" for p in missing))
    return tuple(np.load(p) for p in paths)


# ── Sub-tasks ─────────────────────────────────────────────────────────────────

def run_training(dicom_root: str) -> tuple:
    """Train all three models. Returns (naive, dl, lora, slices, implant_idx_set, scores)."""
    from naive_baseline import run_naive_baseline
    from dict_learning  import run_dict_learning
    from lora_model     import load_sd_components, train_lora, save_lora_weights, run_lora_inference

    slices, implant_idx_set, implant_scores = _load_dicom(dicom_root)
    bone_dir = os.path.join(PROCESSED_DIR, "bone_window")
    split    = _rebuild_split(bone_dir, implant_idx_set)

    print("\n[Model 1] Naive physics-based artifact simulation...")
    naive_output = run_naive_baseline(slices, implant_idx_set, results_dir=RESULTS_DIR, seed=SEED)

    print("\n[Model 2] Sparse dictionary learning reconstruction...")
    _, dl_output, _ = run_dict_learning(
        split["train_paths"], split["val_paths"], results_dir=RESULTS_DIR,
    )

    print("\n[Model 3] LoRA fine-tuning (Stable Diffusion v1-5)...")
    device = "cuda"
    tok, enc, vae, unet, sched, lora_cfg = load_sd_components(device=device)
    train_lora(
        split["train_implant_paths"], split["val_implant_paths"],
        tok, enc, vae, unet, sched, lora_cfg,
        device=device, results_dir=RESULTS_DIR,
    )
    save_lora_weights(unet, save_path=LORA_SAVE_PATH)
    lora_output = run_lora_inference(
        unet, slices, implant_idx_set, implant_scores,
        device=device, results_dir=RESULTS_DIR,
    )

    _cache_outputs(naive_output, dl_output, lora_output)
    return naive_output, dl_output, lora_output, slices, implant_idx_set, implant_scores, split


def run_eval(naive: np.ndarray, dl: np.ndarray, lora: np.ndarray, split: dict) -> None:
    from PIL import Image
    from evaluation import plot_model_comparison, print_summary_table, plot_pixel_distributions

    bone_dir = os.path.join(PROCESSED_DIR, "bone_window")
    ref_src  = (split["train_implant_paths"] or split["train_paths"])[0]
    ref_img  = np.array(Image.open(ref_src).convert("L").resize((IMG_SIZE, IMG_SIZE)))

    scores = plot_model_comparison(ref_img, naive, dl, lora, results_dir=RESULTS_DIR)
    print_summary_table(ref_img, naive, dl, lora, scores)
    plot_pixel_distributions(ref_img, naive, dl, lora, results_dir=RESULTS_DIR)


def run_experiments_task(split: dict) -> None:
    from experiments import run_dl_sensitivity_sweep, run_lora_rank_ablation
    run_dl_sensitivity_sweep(split["train_paths"], split["val_paths"], results_dir=RESULTS_DIR)
    run_lora_rank_ablation(
        split["train_implant_paths"], split["val_implant_paths"], results_dir=RESULTS_DIR,
    )


def run_error_task(slices, implant_idx_set, implant_scores) -> None:
    import torch
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig, get_peft_model
    from config import LORA_RANK, LORA_ALPHA, LORA_TARGET_MODS, LORA_DROPOUT, BASE_MODEL_ID
    from error_analysis import run_error_analysis

    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_ID, subfolder="unet")
    lora_cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
                          target_modules=LORA_TARGET_MODS, lora_dropout=LORA_DROPOUT, bias="none")
    unet.requires_grad_(False)
    unet = get_peft_model(unet, lora_cfg)
    unet.load_adapter(LORA_SAVE_PATH, adapter_name="default")
    unet.to("cuda", dtype=torch.float32)
    unet.eval()

    run_error_analysis(unet, slices, implant_idx_set, implant_scores,
                       device="cuda", results_dir=RESULTS_DIR)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CochleArt predict / evaluate / experiment")
    parser.add_argument("--eval-only",       action="store_true",
                        help="Skip training; evaluate from cached .npy outputs")
    parser.add_argument("--experiments",     action="store_true",
                        help="Run DL sensitivity sweep + LoRA rank ablation after training")
    parser.add_argument("--error-analysis",  action="store_true",
                        help="Generate 5 LoRA misprediction case studies after training")
    args = parser.parse_args()

    _ensure_dirs()
    dicom_root = os.environ.get("DICOM_ROOT", DICOM_ROOT)

    print("=" * 60)
    print("CochleArt — predict.py")
    print("=" * 60)

    if args.eval_only:
        naive, dl, lora = _load_cached()
        slices, implant_idx_set, implant_scores = _load_dicom(dicom_root)
        bone_dir = os.path.join(PROCESSED_DIR, "bone_window")
        split    = _rebuild_split(bone_dir, implant_idx_set)
    else:
        naive, dl, lora, slices, implant_idx_set, implant_scores, split = \
            run_training(dicom_root)

    print("\n[Evaluation]")
    run_eval(naive, dl, lora, split)

    if args.experiments:
        print("\n[Experiments]")
        run_experiments_task(split)

    if args.error_analysis:
        print("\n[Error Analysis]")
        run_error_task(slices, implant_idx_set, implant_scores)

    print("\n" + "=" * 60)
    print("Done.")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Models:  {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
