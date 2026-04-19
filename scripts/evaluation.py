"""
evaluation.py — Model comparison, SSIM scoring, and result visualisation.

Provides utilities for resizing outputs to a common resolution, computing
SSIM, plotting side-by-side model comparisons, and printing summary tables.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from config import IMG_SIZE, RESULTS_DIR, LORA_TRAIN_STEPS


def resize_to(arr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """
    Resize a uint8 grayscale array to (size × size), ensuring L-mode output.

    Args:
        arr: Input uint8 array (H × W).
        size: Target side length in pixels.

    Returns:
        uint8 numpy array of shape (size, size).
    """
    img = Image.fromarray(arr)
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img.resize((size, size)))


def safe_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute SSIM between two images, returning NaN on failure.

    Args:
        a: Reference image (uint8).
        b: Comparison image (uint8).

    Returns:
        SSIM score in [-1, 1], or NaN if computation fails.
    """
    try:
        return ssim(a, b, data_range=255)
    except Exception:
        return float("nan")


def plot_model_comparison(
    ref_img: np.ndarray,
    naive_output: np.ndarray,
    dl_output: np.ndarray,
    lora_output: np.ndarray,
    train_steps: int = LORA_TRAIN_STEPS,
    results_dir: str = RESULTS_DIR,
) -> dict:
    """
    Plot all three model outputs against the real CT reference.

    Args:
        ref_img: Real CT reference image (uint8).
        naive_output: Naive baseline output (uint8).
        dl_output: Dictionary learning output (uint8).
        lora_output: LoRA generation output (uint8).
        train_steps: LoRA step count for axis label.
        results_dir: Directory to save the comparison figure.

    Returns:
        Dict with keys naive_ssim, dl_ssim, lora_ssim.
    """
    naive_r = resize_to(naive_output)
    dl_r    = resize_to(dl_output)
    lora_r  = resize_to(lora_output)

    s_naive = safe_ssim(ref_img, naive_r)
    s_dl    = safe_ssim(ref_img, dl_r)
    s_lora  = safe_ssim(ref_img, lora_r)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle("Model Comparison — All Three vs Real CT",
                 fontsize=14, color="white")

    panels = [
        (ref_img, "Real CT (Reference)",                             "white"),
        (naive_r, f"Naive Baseline\nSSIM: {s_naive:.3f}",           "salmon"),
        (dl_r,    f"Dict Learning\nSSIM: {s_dl:.3f}",               "cornflowerblue"),
        (lora_r,  f"LoRA ({train_steps} steps)\nSSIM: {s_lora:.3f}", "limegreen"),
    ]
    for ax, (img, title, color) in zip(axes, panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, color=color, fontsize=11, pad=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "model_comparison.png"),
        dpi=150, facecolor="#1a1a1a",
    )
    plt.show()

    return {"naive_ssim": s_naive, "dl_ssim": s_dl, "lora_ssim": s_lora}


def print_summary_table(
    ref_img: np.ndarray,
    naive_output: np.ndarray,
    dl_output: np.ndarray,
    lora_output: np.ndarray,
    scores: dict,
    train_steps: int = LORA_TRAIN_STEPS,
) -> None:
    """
    Print a formatted metrics table for all three models.

    Args:
        ref_img: Real CT reference image.
        naive_output, dl_output, lora_output: Model outputs (uint8).
        scores: Dict returned by plot_model_comparison.
        train_steps: LoRA step count for the row label.
    """
    naive_r = resize_to(naive_output)
    dl_r    = resize_to(dl_output)
    lora_r  = resize_to(lora_output)

    print("=" * 65)
    print(f"{'Model':<30} {'SSIM':>8} {'Mean px':>9} {'Std px':>8}")
    print("=" * 65)

    rows = [
        ("Real (Reference)",                 ref_img, 1.0),
        ("Model 1: Naive Baseline",          naive_r, scores["naive_ssim"]),
        ("Model 2: Dict Learning",           dl_r,    scores["dl_ssim"]),
        (f"Model 3: LoRA ({train_steps}st)", lora_r,  scores["lora_ssim"]),
    ]
    for name, img, s in rows:
        print(f"{name:<30} {s:>8.3f} {img.mean():>9.1f} {img.std():>8.1f}")

    print("=" * 65)


def plot_pixel_distributions(
    ref_img: np.ndarray,
    naive_output: np.ndarray,
    dl_output: np.ndarray,
    lora_output: np.ndarray,
    results_dir: str = RESULTS_DIR,
) -> None:
    """
    Plot pixel-value histograms for all four images side by side.

    Args:
        ref_img: Real CT reference.
        naive_output, dl_output, lora_output: Model outputs (uint8).
        results_dir: Directory to save the figure.
    """
    naive_r = resize_to(naive_output)
    dl_r    = resize_to(dl_output)
    lora_r  = resize_to(lora_output)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(
        "Pixel Value Distributions (proxy for HU distribution match)",
        fontsize=12,
    )
    colors = ["white", "salmon", "cornflowerblue", "limegreen"]
    for ax, (name, img), color in zip(
        axes,
        [("Real", ref_img), ("Naive", naive_r), ("Dict", dl_r), ("LoRA", lora_r)],
        colors,
    ):
        ax.hist(img.ravel(), bins=64, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Pixel value (0–255)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 255)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pixel_distributions.png"), dpi=150)
    plt.show()


def run_evaluation(
    slices: list,
    train_implant_paths: list,
    train_paths: list,
    naive_output: np.ndarray,
    dl_output: np.ndarray,
    lora_output: np.ndarray,
    results_dir: str = RESULTS_DIR,
) -> dict:
    """
    Full evaluation pipeline: comparison plot, summary table, distributions.

    Args:
        slices: Loaded DICOM datasets (unused directly; kept for interface consistency).
        train_implant_paths: Implant PNG paths (used to select reference image).
        train_paths: All train PNG paths (fallback if no implant paths).
        naive_output: Naive baseline result array.
        dl_output: Dictionary learning result array.
        lora_output: LoRA generation result array.
        results_dir: Output directory for figures.

    Returns:
        Dict of SSIM scores.
    """
    ref_src = train_implant_paths[0] if train_implant_paths else train_paths[0]
    ref_img = np.array(Image.open(ref_src).convert("L").resize((IMG_SIZE, IMG_SIZE)))

    scores = plot_model_comparison(
        ref_img, naive_output, dl_output, lora_output,
        results_dir=results_dir,
    )
    print_summary_table(ref_img, naive_output, dl_output, lora_output, scores)
    plot_pixel_distributions(ref_img, naive_output, dl_output, lora_output,
                             results_dir=results_dir)
    return scores


if __name__ == "__main__":
    print("Import this module — do not run directly.")
