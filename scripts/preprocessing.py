"""
preprocessing.py — CT slice export, dataset splitting, and path utilities.

Converts DICOM slices to windowed PNG files and creates train/val splits
restricted to anatomically correct (z-filtered) implant slices.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from config import WINDOWS, IMG_SIZE, SEED, PROCESSED_DIR
from dicom_utils import dicom_to_hu, apply_window


def export_slice_as_png(
    ds,
    out_path: str,
    center: float,
    width: float,
    size: int = IMG_SIZE,
) -> None:
    """
    Convert a DICOM slice to a windowed grayscale PNG.

    Metal voxels (HU > 2000) are boosted to pure white (255) so the LoRA
    model learns to associate the implant location with maximum brightness.

    Args:
        ds: pydicom.Dataset for one slice.
        out_path: Destination file path (should end in .png).
        center: Window center in HU.
        width: Window width in HU.
        size: Output image size in pixels (square).
    """
    hu  = dicom_to_hu(ds)
    img = apply_window(hu, center, width)

    metal_mask    = hu > 2000
    metal_resized = np.array(
        Image.fromarray(metal_mask.astype(np.uint8) * 255)
        .resize((size, size), Image.NEAREST)
    ).astype(bool)
    img[metal_resized] = 255

    pil = Image.fromarray(img, mode="L")
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.LANCZOS)
    pil.save(out_path)


def export_all_slices(
    slices: list,
    processed_dir: str = PROCESSED_DIR,
    img_size: int      = IMG_SIZE,
) -> tuple[str, str]:
    """
    Export every DICOM slice as a PNG in both bone and brain HU windows.

    Args:
        slices: List of pydicom.Dataset objects.
        processed_dir: Root directory for processed slice output.
        img_size: Output image size in pixels.

    Returns:
        Tuple of (bone_dir, brain_dir) paths.
    """
    bone_dir  = os.path.join(processed_dir, "bone_window")
    brain_dir = os.path.join(processed_dir, "brain_window")
    os.makedirs(bone_dir,  exist_ok=True)
    os.makedirs(brain_dir, exist_ok=True)

    wc_bone,  ww_bone  = WINDOWS["bone"]
    wc_brain, ww_brain = WINDOWS["brain"]

    for i, ds in enumerate(slices):
        fname = f"slice_{i:04d}.png"
        export_slice_as_png(ds, os.path.join(bone_dir,  fname),
                            wc_bone,  ww_bone,  img_size)
        export_slice_as_png(ds, os.path.join(brain_dir, fname),
                            wc_brain, ww_brain, img_size)

    print(f"Exported {len(slices)} slices × 2 windows")
    print(f"  Bone window:  {bone_dir}")
    print(f"  Brain window: {brain_dir}")
    return bone_dir, brain_dir


def build_dataset_split(
    bone_dir: str,
    implant_idx_set: set,
    test_size: float = 0.2,
    seed: int        = SEED,
) -> dict:
    """
    Create train / val path lists, with implant-only subsets for LoRA.

    Args:
        bone_dir: Directory containing bone-window PNG slices.
        implant_idx_set: Set of slice indices with z-filtered implant signal.
        test_size: Fraction of data to reserve for validation.
        seed: Random seed for the split.

    Returns:
        Dict with keys: all_paths, train_paths, val_paths,
                        train_implant_paths, val_implant_paths.
    """
    all_png = sorted(str(p) for p in Path(bone_dir).glob("*.png"))
    train_paths, val_paths = train_test_split(
        all_png, test_size=test_size, random_state=seed
    )

    def _in_z_range(path: str) -> bool:
        idx = int(Path(path).stem.split("_")[1])
        return idx in implant_idx_set

    train_implant_paths = [p for p in train_paths if _in_z_range(p)]
    val_implant_paths   = [p for p in val_paths   if _in_z_range(p)]

    print(f"Dataset split:")
    print(f"  Total:                      {len(all_png)}")
    print(f"  Train:                      {len(train_paths)}")
    print(f"  Val:                        {len(val_paths)}")
    print(f"  Train implant (z-filtered): {len(train_implant_paths)}")
    print(f"  Val   implant (z-filtered): {len(val_implant_paths)}")

    return {
        "all_paths":           all_png,
        "train_paths":         train_paths,
        "val_paths":           val_paths,
        "train_implant_paths": train_implant_paths,
        "val_implant_paths":   val_implant_paths,
    }


if __name__ == "__main__":
    print("Import this module — do not run directly.")
