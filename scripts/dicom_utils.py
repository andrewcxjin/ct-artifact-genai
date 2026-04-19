"""
dicom_utils.py — DICOM loading, anonymisation, HU conversion, and implant detection.

Provides utilities for loading DICOM series, converting pixel values to
Hounsfield Units, windowing for display, detecting metal implant slices,
and applying a z-position filter to restrict training to skull-base level.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pydicom

from config import (
    WINDOWS, IMG_SIZE,
    METAL_HU_THRESHOLD, MIN_METAL_VOXELS, Z_BUFFER_MM,
    RESULTS_DIR,
)


def load_dicom_series(folder: str) -> list:
    """
    Load all DICOM files in a folder, sorted by InstanceNumber.

    Args:
        folder: Path to directory containing .dcm files.

    Returns:
        List of pydicom.Dataset objects sorted by InstanceNumber.
    """
    datasets = []
    for root, _, files in os.walk(folder):
        for fname in files:
            try:
                ds = pydicom.dcmread(os.path.join(root, fname), force=True)
                if hasattr(ds, "PixelData"):
                    datasets.append(ds)
            except Exception:
                pass

    datasets.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))
    print(f"Loaded {len(datasets)} DICOM slices from {folder}")
    return datasets


def filter_target_series(slices: list, target_series: list) -> list:
    """
    Keep only slices whose SeriesDescription matches one of the target strings.

    Args:
        slices: Full list of DICOM datasets.
        target_series: Allowed SeriesDescription values.

    Returns:
        Filtered list of datasets.
    """
    filtered = [
        ds for ds in slices
        if getattr(ds, "SeriesDescription", "").strip() in
           [s.strip() for s in target_series]
    ]
    print(f"Series filter: {len(slices)} → {len(filtered)} slices")
    return filtered


def anonymize_slice(ds, patient_id: str = "ANON001"):
    """
    Remove PHI tags from a DICOM dataset in-place.

    Args:
        ds: pydicom.Dataset to anonymise.
        patient_id: Replacement patient ID string.

    Returns:
        Modified dataset (same object).
    """
    phi_tags = [
        "PatientName", "PatientID", "PatientBirthDate",
        "PatientSex", "PatientAge", "PatientAddress",
        "InstitutionName", "ReferringPhysicianName",
        "StudyDate", "StudyTime", "AccessionNumber",
    ]
    for tag in phi_tags:
        if hasattr(ds, tag):
            setattr(ds, tag, "")
    ds.PatientID = patient_id
    return ds


def anonymize_series(slices: list, patient_id: str = "ANON001") -> list:
    """
    Anonymise all slices in a series.

    Args:
        slices: List of DICOM datasets.
        patient_id: Replacement patient ID.

    Returns:
        Same list with PHI stripped in-place.
    """
    for ds in slices:
        anonymize_slice(ds, patient_id)
    print(f"Anonymised {len(slices)} slices (patient ID → {patient_id})")
    return slices


def dicom_to_hu(ds) -> np.ndarray:
    """
    Convert a DICOM dataset's pixel array to Hounsfield Units.

    Args:
        ds: pydicom.Dataset with RescaleSlope and RescaleIntercept.

    Returns:
        float32 numpy array of HU values.
    """
    pixels = ds.pixel_array.astype(np.float32)
    slope  = float(getattr(ds, "RescaleSlope",     1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    return pixels * slope + intercept


def apply_window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Apply a HU window and map to [0, 255] uint8.

    Args:
        hu: float32 HU array.
        center: Window center in HU.
        width: Window width in HU.

    Returns:
        uint8 numpy array.
    """
    lo  = center - width / 2
    hi  = center + width / 2
    out = np.clip(hu, lo, hi)
    out = (out - lo) / (hi - lo) * 255
    return out.astype(np.uint8)


def detect_implant_slices(
    slices: list,
    hu_threshold: float = METAL_HU_THRESHOLD,
    min_voxels: int     = MIN_METAL_VOXELS,
) -> tuple:
    """
    Identify slices containing metal implant signal by HU threshold.

    Args:
        slices: List of DICOM datasets.
        hu_threshold: HU value above which a voxel is classified as metal.
        min_voxels: Minimum number of metal voxels to flag a slice.

    Returns:
        Tuple of:
            implant_slices   — list of datasets flagged as implant
            implant_idx_set  — set of indices into slices
            implant_scores   — list of (index, metal_voxel_count)
    """
    implant_slices  = []
    implant_idx_set = set()
    implant_scores  = []

    for i, ds in enumerate(slices):
        hu    = dicom_to_hu(ds)
        count = int((hu > hu_threshold).sum())
        if count >= min_voxels:
            implant_slices.append(ds)
            implant_idx_set.add(i)
            implant_scores.append((i, count))

    print(f"Implant detection: {len(implant_idx_set)} / {len(slices)} slices "
          f"(threshold={hu_threshold} HU, min={min_voxels} voxels)")
    return implant_slices, implant_idx_set, implant_scores


def filter_implants_by_z(
    slices: list,
    implant_slices: list,
    z_buffer_mm: float = Z_BUFFER_MM,
) -> set:
    """
    Restrict implant slices to those within z_buffer_mm of the implant z-cluster.

    Uses ImagePositionPatient[2] to compute z-positions and keeps only slices
    within [z_min - buffer, z_max + buffer] of the detected implant range.

    Args:
        slices: Full list of DICOM datasets (indexed).
        implant_slices: Datasets already flagged as containing implant signal.
        z_buffer_mm: Half-width of the z-range window (millimetres).

    Returns:
        Set of slice indices (into slices) that pass the z filter.
    """
    def _z(ds) -> float:
        pos = getattr(ds, "ImagePositionPatient", None)
        return float(pos[2]) if pos is not None else 0.0

    if not implant_slices:
        return set()

    z_vals  = [_z(ds) for ds in implant_slices]
    z_min   = min(z_vals) - z_buffer_mm
    z_max   = max(z_vals) + z_buffer_mm

    filtered = {
        i for i, ds in enumerate(slices)
        if z_min <= _z(ds) <= z_max
    }
    print(f"Z filter (±{z_buffer_mm} mm): {len(filtered)} slices kept "
          f"[z ∈ {z_min:.1f}, {z_max:.1f}]")
    return filtered


def print_series_info(slices: list) -> None:
    """Print a summary of series descriptions and slice counts."""
    from collections import Counter
    counts = Counter(
        getattr(ds, "SeriesDescription", "Unknown").strip()
        for ds in slices
    )
    print(f"\nLoaded {len(slices)} slices across {len(counts)} series:")
    for desc, n in counts.most_common():
        print(f"  {n:4d}  {desc}")


def plot_scan_overview(
    slices: list,
    implant_idx_set: set,
    n_samples: int   = 6,
    results_dir: str = RESULTS_DIR,
) -> None:
    """
    Plot a grid of evenly-spaced bone-window slices, flagging implant slices.

    Args:
        slices: Full DICOM slice list.
        implant_idx_set: Set of implant slice indices.
        n_samples: Number of sample slices to display.
        results_dir: Directory to save the figure.
    """
    wc, ww  = WINDOWS["bone"]
    step    = max(1, len(slices) // n_samples)
    indices = list(range(0, len(slices), step))[:n_samples]

    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    for ax, idx in zip(axes, indices):
        hu  = dicom_to_hu(slices[idx])
        img = apply_window(hu, wc, ww)
        ax.imshow(img, cmap="gray")
        color = "red" if idx in implant_idx_set else "white"
        ax.set_title(f"Slice {idx}", color=color, fontsize=9)
        ax.axis("off")

    fig.suptitle("Scan Overview (red = implant detected)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "scan_overview.png"), dpi=150)
    plt.show()


def plot_implant_signal(
    slices: list,
    implant_scores: list,
    implant_idx_set: set,
    results_dir: str = RESULTS_DIR,
) -> None:
    """
    Plot metal voxel count per slice to visualise the implant z-profile.

    Args:
        slices: Full DICOM slice list.
        implant_scores: List of (index, count) from detect_implant_slices.
        implant_idx_set: Z-filtered set for colour coding.
        results_dir: Directory to save the figure.
    """
    score_dict = dict(implant_scores)
    xs = list(score_dict.keys())
    ys = list(score_dict.values())
    colors = ["limegreen" if x in implant_idx_set else "salmon" for x in xs]

    plt.figure(figsize=(12, 3))
    plt.bar(xs, ys, color=colors, width=1.0)
    plt.xlabel("Slice index"); plt.ylabel("Metal voxels (HU > 2000)")
    plt.title("Implant signal per slice  (green = z-filtered training set)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "implant_signal.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Import this module — do not run directly.")
