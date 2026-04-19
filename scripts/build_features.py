"""
build_features.py — Preprocessing pipeline.

Loads the DICOM series, detects cochlear implant slices (with z-position
filtering), and exports bone/brain window PNGs to data/processed/.

Run this once after setup.py and before predict.py.

Usage:
    python build_features.py
"""

import os

from config import (
    DICOM_ROOT, PROCESSED_DIR, RESULTS_DIR,
    TARGET_SERIES, METAL_HU_THRESHOLD, MIN_METAL_VOXELS, Z_BUFFER_MM, SEED,
)


def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR,   exist_ok=True)

    from dicom_utils import (
        load_dicom_series, filter_target_series, anonymize_series,
        detect_implant_slices, filter_implants_by_z,
        print_series_info, plot_scan_overview, plot_implant_signal,
    )
    from preprocessing import export_all_slices, build_dataset_split

    print("=" * 60)
    print("CochleArt — Build Features (DICOM → PNG)")
    print("=" * 60)

    dicom_root = os.environ.get("DICOM_ROOT", DICOM_ROOT)
    print(f"\n[1/4] Loading DICOM series from: {dicom_root}")
    slices = load_dicom_series(dicom_root)
    slices = filter_target_series(slices, TARGET_SERIES)
    slices = anonymize_series(slices)
    print_series_info(slices)

    print("\n[2/4] Detecting implant slices...")
    implant_slices, implant_idx_set_all, implant_scores = detect_implant_slices(
        slices, hu_threshold=METAL_HU_THRESHOLD, min_voxels=MIN_METAL_VOXELS,
    )
    implant_idx_set = filter_implants_by_z(slices, implant_slices, Z_BUFFER_MM)

    plot_scan_overview(slices, implant_idx_set, results_dir=RESULTS_DIR)
    plot_implant_signal(slices, implant_scores, implant_idx_set, results_dir=RESULTS_DIR)

    print("\n[3/4] Exporting PNG slices...")
    bone_dir, _ = export_all_slices(slices, processed_dir=PROCESSED_DIR)

    print("\n[4/4] Building train / val split...")
    split = build_dataset_split(bone_dir, implant_idx_set, seed=SEED)

    print("\n" + "=" * 60)
    print("build_features.py complete.")
    print(f"  Processed slices: {PROCESSED_DIR}")
    print(f"  Train:            {len(split['train_paths'])} slices "
          f"({len(split['train_implant_paths'])} implant)")
    print(f"  Val:              {len(split['val_paths'])} slices "
          f"({len(split['val_implant_paths'])} implant)")
    print("Run: python predict.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
