"""
setup.py — CochleArt project setup script.

Creates the required directory structure and validates that the DICOM source
is accessible before any expensive processing begins. Run this once before
build_features.py or predict.py.

Usage:
    python setup.py
    DICOM_ROOT=/path/to/dicom python setup.py
"""

import os
import sys


def main() -> None:
    from config import (
        DICOM_ROOT, DATA_DIR, PROCESSED_DIR, OUTPUT_DIR,
        RESULTS_DIR, MODELS_DIR, LORA_SAVE_PATH,
    )

    print("=" * 60)
    print("CochleArt — Project Setup")
    print("=" * 60)

    dirs = [
        DATA_DIR,
        os.path.join(DATA_DIR, "raw"),
        PROCESSED_DIR,
        OUTPUT_DIR,
        RESULTS_DIR,
        MODELS_DIR,
        LORA_SAVE_PATH,
        "notebooks",
    ]

    print("\nCreating directory structure...")
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  {'created' if not os.path.existed_before else 'exists ':7s}  {d}")

    print("\nValidating DICOM source...")
    dicom_root = os.environ.get("DICOM_ROOT", DICOM_ROOT)
    if os.path.isdir(dicom_root):
        n_files = sum(len(fs) for _, _, fs in os.walk(dicom_root))
        print(f"  OK  {dicom_root}  ({n_files} files)")
    else:
        print(f"  WARNING: DICOM_ROOT not found: {dicom_root}")
        print("  Set the DICOM_ROOT environment variable or update config.py")
        print("  before running build_features.py.")

    print("\nSetup complete. Next steps:")
    print("  1. python build_features.py   # export PNGs from DICOM")
    print("  2. python predict.py          # train all three models")
    print("  3. python predict.py --eval   # run evaluation")
    print("  4. python predict.py --experiments  # run experiments")
    print("  5. python predict.py --error-analysis  # misprediction cases")


if __name__ == "__main__":
    # Patch: track newly created dirs for the status message
    _orig_makedirs = os.makedirs
    _created = set()

    def _makedirs(path, **kw):
        exists = os.path.isdir(path)
        _orig_makedirs(path, **kw)
        if not exists:
            _created.add(path)

    os.makedirs = _makedirs
    os.path.__dict__["existed_before"] = False  # unused sentinel; status printed inline

    main()
