"""
config.py — Central configuration

All paths, hyperparameters, and constants live here so that every
other module imports from a single source of truth.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# Override DICOM_ROOT at runtime: DICOM_ROOT=/path/to/dicom python predict.py
DICOM_ROOT    = os.environ.get("DICOM_ROOT", "/content/S0000001")
DATA_DIR      = os.environ.get("DATA_DIR",   "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR    = os.path.join(DATA_DIR, "output")
RESULTS_DIR   = os.path.join(OUTPUT_DIR, "results")
MODELS_DIR    = "models"
LORA_SAVE_PATH = os.path.join(MODELS_DIR, "lora_weights")

# ── DICOM series to keep (axial only) ────────────────────────────────────────
TARGET_SERIES = [
    "BONE                  DFOV 17",
    "PRE OP SINUS MEDTRONICS",
]

# ── HU windowing ─────────────────────────────────────────────────────────────
WINDOWS = {
    "bone":  (500,  3000),
    "brain": (40,   80),
}

# ── Image processing ──────────────────────────────────────────────────────────
IMG_SIZE           = 512
METAL_HU_THRESHOLD = 2000   # voxels above this are treated as metal
MIN_METAL_VOXELS   = 10     # minimum voxels to flag a slice as implant
Z_BUFFER_MM        = 8      # ± mm around implant z-cluster for training

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Naive baseline hyperparameters ───────────────────────────────────────────
NAIVE_STREAK_INTENSITY = 0.35
NAIVE_N_ANGLES         = 24
NAIVE_STREAK_WIDTH     = 3
NAIVE_NOISE_STD        = 0.04

# ── Dictionary Learning hyperparameters ──────────────────────────────────────
DL_PATCH_SIZE    = 16
DL_N_ATOMS       = 128
DL_N_NONZERO     = 6
DL_N_ITER        = 300
DL_MAX_IMGS      = 30
DL_MAX_PATCHES   = 8_000
DL_RECON_STRIDE  = 4

# ── LoRA / Stable Diffusion hyperparameters ───────────────────────────────────
BASE_MODEL_ID    = "runwayml/stable-diffusion-v1-5"
LORA_RANK        = 16
LORA_ALPHA       = 32
LORA_TARGET_MODS = ["to_q", "to_k", "to_v", "to_out.0"]
LORA_DROPOUT     = 0.05
LORA_TRAIN_STEPS = 1500
LORA_LR          = 5e-5
LORA_BATCH_SIZE  = 1
LORA_IMG_SIZE    = 256

TRAIN_PROMPT = (
    "axial head CT scan at skull base level, petrous temporal bone, "
    "cochlear implant with electrode array, bone window, "
    "bright white hyperdense metal artifact, beam-hardening streak artifacts, "
    "high-resolution grayscale HRCT"
)

NEGATIVE_PROMPT = (
    "color photograph, MRI, sinus, nasal cavity, "
    "paranasal sinuses, nose, orbit, no implant, artifact-free"
)

# ── Inference parameters ──────────────────────────────────────────────────────
INFERENCE_STEPS    = 50
INFERENCE_GUIDANCE = 13.0
INFERENCE_STRENGTH = 0.55
INFERENCE_SEED     = 30

# ── Experiment 1: DL sensitivity sweep ───────────────────────────────────────
ATOM_GRID         = [32, 64, 128, 256]
NONZERO_GRID      = [4, 8, 12]
SWEEP_MAX_IMGS    = 20
SWEEP_MAX_PATCHES = 5_000
SWEEP_N_ITER      = 200

# ── Experiment 2: LoRA rank ablation ─────────────────────────────────────────
ABLATION_RANKS = [4, 8, 16]
