# CochleArt — CT Artifact Generation

An interactive proof-of-concept for synthesising cochlear-implant metal artifacts
in CT scans via three complementary approaches: physics simulation, sparse
dictionary learning, and LoRA-fine-tuned diffusion.

---

## Project structure

```
├── app.py                      ← Flask web application (entry point)
├── setup.py                    ← Project initialisation (create dirs, validate data)
├── requirements.txt            ← Python dependencies
├── runtime.txt                 ← Python version pin (Railway/Nixpacks)
├── Procfile                    ← Gunicorn start command
├── railway.toml                ← Railway deployment config
├── LICENSE
├── .gitignore
├── templates/
│   └── index.html              ← Dark luxury front-end
├── static/
│   ├── css/style.css
│   └── js/main.js
├── scripts/
│   ├── config.py               ← Central hyperparameters
│   ├── naive_baseline.py       ← Model 1: physics ray-casting
│   ├── dict_learning.py        ← Model 2: sparse dictionary learning
│   ├── lora_model.py           ← Model 3: LoRA fine-tuning & inference
│   ├── build_features.py       ← DICOM → PNG preprocessing pipeline
│   ├── preprocessing.py        ← Windowing, PNG export, train/val split
│   ├── dicom_utils.py          ← DICOM loading, HU conversion, implant detection
│   ├── cochlearart_dataset.py  ← PyTorch dataset for CT slices
│   ├── predict.py              ← Training orchestrator (CLI)
│   └── evaluation.py           ← SSIM metrics & comparison plots
├── experiments/
│   ├── experiments.py          ← Sensitivity sweeps & LoRA rank ablation
│   ├── experiment_summary.json ← Aggregated experiment results
│   ├── dl_ssim_grid.npy        ← Dictionary-learning SSIM sweep
│   └── lora_losses_rank{4,8,16}.npy ← LoRA rank-ablation loss curves
├── models/
│   ├── naive_baseline/config.json
│   ├── dict_learning/
│   │   ├── atoms.npy           ← 128 learned CT texture atoms (16×16)
│   │   ├── ssim_grid.npy       ← Cached SSIM grid from sweeps
│   │   └── config.json
│   └── lora/
│       ├── adapter_config.json ← PEFT LoRA config (rank 16, SD v1.5)
│       └── adapter_model.safetensors ← 12.8 MB adapter weights
└── data/
    └── raw/                    ← DICOM source files (e.g. S0000001/)
```

---

## Quick start (local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> GPU strongly recommended for the LoRA model. CPU works for Physics and
> Dictionary Learning.

### 2. Run the Flask app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 3. Use the interface

| Model | Speed | Notes |
|---|---|---|
| Physics Simulation | ~1–2 s | Always available; adjust sliders |
| Dictionary Learning | ~5–10 s | Requires `models/dict_learning/atoms.npy` |
| LoRA Diffusion | ~30–90 s | Downloads SD v1.5 base (~4 GB) on first run |

---

## Running the full pipeline (optional)

To retrain models from your own DICOM data:

```bash
# 1. Build features (DICOM → PNG)
DICOM_ROOT=/path/to/dicoms python scripts/build_features.py

# 2. Train all three models + evaluate
python scripts/predict.py

# 3. Sensitivity experiments
python scripts/predict.py --experiments
```

---

## Deployment on Railway

This repo ships with everything Railway's Nixpacks builder needs:

- `runtime.txt` pins Python 3.11
- `requirements.txt` declares dependencies
- `Procfile` / `railway.toml` start Gunicorn: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 600`
- `app.py` reads `PORT` from the environment
- Commit `models/` (or mount as a volume) so weights are available at runtime
- The LoRA pipeline is cached after first load — keep at least one instance
  warm to avoid cold-start latency

---

## Models

### Model 1 · Physics Simulation
Deterministic ray-casting inserts a high-HU titanium ellipse and adds
beam-hardening streaks via exponentially-decaying alternating bright/dark rays.
No training data required.

### Model 2 · Dictionary Learning
`MiniBatchDictionaryLearning` (scikit-learn) was trained on overlapping 16 × 16
patches from real cochlear-implant CT slices, learning 128 texture atoms.
Inference uses Orthogonal Matching Pursuit (k = 6) for sparse coding.

### Model 3 · LoRA Diffusion
Stable Diffusion v1.5 UNet fine-tuned for 1 500 steps on z-filtered implant
slices via PEFT LoRA (rank 16, α 32, targeting `to_q / to_k / to_v / to_out.0`).
Inference runs img2img: a synthetic skull phantom seeds the latent and the
text prompt steers artifact characteristics.

---

## Dependencies

Core: `numpy`, `Pillow`, `matplotlib`, `scikit-learn`, `scikit-image`  
Medical: `pydicom`  
Deep learning: `torch`, `torchvision`, `diffusers`, `transformers`, `peft`, `accelerate`  
Web: `flask`, `gunicorn`
