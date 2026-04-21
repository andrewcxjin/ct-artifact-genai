# CochleArt — Synthetic CT Artifact Generation

An interactive web application for synthesising cochlear-implant metal artifacts in CT scans. Three complementary approaches are demonstrated side by side: deterministic physics simulation, sparse dictionary learning, and LoRA-fine-tuned diffusion.

**Live demo → [web-production-58def.up.railway.app](https://web-production-58def.up.railway.app)**

---

## What it does

Cochlear implant CT scans exhibit characteristic metal artifacts — bright hyperdense streaks, beam-hardening shadows, and photon-starvation noise — that complicate post-operative imaging. CochleArt lets you generate synthetic versions of these artifacts using three distinct techniques, with real skull-base DICOM data as the anatomical foundation.

---

## Models

### Physics Simulation
Deterministic ray-casting places a high-HU titanium ellipse and emits beam-hardening streaks via exponentially-decaying alternating bright/dark rays across configurable angles. No training required. Controllable via streak intensity, noise level, and streak angle sliders.

### Dictionary Learning
A `MiniBatchDictionaryLearning` dictionary (scikit-learn) trained on overlapping 16 × 16 patches from real cochlear-implant CT slices, learning 128 texture atoms. Inference reconstructs slices patch-by-patch using Orthogonal Matching Pursuit (k = 6 non-zero coefficients).

### LoRA Diffusion
Stable Diffusion v1.5 UNet fine-tuned for 1,500 steps on z-filtered implant slices via PEFT LoRA (rank 16, α 32, targeting `to_q / to_k / to_v / to_out.0`). Inference runs img2img: a real CT slice seeds the latent and a text prompt steers artifact characteristics. GPU inference runs on Replicate.

---

## Tech stack

| Layer | Technology |
|---|---|
| Web frontend | Flask + Jinja2, vanilla JS |
| Hosting | Railway (CPU) |
| GPU inference | Replicate (T4) |
| Physics model | NumPy |
| Dictionary model | scikit-learn, scikit-image |
| Diffusion model | Stable Diffusion v1.5, diffusers, PEFT |
| Medical imaging | pydicom |

---

## Repository structure

```
├── app.py                      ← Flask web application (entry point)
├── predict.py                  ← Cog predictor for Replicate deployment
├── cog.yaml                    ← Replicate container config
├── requirements.txt            ← Python dependencies
├── Procfile                    ← Gunicorn start command (Railway)
├── railway.toml                ← Railway deployment config
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
│   ├── dicom_utils.py          ← DICOM loading, HU conversion, implant detection
│   ├── build_features.py       ← DICOM → PNG preprocessing pipeline
│   ├── cochlearart_dataset.py  ← PyTorch dataset for CT slices
│   └── evaluation.py           ← SSIM metrics & comparison plots
├── models/
│   ├── dict_learning/
│   │   └── atoms.npy           ← 128 learned CT texture atoms (16×16)
│   └── lora/
│       ├── adapter_config.json ← PEFT LoRA config (rank 16, SD v1.5)
│       └── adapter_model.safetensors ← LoRA adapter weights
├── experiments/
│   ├── experiments.py          ← Sensitivity sweeps & LoRA rank ablation
│   └── experiment_summary.json ← Aggregated results
└── data/
    └── raw/                    ← DICOM source files
```

---

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000**. Physics Simulation and Dictionary Learning run on CPU. LoRA Diffusion requires a GPU or a [Replicate](https://replicate.com) API token set as `REPLICATE_MODEL=andrewcxjin/cochle-art`.
