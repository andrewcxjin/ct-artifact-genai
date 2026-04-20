"""
app.py — CochleArt Web Application

Flask backend for the CochleArt interactive CT artifact generation demo.
Supports three inference modes (no training):
  1. Physics Simulation  — deterministic ray-casting (Naive)
  2. Dictionary Learning — sparse reconstruction from learned atoms
  3. LoRA Diffusion      — Stable Diffusion fine-tuned on cochlear CT
"""

import contextlib
import io
import os
import sys
import base64
import threading
import time
import urllib.request
import uuid
from pathlib import Path

# If set, inference is routed to Replicate instead of running locally.
# Set this env var on Railway once the model is pushed to Replicate.
# Format: "owner/model-name" e.g. "andrewcxjin/cochle-art"
REPLICATE_MODEL = os.environ.get("REPLICATE_MODEL")

import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))

app = Flask(__name__)


# ── Real CT loader ────────────────────────────────────────────────────────────

# Cache: slices + implant detection results so we don't redo this every request
_ct_data: dict | None = None


def _get_ct_data() -> dict:
    """
    Load all DICOM slices from data/raw/ and run the same implant-detection
    pipeline used during training:
      1. Load + sort by InstanceNumber
      2. detect_implant_slices  (HU > 2000, min 10 voxels)
      3. filter_implants_by_z   (±8 mm around implant cluster)

    Returns a dict with keys:
        slices          — full sorted list of DICOM datasets
        implant_idx_set — set of indices with detected metal (pre z-filter)
        z_filtered_idx  — set of indices passing the z-filter (skull-base level)
        implant_scores  — list of (index, metal_voxel_count)
    """
    global _ct_data
    if _ct_data is not None:
        return _ct_data

    from dicom_utils import (
        load_dicom_series, detect_implant_slices, filter_implants_by_z,
    )

    raw_dir = str(ROOT / "data" / "raw")
    slices  = load_dicom_series(raw_dir)
    if not slices:
        raise ValueError(
            "No DICOM data found in data/raw/. "
            "Ensure your DICOM files are present and re-run."
        )

    implant_slices, implant_idx_set, implant_scores = detect_implant_slices(slices)
    z_filtered_idx = filter_implants_by_z(slices, implant_slices)

    # Fallback: if no implant detected, use entire series
    if not z_filtered_idx:
        z_filtered_idx = set(range(len(slices)))
        print("Warning: no implant slices detected — using full series")

    _ct_data = {
        "slices":          slices,
        "implant_idx_set": implant_idx_set,
        "z_filtered_idx":  z_filtered_idx,
        "implant_scores":  implant_scores,
    }
    print(f"CT data ready: {len(slices)} total, "
          f"{len(implant_idx_set)} implant, "
          f"{len(z_filtered_idx)} z-filtered")
    return _ct_data


def load_real_ct_slice(seed: int = 42, prefer_implant: bool = True) -> np.ndarray:
    """
    Return a bone-windowed uint8 CT slice at skull-base / cochlear implant level.

    prefer_implant=True  → picks from the slices with highest metal signal
                           (used for LoRA img2img, matches training distribution)
    prefer_implant=False → picks a z-filtered but non-implant slice
                           (used for Naive/Dict, gives a clean starting point)
    """
    from dicom_utils import dicom_to_hu, apply_window
    from config import WINDOWS

    data   = _get_ct_data()
    slices = data["slices"]

    if prefer_implant and data["implant_scores"]:
        # Sort by metal voxel count descending; rotate with seed for variety
        ranked = sorted(data["implant_scores"], key=lambda x: x[1], reverse=True)
        pool   = [slices[idx] for idx, _ in ranked[:max(1, len(ranked) // 2)]]
    else:
        # Z-filtered slices that are NOT bright-metal (clean skull-base anatomy)
        clean_idx = data["z_filtered_idx"] - data["implant_idx_set"]
        if not clean_idx:
            clean_idx = data["z_filtered_idx"]
        pool = [slices[i] for i in sorted(clean_idx)]

    ds       = pool[seed % len(pool)]
    hu       = dicom_to_hu(ds)
    windowed = apply_window(hu, *WINDOWS["bone"])
    return np.array(Image.fromarray(windowed).resize((256, 256)))


# ── Inference helpers ─────────────────────────────────────────────────────────

def encode_image(arr: np.ndarray) -> str:
    """Convert a uint8 numpy array to a base64-encoded PNG string."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run_naive(base_img: np.ndarray, params: dict, seed: int) -> tuple:
    """Physics-based ray-casting artifact simulation (no training required)."""
    from naive_baseline import NaiveArtifactSimulator

    sim = NaiveArtifactSimulator(
        streak_intensity=float(params.get("streak_intensity", 0.35)),
        n_angles=int(params.get("n_angles", 24)),
        streak_width=int(params.get("streak_width", 3)),
        noise_std=float(params.get("noise_std", 0.04)),
        seed=seed,
    )
    return sim.simulate(base_img), {}


def run_dict(base_img: np.ndarray) -> tuple:
    """Sparse reconstruction from the pre-trained dictionary atoms."""
    atoms_path = ROOT / "models" / "dict_learning" / "atoms.npy"
    if not atoms_path.exists():
        return None, {"error": "Dictionary atoms not found. Run setup.py first."}

    from sklearn.decomposition import sparse_encode
    from skimage.metrics import structural_similarity as ssim

    atoms  = np.load(str(atoms_path))   # (128, 256) — 128 atoms × 16×16 patches
    ps, stride, n_nz = 16, 8, 6

    # Reconstruct at 128×128 for acceptable latency
    small  = np.array(Image.fromarray(base_img).resize((128, 128)))
    h, w   = small.shape
    output = np.zeros((h, w), dtype=np.float32)
    count  = np.zeros((h, w), dtype=np.float32)

    for row in range(0, h - ps + 1, stride):
        for col in range(0, w - ps + 1, stride):
            patch = small[row:row + ps, col:col + ps].ravel().astype(np.float32) / 255.0
            code  = sparse_encode(patch.reshape(1, -1), atoms,
                                  algorithm="omp", n_nonzero_coefs=n_nz)
            recon = (code @ atoms).reshape(ps, ps)
            output[row:row + ps, col:col + ps] += recon
            count[row:row + ps,  col:col + ps] += 1

    count       = np.where(count == 0, 1, count)
    recon_small = np.clip(output / count * 255, 0, 255).astype(np.uint8)
    result      = np.array(
        Image.fromarray(recon_small).resize((base_img.shape[1], base_img.shape[0]))
    )

    score = ssim(base_img, result, data_range=255)
    return result, {"ssim": f"{score:.3f}"}


# Cache the heavy LoRA pipeline to avoid re-loading on every request
_lora_pipe_cache: tuple | None = None


def _load_lora_pipeline():
    global _lora_pipe_cache
    if _lora_pipe_cache is not None:
        return _lora_pipe_cache

    try:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            f"Missing package '{exc.name}'. "
            "Run: pip install diffusers transformers peft accelerate"
        ) from exc

    lora_dir = ROOT / "models" / "lora"
    if not (lora_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(
            "LoRA adapter not found at models/lora/. "
            "Ensure adapter_model.safetensors and adapter_config.json exist."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading Stable Diffusion base model on {device} (first run may take a few minutes)…")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    print("Applying LoRA adapter…")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_dir))
    pipe      = pipe.to(device)
    pipe.enable_attention_slicing()

    _lora_pipe_cache = (pipe, device)
    print(f"LoRA pipeline ready on {device}")
    return _lora_pipe_cache


def run_lora(base_img: np.ndarray, prompt: str, seed: int) -> tuple:
    """LoRA img2img inference using the fine-tuned Stable Diffusion adapter."""
    try:
        import torch
        from config import (
            INFERENCE_GUIDANCE, INFERENCE_STEPS, INFERENCE_STRENGTH,
            NEGATIVE_PROMPT,
        )

        pipe, device = _load_lora_pipeline()
        pil_in = Image.fromarray(base_img).convert("RGB").resize((256, 256))

        # CPU inference is ~150 s/step; cap at 4 steps to keep wait under 15 min.
        steps = INFERENCE_STEPS if device == "cuda" else 4

        ctx = torch.autocast(device) if device == "cuda" else contextlib.nullcontext()
        with ctx:
            out = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=pil_in,
                strength=INFERENCE_STRENGTH,
                num_inference_steps=steps,
                guidance_scale=INFERENCE_GUIDANCE,
                generator=torch.Generator(device).manual_seed(seed),
            )
        return np.array(out.images[0].convert("L")), {}

    except Exception as exc:
        return None, {"error": str(exc)}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    """Return which models are available given the current file state."""
    return jsonify({
        "naive": True,
        "dict":  (ROOT / "models" / "dict_learning" / "atoms.npy").exists(),
        "lora":  (ROOT / "models" / "lora" / "adapter_model.safetensors").exists(),
    })


# ── Async job queue ───────────────────────────────────────────────────────────
# Keyed by job_id; values: {status, output_image, model, metrics, error, started_at}
_jobs: dict = {}


def _run_replicate_job(job_id: str, data: dict) -> None:
    """Background thread: call Replicate API and write result into _jobs."""
    try:
        import replicate

        model_type = data.get("model", "naive")
        seed       = int(data.get("seed", 42))
        prompt     = data.get("prompt", "").strip()
        params     = data.get("params", {})

        inp = {"model": model_type, "seed": seed, "prompt": prompt}
        if model_type == "naive":
            inp["streak_intensity"] = float(params.get("streak_intensity", 0.35))
            inp["noise_std"]        = float(params.get("noise_std", 0.04))
            inp["n_angles"]         = int(params.get("n_angles", 24))

        # Resolve to a specific version to avoid 404s with versionless API calls
        if ":" not in REPLICATE_MODEL:
            model      = replicate.models.get(REPLICATE_MODEL)
            version_id = model.latest_version.id
            model_ref  = f"{REPLICATE_MODEL}:{version_id}"
            print(f"Replicate: resolved to {model_ref}", flush=True)
        else:
            model_ref = REPLICATE_MODEL

        output = replicate.run(model_ref, input=inp)
        print(f"Replicate: raw output type={type(output)} value={output!r}", flush=True)

        # output may be a URL string, a FileOutput, or a list — normalise it
        url = output[0] if isinstance(output, list) else output
        img_data = urllib.request.urlopen(str(url)).read()

        _jobs[job_id] = {
            "status":       "done",
            "output_image": base64.b64encode(img_data).decode(),
            "model":        model_type,
            "metrics":      {},
        }
    except Exception as exc:
        import traceback
        print(f"Replicate job {job_id} failed: {exc}", flush=True)
        traceback.print_exc()
        _jobs[job_id] = {"status": "error", "error": str(exc)}


def _run_job(job_id: str, data: dict) -> None:
    """Background thread: run inference and write result into _jobs."""
    model_type = data.get("model", "naive")
    prompt     = data.get("prompt", "").strip()
    params     = data.get("params", {})
    seed       = int(data.get("seed", 42))

    try:
        prefer_implant = (model_type == "lora")
        base_img = load_real_ct_slice(seed=seed % 97, prefer_implant=prefer_implant)
    except ValueError as exc:
        _jobs[job_id] = {"status": "error", "error": str(exc)}
        return

    try:
        if model_type == "naive":
            result, meta = run_naive(base_img, params, seed)
        elif model_type == "dict":
            result, meta = run_dict(base_img)
        elif model_type == "lora":
            if not prompt:
                from config import TRAIN_PROMPT
                prompt = TRAIN_PROMPT
            result, meta = run_lora(base_img, prompt, seed)
        else:
            _jobs[job_id] = {"status": "error", "error": "Unknown model type"}
            return

        if result is None:
            _jobs[job_id] = {"status": "error",
                             "error": meta.get("error", "Generation failed")}
            return

        _jobs[job_id] = {
            "status":       "done",
            "output_image": encode_image(result),
            "model":        model_type,
            "metrics":      meta,
        }
    except Exception as exc:
        _jobs[job_id] = {"status": "error", "error": str(exc)}


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Start an async generation job. Returns immediately with a job_id.
    Poll /api/job/<job_id> for status and result.
    """
    data     = request.get_json(force=True)
    job_id   = uuid.uuid4().hex[:10]
    _jobs[job_id] = {"status": "running", "started_at": time.time()}

    model_type = data.get("model", "naive")
    use_replicate = REPLICATE_MODEL and model_type == "lora"
    target = _run_replicate_job if use_replicate else _run_job
    t = threading.Thread(target=target, args=(job_id, data), daemon=True)
    t.start()

    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/job/<job_id>")
def job_status(job_id: str):
    """Poll for job result. Returns status + result when done."""
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404

    resp = {"status": job["status"]}
    if job["status"] == "running":
        resp["elapsed"] = round(time.time() - job.get("started_at", time.time()))
    elif job["status"] == "done":
        resp.update({
            "success":      True,
            "output_image": job["output_image"],
            "model":        job["model"],
            "metrics":      job["metrics"],
        })
        # Clean up to free memory
        del _jobs[job_id]
    elif job["status"] == "error":
        resp["error"] = job.get("error", "Unknown error")
        del _jobs[job_id]

    return jsonify(resp)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
