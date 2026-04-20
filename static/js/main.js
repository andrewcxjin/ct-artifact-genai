/* ─────────────────────────────────────────────────────────────────────────────
   CochleArt — main.js
   Handles: model selection, parameter controls, prompt presets,
            API calls, result rendering, nav scroll behaviour.
───────────────────────────────────────────────────────────────────────────── */

// ── Preset prompts ────────────────────────────────────────────────────────────
const PRESETS = {
  standard: (
    "axial head CT scan at skull base level, petrous temporal bone, " +
    "cochlear implant with electrode array, bone window, " +
    "bright white hyperdense metal artifact, beam-hardening streak artifacts, " +
    "high-resolution grayscale HRCT"
  ),
  severe: (
    "cochlear implant axial CT, severe beam-hardening, " +
    "strong dark and bright streak artifacts radiating from dense titanium electrode, " +
    "photon-starvation noise, skull base level, bone window HRCT"
  ),
  subtle: (
    "axial CT skull base, mild cochlear implant artifact, " +
    "faint beam-hardening streaks, low-intensity metal artifact, " +
    "clean bone window, high-resolution temporal bone HRCT"
  ),
  custom: ""
};

const MODEL_LABELS = {
  naive: "Physics Simulation",
  dict:  "Dictionary Learning",
  lora:  "LoRA Diffusion"
};

const LOADING_LABELS = {
  naive: "Casting rays…",
  dict:  "Encoding patches…",
  lora:  "Diffusing latents…"
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const nav           = document.getElementById("nav");
const modelCards    = document.getElementById("modelCards");
const promptGroup   = document.getElementById("promptGroup");
const paramsGroup   = document.getElementById("paramsGroup");
const promptInput   = document.getElementById("promptInput");
const presetRow     = document.getElementById("presetRow");
const seedInput     = document.getElementById("seedInput");
const seedRandom    = document.getElementById("seedRandom");
const generateBtn   = document.getElementById("generateBtn");
const btnText       = generateBtn.querySelector(".generate-btn__text");

const outputPlaceholder = document.getElementById("outputPlaceholder");
const outputResult      = document.getElementById("outputResult");
const outputError       = document.getElementById("outputError");
const outputLoading     = document.getElementById("outputLoading");
const loadingLabel      = document.getElementById("loadingLabel");

const imgOutput   = document.getElementById("imgOutput");
const outputLabel = document.getElementById("outputLabel");
const metaModel   = document.getElementById("metaModel");
const metaSeed    = document.getElementById("metaSeed");
const metaSSIM    = document.getElementById("metaSSIM");

// Sliders
const sliders = {
  streakIntensity: document.getElementById("streakIntensity"),
  noiseStd:        document.getElementById("noiseStd"),
  nAngles:         document.getElementById("nAngles"),
};
const sliderVals = {
  streakIntensity: document.getElementById("valStreakIntensity"),
  noiseStd:        document.getElementById("valNoiseStd"),
  nAngles:         document.getElementById("valNAngles"),
};

// ── State ─────────────────────────────────────────────────────────────────────
let selectedModel  = "naive";
let activePreset   = "standard";
let isGenerating   = false;

// ── Nav scroll behaviour ──────────────────────────────────────────────────────
window.addEventListener("scroll", () => {
  nav.classList.toggle("scrolled", window.scrollY > 40);
}, { passive: true });

// ── Model availability check ──────────────────────────────────────────────────
async function checkStatus() {
  try {
    const res  = await fetch("/api/status");
    const data = await res.json();

    const dictCard = document.getElementById("dictCard");
    const loraCard = document.getElementById("loraCard");

    if (!data.dict && dictCard) {
      dictCard.classList.add("unavailable");
      dictCard.title = "Dictionary atoms not found (run setup.py first)";
      dictCard.querySelector(".model-card__badge").textContent = "Unavailable";
    }
    if (!data.lora && loraCard) {
      loraCard.title = "LoRA adapter found — base SD model will download on first use (~4 GB)";
    }
  } catch (_) {
    // Status check is non-critical
  }
}
checkStatus();

// ── Model card selection ──────────────────────────────────────────────────────
modelCards.addEventListener("click", (e) => {
  const card = e.target.closest(".model-card");
  if (!card || card.classList.contains("unavailable")) return;

  document.querySelectorAll(".model-card").forEach(c => {
    c.classList.remove("active");
    c.setAttribute("aria-pressed", "false");
  });
  card.classList.add("active");
  card.setAttribute("aria-pressed", "true");

  selectedModel = card.dataset.model;
  updateControlVisibility();
});

function updateControlVisibility() {
  const isLora  = selectedModel === "lora";
  const isNaive = selectedModel === "naive";

  promptGroup.style.display = isLora  ? "block" : "none";
  paramsGroup.style.display = isNaive ? "block" : "none";
}

// ── Preset prompts ────────────────────────────────────────────────────────────
presetRow.addEventListener("click", (e) => {
  const btn = e.target.closest(".preset-btn");
  if (!btn) return;

  presetRow.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");

  activePreset = btn.dataset.preset;
  if (activePreset !== "custom") {
    promptInput.value = PRESETS[activePreset];
  }
});

// Set default prompt
promptInput.value = PRESETS.standard;

// When user edits textarea, switch to custom preset
promptInput.addEventListener("input", () => {
  if (activePreset !== "custom") {
    presetRow.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
    presetRow.querySelector('[data-preset="custom"]').classList.add("active");
    activePreset = "custom";
  }
});

// ── Slider live update ────────────────────────────────────────────────────────
Object.keys(sliders).forEach(key => {
  sliders[key].addEventListener("input", () => {
    sliderVals[key].textContent = sliders[key].value;
  });
});

// ── Seed randomiser ───────────────────────────────────────────────────────────
seedRandom.addEventListener("click", () => {
  seedInput.value = Math.floor(Math.random() * 9000) + 1;
});

// ── Generate ──────────────────────────────────────────────────────────────────
generateBtn.addEventListener("click", generate);

async function generate() {
  if (isGenerating) return;
  isGenerating = true;

  const seed = parseInt(seedInput.value, 10) || 42;

  // Build request payload
  const payload = {
    model: selectedModel,
    seed,
    params: {},
    prompt: selectedModel === "lora" ? promptInput.value.trim() : "",
  };

  if (selectedModel === "naive") {
    payload.params = {
      streak_intensity: parseFloat(sliders.streakIntensity.value),
      noise_std:        parseFloat(sliders.noiseStd.value),
      n_angles:         parseInt(sliders.nAngles.value, 10),
      streak_width:     3,
    };
  }

  // UI → loading state
  setUIState("loading");

  try {
    const res  = await fetch("/api/generate", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok || !data.success) {
      showError(data.error || `Server error ${res.status}`);
      return;
    }

    showResult(data, seed);

  } catch (err) {
    showError("Network error — is the Flask server running?");
  } finally {
    isGenerating = false;
    generateBtn.classList.remove("loading");
    btnText.textContent = "Synthesize CT Scan";
  }
}

// ── UI state helpers ──────────────────────────────────────────────────────────
function setUIState(state) {
  outputPlaceholder.style.display = "none";
  outputResult.style.display      = "none";
  outputError.style.display       = "none";
  outputLoading.style.display     = "none";

  if (state === "loading") {
    outputLoading.style.display     = "flex";
    loadingLabel.textContent        = LOADING_LABELS[selectedModel] || "Synthesizing…";
    generateBtn.classList.add("loading");
    btnText.textContent = "Synthesizing…";
  } else if (state === "placeholder") {
    outputPlaceholder.style.display = "flex";
  }
}

function showResult(data, seed) {
  outputLoading.style.display = "none";
  outputResult.style.display  = "flex";

  // Set output image
  imgOutput.src = `data:image/png;base64,${data.output_image}`;

  // Output label
  outputLabel.textContent = MODEL_LABELS[data.model]?.toUpperCase() + " OUTPUT"
    || "GENERATED OUTPUT";

  // Meta pills
  metaModel.textContent = MODEL_LABELS[data.model] || data.model;
  metaSeed.textContent  = `seed ${seed}`;

  if (data.metrics?.ssim) {
    metaSSIM.textContent  = `SSIM ${data.metrics.ssim}`;
    metaSSIM.style.display = "block";
  } else {
    metaSSIM.style.display = "none";
  }
}

function showError(message) {
  outputLoading.style.display = "none";
  outputError.style.display   = "flex";
  document.getElementById("errorMessage").textContent = message;
}

// ── Smooth anchor scrolling with nav offset ───────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener("click", (e) => {
    const target = document.querySelector(link.getAttribute("href"));
    if (!target) return;
    e.preventDefault();
    const top = target.getBoundingClientRect().top + window.scrollY - 70;
    window.scrollTo({ top, behavior: "smooth" });
  });
});
