"""
naive_baseline.py — Physics-inspired metal artifact simulator (Model 1).

Deterministic baseline: inserts a synthetic cochlear implant ellipse into
a clean CT slice and adds beam-hardening streak artifacts using ray-casting.
No training data or learned parameters are used.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from config import WINDOWS, RESULTS_DIR, NAIVE_STREAK_INTENSITY, NAIVE_N_ANGLES, NAIVE_STREAK_WIDTH, NAIVE_NOISE_STD


class NaiveArtifactSimulator:
    """
    Deterministic physics-inspired metal artifact simulator.

    Simulates cochlear implant artifacts by:
      1. Inserting a high-HU ellipse (titanium electrode cross-section).
      2. Casting rays outward from the ellipse centre and adding
         intensity-decaying alternating bright/dark streaks (beam hardening).
      3. Adding Gaussian photon-starvation noise.

    Args:
        streak_intensity: Amplitude of streak artifacts [0, 1].
        n_angles: Number of streak directions (evenly spaced over π).
        streak_width: Width of each streak in pixels.
        noise_std: Standard deviation of additive Gaussian noise (fraction of 255).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        streak_intensity: float = NAIVE_STREAK_INTENSITY,
        n_angles: int           = NAIVE_N_ANGLES,
        streak_width: int       = NAIVE_STREAK_WIDTH,
        noise_std: float        = NAIVE_NOISE_STD,
        seed: int               = 42,
    ):
        self.streak_intensity = streak_intensity
        self.n_angles         = n_angles
        self.streak_w         = streak_width
        self.noise_std        = noise_std
        self.rng              = np.random.default_rng(seed)

    def _ellipse_mask(self, shape, cx, cy, rx, ry, angle_deg) -> np.ndarray:
        """Return a boolean mask for a rotated ellipse."""
        Y, X = np.ogrid[:shape[0], :shape[1]]
        a  = np.deg2rad(angle_deg)
        Xr = (X - cx) * np.cos(a) + (Y - cy) * np.sin(a)
        Yr = -(X - cx) * np.sin(a) + (Y - cy) * np.cos(a)
        return (Xr / rx) ** 2 + (Yr / ry) ** 2 <= 1

    def _add_streaks(self, img: np.ndarray, cx: int, cy: int) -> np.ndarray:
        """
        Cast rays from (cx, cy) and deposit alternating bright/dark streaks.

        Streak intensity decays exponentially with distance from the source.
        """
        result = img.copy().astype(np.float32)
        h, w   = img.shape

        for angle in np.linspace(0, np.pi, self.n_angles, endpoint=False):
            angle += self.rng.uniform(-0.05, 0.05)
            dx, dy = np.cos(angle), np.sin(angle)

            for direction in [1, -1]:
                for step in range(1, max(h, w)):
                    x = int(cx + direction * dx * step)
                    y = int(cy + direction * dy * step)
                    if not (0 <= x < w and 0 <= y < h):
                        break

                    decay = np.exp(-step / (max(h, w) * 0.3))
                    sign  = 1 if (step % 20 < 10) else -1

                    for dw in range(-self.streak_w // 2, self.streak_w // 2 + 1):
                        xw = int(x + dy * dw)
                        yw = int(y - dx * dw)
                        if 0 <= xw < w and 0 <= yw < h:
                            result[yw, xw] += sign * self.streak_intensity * decay * 255

        return np.clip(result, 0, 255).astype(np.uint8)

    def simulate(self, clean_img: np.ndarray) -> np.ndarray:
        """
        Insert a simulated cochlear implant artifact into a clean CT slice.

        Args:
            clean_img: uint8 grayscale CT slice (H × W).

        Returns:
            uint8 array with simulated metal and streak artifacts.
        """
        h, w = clean_img.shape
        cx   = int(w * self.rng.uniform(0.25, 0.40))
        cy   = int(h * self.rng.uniform(0.40, 0.60))
        rx   = int(self.rng.integers(5, 12))
        ry   = int(self.rng.integers(3, 7))
        ang  = float(self.rng.uniform(0, 90))

        mask   = self._ellipse_mask((h, w), cx, cy, rx, ry, ang)
        result = clean_img.copy().astype(np.float32)
        result[mask] = 255
        result = self._add_streaks(result.astype(np.uint8), cx, cy)

        noise = self.rng.normal(0, self.noise_std * 255, (h, w))
        return np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def run_naive_baseline(
    slices: list,
    implant_idx_set: set,
    results_dir: str = RESULTS_DIR,
    seed: int        = 42,
) -> np.ndarray:
    """
    Run the naive baseline on a clean (non-implant) CT slice and save results.

    Args:
        slices: Full list of loaded DICOM datasets.
        implant_idx_set: Set of slice indices that contain implants (to avoid).
        results_dir: Directory for saving output figures.
        seed: Random seed passed to the simulator.

    Returns:
        uint8 numpy array of the simulated output image.
    """
    from dicom_utils import dicom_to_hu, apply_window

    simulator = NaiveArtifactSimulator(seed=seed)

    non_implant = [i for i in range(len(slices)) if i not in implant_idx_set]
    clean_idx   = non_implant[len(non_implant) // 2] if non_implant else 0

    wc, ww    = WINDOWS["bone"]
    clean_hu  = dicom_to_hu(slices[clean_idx])
    clean_img = apply_window(clean_hu, wc, ww)
    synth     = simulator.simulate(clean_img)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Model 1: Naive Baseline — Physics Artifact Simulation", fontsize=14)
    axes[0].imshow(clean_img, cmap="gray")
    axes[0].set_title("Clean Input Slice"); axes[0].axis("off")
    axes[1].imshow(synth, cmap="gray")
    axes[1].set_title("Simulated Implant Artifact"); axes[1].axis("off")
    diff = np.abs(clean_img.astype(float) - synth.astype(float))
    axes[2].imshow(diff, cmap="gray")
    axes[2].set_title("Difference (artifact added)"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "naive_baseline_output.png"), dpi=150)
    plt.show()

    print("Naive baseline complete")
    return synth


if __name__ == "__main__":
    print("Import this module — do not run directly.")
