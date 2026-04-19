"""
dict_learning.py — Sparse Dictionary Learning image reconstruction (Model 2).

Classical ML baseline: learns a set of CT texture atoms using
MiniBatchDictionaryLearning, then reconstructs images by sparse coding
each patch against the learned dictionary.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import MiniBatchDictionaryLearning
from skimage.metrics import structural_similarity as ssim

from config import (
    IMG_SIZE, RESULTS_DIR, SEED,
    DL_PATCH_SIZE, DL_N_ATOMS, DL_N_NONZERO, DL_N_ITER,
    DL_MAX_IMGS, DL_MAX_PATCHES, DL_RECON_STRIDE,
)


class DictionaryLearningGenerator:
    """
    Classical ML approach: sparse dictionary learning on CT patches.

    Learns a dictionary of image atoms from CT slice patches using
    MiniBatchDictionaryLearning with Orthogonal Matching Pursuit (OMP)
    for sparse coding. Reconstructs new images by encoding patches
    against the learned atoms and averaging overlapping regions.

    Args:
        patch_size: Side length of square image patches (pixels).
        n_atoms: Number of dictionary atoms to learn.
        n_nonzero: Maximum number of non-zero coefficients per patch (OMP sparsity).
        n_iter: Number of MiniBatchDictionaryLearning iterations.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        patch_size: int   = DL_PATCH_SIZE,
        n_atoms: int      = DL_N_ATOMS,
        n_nonzero: int    = DL_N_NONZERO,
        n_iter: int       = DL_N_ITER,
        random_state: int = SEED,
    ):
        self.ps      = patch_size
        self.n_atoms = n_atoms
        self.n_nz    = n_nonzero
        self.n_iter  = n_iter
        self.rs      = random_state
        self.model   = None

    def _extract_patches(self, images: list, stride: int = 8) -> np.ndarray:
        """
        Extract overlapping patches from a list of uint8 grayscale images.

        Args:
            images: List of H×W uint8 numpy arrays.
            stride: Step size between patches.

        Returns:
            Float32 array of shape (n_patches, patch_size²), normalised to [0, 1].
        """
        patches = []
        for img in images:
            h, w = img.shape
            for y in range(0, h - self.ps + 1, stride):
                for x in range(0, w - self.ps + 1, stride):
                    patch = img[y:y + self.ps, x:x + self.ps].ravel()
                    patches.append(patch.astype(np.float32) / 255.0)
        return np.array(patches)

    def fit(self, images: list, max_patches: int = DL_MAX_PATCHES) -> None:
        """
        Train the dictionary on a list of uint8 grayscale images.

        Args:
            images: List of H×W uint8 numpy arrays (training set).
            max_patches: Maximum number of patches to sample for training.
        """
        patches = self._extract_patches(images)
        if len(patches) > max_patches:
            idx     = np.random.choice(len(patches), max_patches, replace=False)
            patches = patches[idx]

        print(f"Training dictionary on {len(patches)} patches ({self.ps}×{self.ps})...")
        self.model = MiniBatchDictionaryLearning(
            n_components=self.n_atoms,
            transform_algorithm="omp",
            transform_n_nonzero_coefs=self.n_nz,
            max_iter=self.n_iter,
            random_state=self.rs,
            verbose=1,
        )
        self.model.fit(patches)
        print(f"Dictionary fitted: {self.n_atoms} atoms of dim {self.ps ** 2}")

    def reconstruct(self, image: np.ndarray, stride: int = DL_RECON_STRIDE) -> np.ndarray:
        """
        Encode and reconstruct a uint8 grayscale image using the learned dictionary.

        Args:
            image: H×W uint8 grayscale image to reconstruct.
            stride: Step size between reconstruction patches.

        Returns:
            uint8 reconstructed image of the same shape.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before reconstruct().")

        h, w   = image.shape
        output = np.zeros((h, w), np.float32)
        count  = np.zeros((h, w), np.float32)

        for y in range(0, h - self.ps + 1, stride):
            for x in range(0, w - self.ps + 1, stride):
                patch = image[y:y + self.ps, x:x + self.ps].ravel().astype(np.float32) / 255.0
                code  = self.model.transform(patch.reshape(1, -1))
                recon = (code @ self.model.components_).reshape(self.ps, self.ps)
                output[y:y + self.ps, x:x + self.ps] += recon
                count[y:y + self.ps,  x:x + self.ps] += 1

        count = np.where(count == 0, 1, count)
        return np.clip(output / count * 255, 0, 255).astype(np.uint8)

    def visualize_atoms(self, n: int = 64, save_path: str = None) -> None:
        """
        Plot the first n learned dictionary atoms as a grid.

        Args:
            n: Number of atoms to display.
            save_path: If given, save the figure here.
        """
        if self.model is None:
            raise RuntimeError("Call fit() first.")

        n    = min(n, self.n_atoms)
        cols = 8
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        for i, ax in enumerate(axes.ravel()):
            if i < n:
                ax.imshow(
                    self.model.components_[i].reshape(self.ps, self.ps),
                    cmap="gray", interpolation="nearest",
                )
            ax.axis("off")
        plt.suptitle(f"Dictionary Atoms (first {n} of {self.n_atoms})", fontsize=11)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


def run_dict_learning(
    train_paths: list,
    val_paths: list,
    results_dir: str = RESULTS_DIR,
    img_size: int    = IMG_SIZE,
    max_imgs: int    = DL_MAX_IMGS,
) -> tuple:
    """
    Train a DictionaryLearningGenerator and reconstruct a validation slice.

    Args:
        train_paths: List of paths to training bone-window PNG slices.
        val_paths: List of paths to validation bone-window PNG slices.
        results_dir: Directory for saving output figures.
        img_size: Display size for output images.
        max_imgs: Maximum training images to load.

    Returns:
        Tuple of (dl_gen, dl_output, ssim_score).
    """
    train_imgs = [
        np.array(Image.open(p).convert("L")) for p in train_paths[:max_imgs]
    ]
    print(f"Loaded {len(train_imgs)} training images "
          f"({train_imgs[0].shape[0]}×{train_imgs[0].shape[1]})")

    dl_gen = DictionaryLearningGenerator()
    dl_gen.fit(train_imgs)
    dl_gen.visualize_atoms(
        save_path=os.path.join(results_dir, "dictionary_atoms.png")
    )

    print("Reconstructing validation slice...")
    val_img    = np.array(Image.open(val_paths[5]).convert("L"))
    val_small  = np.array(Image.fromarray(val_img).resize((128, 128)))
    recon_sm   = dl_gen.reconstruct(val_small)
    recon_disp = np.array(Image.fromarray(recon_sm).resize((img_size, img_size)))
    val_disp   = np.array(Image.fromarray(val_small).resize((img_size, img_size)))

    ssim_score = ssim(val_disp, recon_disp, data_range=255)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Model 2: Dictionary Learning Reconstruction", fontsize=14)
    axes[0].imshow(val_img,    cmap="gray"); axes[0].set_title("Original Slice");                        axes[0].axis("off")
    axes[1].imshow(recon_disp, cmap="gray"); axes[1].set_title(f"Dictionary Recon\nSSIM: {ssim_score:.3f}"); axes[1].axis("off")
    diff = np.abs(val_disp.astype(float) - recon_disp.astype(float))
    axes[2].imshow(diff,       cmap="gray"); axes[2].set_title("Difference Map");                        axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "dict_learning_output.png"), dpi=150)
    plt.show()

    print(f"Dictionary learning complete — SSIM: {ssim_score:.3f}")
    return dl_gen, recon_disp, ssim_score


if __name__ == "__main__":
    print("Import this module — do not run directly.")
