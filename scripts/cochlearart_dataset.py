"""
cochlearart_dataset.py — PyTorch dataset for bone-window CT PNG slices.

Converts grayscale CT slices to 3-channel RGB (duplicating the single channel)
to match Stable Diffusion's expected input format, and applies augmentation
to combat overfitting on the small cochlear implant dataset.
"""

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

from config import LORA_IMG_SIZE, TRAIN_PROMPT


class CTSliceDataset(Dataset):
    """
    PyTorch Dataset for bone-window CT PNG slices.

    Converts grayscale images to 3-channel RGB (duplicating the single
    channel) to match Stable Diffusion's expected input format. Applies
    augmentation to combat overfitting on the small implant dataset.

    Args:
        paths: List of file paths to PNG slices.
        prompt: Text prompt to tokenize for each image.
        tokenizer: CLIPTokenizer instance.
        size: Resize target (square) in pixels.
    """

    def __init__(self, paths: list, prompt: str, tokenizer, size: int = LORA_IMG_SIZE):
        self.paths     = paths
        self.prompt    = prompt
        self.tokenizer = tokenizer
        self.transform = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.LANCZOS),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=12),                   # cochlea can be tilted
            T.ColorJitter(brightness=0.12, contrast=0.12),  # simulates HU variation
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        img     = Image.open(self.paths[idx]).convert("L")
        img_rgb = Image.merge("RGB", [img, img, img])
        pixels  = self.transform(img_rgb)
        tokens  = self.tokenizer(
            self.prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {"pixel_values": pixels, "input_ids": tokens.input_ids.squeeze(0)}


if __name__ == "__main__":
    print("Import CTSliceDataset from this module — do not run directly.")
