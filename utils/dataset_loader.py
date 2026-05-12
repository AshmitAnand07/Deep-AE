"""
dataset_loader.py
=================
Deep-AE Phase 1 — PyTorch Dataset & DataLoader Module

This module handles:
  1. Loading Mel Spectrogram PNG images from dataset/spectrograms/.
  2. Applying image transforms (resize, tensor conversion, normalization).
  3. Splitting the dataset into Train / Validation / Test sets.
  4. Creating PyTorch DataLoaders for each split.

The class folder structure is used as the label source:
  dataset/spectrograms/crack/    → label 0
  dataset/spectrograms/healthy/  → label 1

Author: Deep-AE Team
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Class-to-index mapping (alphabetical order matches torchvision.ImageFolder)
CLASS_MAP: Dict[str, int] = {
    "crack": 0,
    "healthy": 1,
}

# Reverse mapping for display
INDEX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_MAP.items()}

# Default image transforms for training
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),          # Resize to a fixed 128x128 input
    transforms.Grayscale(num_output_channels=1),  # Ensure single channel
    transforms.ToTensor(),                  # Convert PIL Image to [0, 1] tensor
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
])

# Validation/Test transforms — no augmentation, just resize + normalize
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# ============================================================
# 2. CUSTOM DATASET CLASS
# ============================================================

class SpectrogramDataset(Dataset):
    """
    A PyTorch Dataset that loads Mel Spectrogram images from disk.

    Folder structure expected:
        root_dir/
            crack/
                crack_001.png
                crack_002.png
                ...
            healthy/
                healthy_001.png
                ...

    Each subfolder name becomes the label via CLASS_MAP.
    """

    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir (str): Path to the spectrograms root directory.
            transform: torchvision transforms to apply to each image.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []   # List of (file_path, label_index) tuples

        # Walk through each class folder and collect all PNG paths
        for class_name, class_idx in CLASS_MAP.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            for img_file in sorted(class_dir.glob("*.png")):
                self.samples.append((str(img_file), class_idx))

        logger.info(
            f"Dataset loaded: {len(self.samples)} samples from {self.root_dir} "
            f"({', '.join(f'{k}: {sum(1 for _, l in self.samples if l == v)}' for k, v in CLASS_MAP.items())})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Loads a single spectrogram image and its label.

        Args:
            index (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, int]: (image_tensor, label_index)
        """
        img_path, label = self.samples[index]

        try:
            # Open image with PIL (works with PNG, JPEG, etc.)
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            logger.warning(f"[SKIPPED] Failed to load {img_path}: {e}")
            # Return a zero tensor as a fallback so the batch doesn't break
            return torch.zeros(1, 128, 128), label


# ============================================================
# 3. DATASET SPLITTING & DATALOADER CREATION
# ============================================================

def get_data_loaders(
    data_dir: str = "dataset/spectrograms",
    batch_size: int = 16,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates Train, Validation, and Test DataLoaders from the spectrogram dataset.

    The dataset is split randomly using a fixed seed for reproducibility.
    Training set gets augmentation-ready transforms; val/test get clean transforms.

    Args:
        data_dir (str): Root directory of spectrogram images.
        batch_size (int): Number of samples per batch.
        train_ratio (float): Fraction of data for training (default 0.70).
        val_ratio (float): Fraction of data for validation (default 0.15).
        test_ratio (float): Fraction of data for testing (default 0.15).
        seed (int): Random seed for reproducible splitting.
        num_workers (int): Number of background data loading processes.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    # Load the full dataset with training transforms initially
    full_dataset = SpectrogramDataset(data_dir, transform=TRAIN_TRANSFORM)
    total = len(full_dataset)

    if total == 0:
        raise RuntimeError(f"No samples found in {data_dir}. Check folder structure.")

    # Calculate split sizes (ensure they sum to total)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size  # Remainder goes to test

    logger.info(
        f"Splitting dataset: Total={total}, "
        f"Train={train_size}, Val={val_size}, Test={test_size}"
    )

    # Split with a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Override transforms for val/test (no augmentation)
    # Since random_split returns Subset objects, we wrap them
    val_set.dataset = SpectrogramDataset(data_dir, transform=VAL_TRANSFORM)
    test_set.dataset = SpectrogramDataset(data_dir, transform=VAL_TRANSFORM)

    # Create DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
