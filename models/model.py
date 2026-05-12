"""
model.py
========
Deep-AE Phase 1 — Lightweight CNN Architecture

This module defines the DeepAE_CNN model, a lightweight Convolutional Neural
Network designed for binary classification of Mel Spectrogram images
(crack vs. healthy).

ARCHITECTURE DESIGN DECISIONS:
------------------------------
  - 3 Convolutional blocks with increasing filter depth (16 → 32 → 64).
    This progressively extracts higher-level features while keeping the
    model small enough for future Raspberry Pi / edge deployment.

  - BatchNorm after each Conv2D stabilizes training and allows slightly
    higher learning rates.

  - MaxPool2D(2,2) halves the spatial dimensions after each block,
    reducing computation: 128→64→32→16.

  - A single fully-connected hidden layer (256 units) with Dropout(0.3)
    for regularization before the final classification head.

  - Output: 2 neurons (crack, healthy) — fed to CrossEntropyLoss which
    applies LogSoftmax internally, so we do NOT apply Softmax in forward().

INPUT:  (Batch, 1, 128, 128)  — single-channel grayscale spectrogram
OUTPUT: (Batch, 2)            — raw logits for each class

Author: Deep-AE Team
"""

import torch
import torch.nn as nn


class DeepAE_CNN(nn.Module):
    """
    A lightweight 3-block CNN for Mel Spectrogram classification.

    Architecture:
        Block 1: Conv2D(1→16) → BatchNorm → ReLU → MaxPool
        Block 2: Conv2D(16→32) → BatchNorm → ReLU → MaxPool
        Block 3: Conv2D(32→64) → BatchNorm → ReLU → MaxPool
        Classifier: Flatten → Dense(256) → ReLU → Dropout → Dense(num_classes)
    """

    def __init__(self, num_classes: int = 2):
        """
        Args:
            num_classes (int): Number of output classes (default 2: crack, healthy).
        """
        super(DeepAE_CNN, self).__init__()

        # --- Convolutional Feature Extractor ---

        # Block 1: Input (1, 128, 128) → Output (16, 64, 64)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: Input (16, 64, 64) → Output (32, 32, 32)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: Input (32, 32, 32) → Output (64, 16, 16)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # --- Classifier Head ---
        # After 3 MaxPool layers: 128 / (2^3) = 16, so feature map is 64 * 16 * 16
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # (Batch, 64, 16, 16) → (Batch, 16384)
            nn.Linear(64 * 16 * 16, 256),      # Dense hidden layer
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),                 # Regularization to prevent overfitting
            nn.Linear(256, num_classes)         # Output logits (NOT softmax — handled by loss)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 1, 128, 128).

        Returns:
            torch.Tensor: Raw logits of shape (Batch, num_classes).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def get_model_summary(model: nn.Module) -> str:
    """
    Returns a human-readable summary of the model's parameter count.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        str: Summary string.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (
        f"Total Parameters     : {total_params:,}\n"
        f"Trainable Parameters : {trainable_params:,}\n"
        f"Model Size (approx)  : {total_params * 4 / (1024**2):.2f} MB (float32)"
    )
