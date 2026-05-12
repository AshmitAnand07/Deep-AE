"""
inference_utils.py
==================
Deep-AE Phase 1 — Inference Utility Functions

This module contains all the helper functions needed for the prediction
pipeline. It bridges the gap between a raw audio file and a CNN-ready
tensor, reusing the EXACT same preprocessing and spectrogram parameters
that were used during training.

CRITICAL DESIGN RULE:
    The preprocessing steps here MUST be identical to those used in
    spectrogram_generator.py and dataset_loader.py. Any divergence
    (e.g., different n_mels, different normalization) will silently
    produce incorrect predictions.

Supported Input Formats: .wav, .mp3, .flac

Author: Deep-AE Team
"""

import os
import io
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import librosa
import librosa.display
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.model import DeepAE_CNN

logger = logging.getLogger(__name__)


# ============================================================
# 1. CONSTANTS (must match training pipeline exactly)
# ============================================================

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
IMG_SIZE = 128

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac"}

CLASS_MAP = {0: "crack", 1: "healthy"}

# Same transform used during validation/test (no augmentation)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# ============================================================
# 2. MODEL LOADING
# ============================================================

def load_model(
    checkpoint_path: str = "models/deepae_cnn.pth",
    num_classes: int = 2,
    device: torch.device = None
) -> Tuple[nn.Module, torch.device]:
    """
    Safely loads a trained DeepAE_CNN model from a checkpoint file.

    This function handles:
      - Missing checkpoint files (raises FileNotFoundError).
      - Architecture mismatches (raises RuntimeError).
      - Automatic device selection (GPU if available, else CPU).

    Args:
        checkpoint_path (str): Path to the .pth checkpoint file.
        num_classes (int): Number of output classes (must match training).
        device (torch.device): Device to load the model onto. Auto-detected if None.

    Returns:
        Tuple[nn.Module, torch.device]: (loaded_model_in_eval_mode, device)

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If model weights don't match the architecture.
    """
    # Auto-detect device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"[MODEL_NOT_FOUND] Checkpoint not found at: '{checkpoint_path}'. "
            f"Please train the model first using: python train.py"
        )

    # Initialize model architecture
    model = DeepAE_CNN(num_classes=num_classes)

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()  # Set to evaluation mode (disables Dropout, BatchNorm updates)

        epoch = checkpoint.get("epoch", "N/A")
        val_acc = checkpoint.get("val_accuracy", "N/A")
        logger.info(
            f"[MODEL_LOADED] Loaded checkpoint from '{checkpoint_path}' "
            f"(Epoch: {epoch}, Val Acc: {val_acc})"
        )

    except KeyError as e:
        raise RuntimeError(
            f"[MODEL_LOAD_FAIL] Checkpoint is missing expected key: {e}. "
            f"The checkpoint may be from an incompatible version."
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"[MODEL_LOAD_FAIL] Architecture mismatch. Weights don't match "
            f"the current model definition. Error: {e}"
        )

    return model, device


# ============================================================
# 3. AUDIO FILE VALIDATION
# ============================================================

def validate_audio_file(file_path: str) -> None:
    """
    Validates that an audio file exists and is in a supported format.

    Args:
        file_path (str): Path to the audio file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[FILE_NOT_FOUND] Audio file not found: '{file_path}'")

    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"[INVALID_FORMAT] Unsupported format '{ext}' for file: '{file_path}'. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    if os.path.getsize(file_path) == 0:
        raise ValueError(f"[EMPTY_FILE] File has zero bytes: '{file_path}'")


# ============================================================
# 4. AUDIO TO SPECTROGRAM (in-memory, no disk I/O)
# ============================================================

def audio_to_spectrogram_tensor(
    file_path: str,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS
) -> torch.Tensor:
    """
    Converts an audio file directly into a CNN-ready tensor.

    This function performs the ENTIRE preprocessing pipeline in memory:
      1. Load audio (any supported format) and convert to mono.
      2. Generate Mel Spectrogram using librosa.
      3. Convert power to dB scale.
      4. Render the spectrogram as an in-memory image (no disk write).
      5. Apply the same transforms used during training.
      6. Return a tensor ready for model.forward().

    Args:
        file_path (str): Path to the audio file (.wav, .mp3, .flac).
        sr (int): Target sample rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop length.
        n_mels (int): Number of Mel bands.

    Returns:
        torch.Tensor: Shape (1, 1, 128, 128) — ready for CNN forward pass.

    Raises:
        ValueError: If the waveform is silent.
        IOError: If the audio file cannot be read.
    """
    # Step 1: Load audio (librosa handles wav, mp3, flac)
    y, loaded_sr = librosa.load(file_path, sr=sr, mono=True)

    # Step 2: Guard against silent audio
    if np.max(np.abs(y)) < 1e-6:
        raise ValueError(f"[SILENT_AUDIO] Waveform is effectively silent: '{file_path}'")

    # Step 3: Generate Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=loaded_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Step 4: Render spectrogram to an in-memory image buffer
    #         This matches the exact same rendering used in spectrogram_generator.py
    fig, ax = plt.subplots(1, 1, figsize=(2.56, 2.56))
    librosa.display.specshow(
        mel_spec_db, sr=sr, hop_length=hop_length,
        x_axis=None, y_axis=None, ax=ax
    )
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save to an in-memory bytes buffer instead of disk
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Step 5: Open as PIL Image and apply training transforms
    image = Image.open(buf).convert("RGB")
    tensor = INFERENCE_TRANSFORM(image)

    # Step 6: Add batch dimension: (1, 128, 128) -> (1, 1, 128, 128)
    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# 5. PREDICTION LOGIC
# ============================================================

def predict(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device
) -> Dict[str, object]:
    """
    Runs CNN inference on a single spectrogram tensor.

    Args:
        model (nn.Module): Trained model in eval() mode.
        tensor (torch.Tensor): Shape (1, 1, 128, 128).
        device (torch.device): CPU or GPU.

    Returns:
        dict: {
            "label": str,
            "label_index": int,
            "confidence": float,
            "probabilities": {"crack": float, "healthy": float}
        }
    """
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probabilities))
    pred_label = CLASS_MAP[pred_idx]
    confidence = float(probabilities[pred_idx])

    result = {
        "label": pred_label,
        "label_index": pred_idx,
        "confidence": confidence,
        "probabilities": {
            CLASS_MAP[i]: float(probabilities[i])
            for i in range(len(CLASS_MAP))
        },
    }

    return result


# ============================================================
# 6. VISUALIZATION — CONFIDENCE BAR CHART
# ============================================================

def plot_prediction(
    result: Dict[str, object],
    file_name: str,
    save_path: Optional[str] = None
) -> None:
    """
    Generates a confidence bar chart for a prediction result.

    Args:
        result (dict): Output from predict().
        file_name (str): Name of the audio file (for title).
        save_path (str, optional): If provided, saves the chart as PNG.
                                   If None, displays interactively.
    """
    probs = result["probabilities"]
    classes = list(probs.keys())
    values = list(probs.values())

    # Color the predicted class green, others gray
    colors = [
        "#2ecc71" if cls == result["label"] else "#bdc3c7"
        for cls in classes
    ]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(classes, values, color=colors, edgecolor="#2c3e50", linewidth=0.8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_title(
        f"Prediction: {result['label'].upper()} ({result['confidence']:.1%})\n"
        f"File: {file_name}",
        fontsize=13, fontweight="bold"
    )

    # Add percentage text on each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontsize=11, fontweight="bold"
        )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT_SAVED] Confidence chart saved to: {save_path}")
    else:
        plt.show()
