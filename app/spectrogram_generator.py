"""
spectrogram_generator.py
========================
Deep-AE Phase 1 — Mel Spectrogram Generation Module

This script reads preprocessed WAV audio files from dataset/processed/,
converts each one into a Mel Spectrogram image, and saves the result as a
PNG file into dataset/spectrograms/ while preserving the class folder
structure (crack/, healthy/, etc.).

WHY MEL SPECTROGRAMS?
---------------------
A Mel Spectrogram is a visual representation of an audio signal that shows
how the distribution of energy across frequencies changes over time.

  - The X-axis represents TIME (left to right).
  - The Y-axis represents FREQUENCY, mapped to the Mel scale.
  - The COLOR intensity represents AMPLITUDE (energy) in decibels.

The "Mel" scale is a perceptual scale that mirrors how humans hear pitch:
we are much better at distinguishing low-frequency differences than
high-frequency ones. By using the Mel scale, the spectrogram emphasizes
the frequency ranges that carry the most meaningful acoustic information,
which is critical for detecting crack vs. healthy structural sounds.

Once generated, these 2D images are perfect inputs for a Convolutional
Neural Network (CNN), which excels at learning spatial patterns from images.

PARAMETER CHOICES:
------------------
  sample_rate = 22050 Hz  — Standard for audio ML. Captures frequencies
                            up to ~11 kHz (Nyquist), which is more than
                            enough for structural acoustic signatures.

  n_fft       = 2048      — The FFT window size. Larger windows give
                            better frequency resolution but worse time
                            resolution. 2048 is the go-to default for
                            most audio classification tasks.

  hop_length  = 512       — Number of samples between consecutive STFT
                            columns. This controls the time resolution
                            of the spectrogram. 512 gives a good balance
                            between time detail and computation cost.

  n_mels      = 128       — Number of Mel frequency bands. 128 gives a
                            high-resolution frequency axis for the CNN
                            to learn from, while keeping the image size
                            manageable (128 pixels tall).

Author: Deep-AE Team
"""

import os
import logging
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend (no GUI window needed)
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Paths (relative to the project root)
PROCESSED_DIR = Path("dataset/processed")
SPECTROGRAM_DIR = Path("dataset/spectrograms")

# Audio ML Parameters
SAMPLE_RATE = 22050    # Standard sample rate for audio ML
N_FFT = 2048           # FFT window size (frequency resolution)
HOP_LENGTH = 512       # Hop length (time resolution)
N_MELS = 128           # Number of Mel frequency bands

# Categories to process
CATEGORIES = ["crack", "healthy"]

# Image settings
IMAGE_DPI = 100        # Dots per inch for the saved PNG
FIGURE_SIZE = (2.56, 2.56)  # Produces a ~256x256 pixel image at 100 DPI


# ============================================================
# 2. LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("spectrogram_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def create_output_directories(categories: list, output_base: Path) -> None:
    """
    Creates the output spectrogram directories for each class category.
    If the directories already exist, this function does nothing.

    Args:
        categories (list): List of class folder names (e.g., ["crack", "healthy"]).
        output_base (Path): Base output directory for spectrograms.
    """
    for category in categories:
        output_dir = output_base / category
        output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directories ensured at: {output_base.absolute()}")


def generate_mel_spectrogram(audio_path: str,
                             sr: int = SAMPLE_RATE,
                             n_fft: int = N_FFT,
                             hop_length: int = HOP_LENGTH,
                             n_mels: int = N_MELS) -> np.ndarray:
    """
    Loads a WAV file and generates a Mel Spectrogram in decibel scale.

    Steps:
        1. Load the audio waveform using librosa.
        2. Compute the Mel-scaled power spectrogram.
        3. Convert the power values to decibels (log scale) for better
           dynamic range — this is critical because raw power values span
           many orders of magnitude and would be unusable as pixel values.

    Args:
        audio_path (str): Path to the preprocessed .wav file.
        sr (int): Sample rate (default 22050).
        n_fft (int): FFT window size (default 2048).
        hop_length (int): Hop length between frames (default 512).
        n_mels (int): Number of Mel bands (default 128).

    Returns:
        np.ndarray: 2D Mel Spectrogram array in dB scale (shape: n_mels x time_frames).

    Raises:
        IOError: If the audio file cannot be loaded.
        ValueError: If the loaded waveform is completely silent.
    """
    # Step 1: Load the audio waveform
    y, loaded_sr = librosa.load(audio_path, sr=sr, mono=True)

    # Step 2: Guard against silent/empty waveforms
    if np.max(np.abs(y)) < 1e-6:
        raise ValueError(f"Waveform is effectively silent (near-zero amplitude): {audio_path}")

    # Step 3: Compute the Mel Spectrogram (returns power spectrogram)
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=loaded_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # Step 4: Convert power to decibels (log scale)
    # ref=np.max normalizes relative to the peak, giving values in [-80, 0] dB range
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def save_spectrogram_image(mel_spec_db: np.ndarray,
                           output_path: str,
                           sr: int = SAMPLE_RATE,
                           hop_length: int = HOP_LENGTH,
                           figsize: tuple = FIGURE_SIZE,
                           dpi: int = IMAGE_DPI) -> None:
    """
    Saves a Mel Spectrogram as a clean PNG image (no axes, labels, or borders).

    The image is saved without any matplotlib decorations so that the CNN
    receives a pure visual representation of the frequency content — no
    axis ticks, titles, or whitespace that would introduce noise into
    the model's input.

    Args:
        mel_spec_db (np.ndarray): Mel Spectrogram in dB scale.
        output_path (str): File path to save the PNG image.
        sr (int): Sample rate used during generation.
        hop_length (int): Hop length used during generation.
        figsize (tuple): Figure size in inches (width, height).
        dpi (int): Dots per inch for the output image.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Display the spectrogram using librosa's display helper
    librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis=None,   # No time axis labels
        y_axis=None,   # No frequency axis labels
        ax=ax
    )

    # Remove all axes, borders, and padding for a clean image
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the image
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # Free memory — critical during batch processing


def process_single_file(input_path: Path, output_path: Path) -> bool:
    """
    Full pipeline for a single file: generate spectrogram and save as PNG.

    Args:
        input_path (Path): Path to the input .wav file.
        output_path (Path): Path to save the output .png file.

    Returns:
        bool: True if processing succeeded, False if it was skipped due to error.
    """
    try:
        # Generate the Mel Spectrogram
        mel_spec_db = generate_mel_spectrogram(str(input_path))

        # Save as PNG image
        save_spectrogram_image(mel_spec_db, str(output_path))

        return True

    except Exception as e:
        logger.warning(f"[SKIPPED] {input_path.name} — {type(e).__name__}: {e}")
        return False


# ============================================================
# 4. BATCH PROCESSING
# ============================================================

def batch_generate_spectrograms(input_base: Path = PROCESSED_DIR,
                                output_base: Path = SPECTROGRAM_DIR,
                                categories: list = None) -> None:
    """
    Batch processes all WAV files across all class categories.

    For each category folder inside `input_base`, this function:
        1. Finds all .wav files.
        2. Generates a Mel Spectrogram for each.
        3. Saves the spectrogram as a PNG in the corresponding
           category folder under `output_base`.

    Args:
        input_base (Path): Root directory of processed audio files.
        output_base (Path): Root directory for spectrogram output.
        categories (list): List of category folder names to process.
                           Defaults to CATEGORIES if not provided.
    """
    if categories is None:
        categories = CATEGORIES

    # Ensure output directories exist
    create_output_directories(categories, output_base)

    total_success = 0
    total_skipped = 0

    for category in categories:
        input_dir = input_base / category
        output_dir = output_base / category

        # Find all WAV files in this category
        wav_files = sorted(input_dir.glob("*.wav"))

        if not wav_files:
            logger.warning(f"No .wav files found in: {input_dir}")
            continue

        logger.info(f"Processing '{category}' — {len(wav_files)} files found.")

        for wav_file in wav_files:
            # Replace .wav extension with .png for the output filename
            output_filename = wav_file.stem + ".png"
            output_path = output_dir / output_filename

            success = process_single_file(wav_file, output_path)

            if success:
                total_success += 1
                logger.debug(f"[OK] {category}/{output_filename}")
            else:
                total_skipped += 1

    # Summary
    logger.info("=" * 50)
    logger.info("Spectrogram Generation Complete.")
    logger.info(f"Successfully generated : {total_success}")
    logger.info(f"Skipped (errors)       : {total_skipped}")
    logger.info(f"Output directory       : {output_base.absolute()}")
    logger.info("=" * 50)


# ============================================================
# 5. VISUALIZATION HELPER (for debugging / notebooks)
# ============================================================

def visualize_spectrogram(audio_path: str, title: str = "Mel Spectrogram") -> None:
    """
    Generates and displays a Mel Spectrogram with full axis labels.

    This function is intended for use in Jupyter Notebooks or interactive
    debugging sessions. Unlike `save_spectrogram_image`, this version
    includes axes, colorbar, and a title for human readability.

    Args:
        audio_path (str): Path to the .wav file to visualize.
        title (str): Title to display on the plot.
    """
    mel_spec_db = generate_mel_spectrogram(audio_path)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    img = librosa.display.specshow(
        mel_spec_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma"
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel Frequency (Hz)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Amplitude (dB)")
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Deep-AE Mel Spectrogram Generator")
    logger.info("=" * 50)

    # Run the batch pipeline
    batch_generate_spectrograms()
