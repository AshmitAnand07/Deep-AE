"""
predict.py
==========
Deep-AE Phase 1 — Prediction / Inference Pipeline

This is the main entry point for running predictions on new audio files
using the trained DeepAE_CNN model.

COMPLETE PIPELINE (per file):
    Raw Audio (.wav/.mp3/.flac)
        -> Validate file format & integrity
        -> Load audio + convert to mono + resample to 22050 Hz
        -> Generate Mel Spectrogram (librosa)
        -> Convert spectrogram to dB scale
        -> Render as in-memory image
        -> Apply training transforms (resize 128x128, normalize)
        -> CNN forward pass
        -> Softmax probabilities
        -> Predicted class + confidence score

USAGE:
    # Single file prediction
    python predict.py dataset/processed/crack/crack_001.wav

    # Predict from test folder
    python predict.py dataset/processed/crack/crack_005.wav dataset/processed/healthy/healthy_003.wav

    # With confidence chart saved
    python predict.py dataset/processed/crack/crack_001.wav --save-chart

IMPORTANT:
    This script uses the EXACT same preprocessing pipeline as training.
    The parameters (sr=22050, n_mels=128, n_fft=2048, hop_length=512) and
    the image transforms (resize 128x128, grayscale, normalize [-1,1]) are
    identical to those used in spectrogram_generator.py and dataset_loader.py.

Author: Deep-AE Team
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Add root directory to python path for package imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.inference_utils import (
    load_model,
    validate_audio_file,
    audio_to_spectrogram_tensor,
    predict,
    plot_prediction,
    CLASS_MAP,
    SUPPORTED_FORMATS,
)


# ============================================================
# 1. LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 2. CONFIGURATION
# ============================================================

MODEL_PATH = "models/deepae_cnn.pth"
CHART_OUTPUT_DIR = "predictions"


# ============================================================
# 3. SINGLE FILE PREDICTION
# ============================================================

def predict_audio_file(
    file_path: str,
    model,
    device,
    save_chart: bool = False
) -> dict:
    """
    Runs the full prediction pipeline on a single audio file.

    Steps:
        1. Validate the file (format, existence, size).
        2. Convert audio to a CNN-ready spectrogram tensor.
        3. Run the model forward pass.
        4. Print and return the results.
        5. Optionally save a confidence bar chart.

    Args:
        file_path (str): Path to the audio file.
        model: Trained CNN model in eval mode.
        device: CPU or GPU device.
        save_chart (bool): Whether to save a confidence visualization.

    Returns:
        dict: Prediction result (label, confidence, probabilities).
              Returns None if the file could not be processed.
    """
    file_name = Path(file_path).name
    logger.info(f"Processing: {file_name}")

    # --- Step 1: Validate ---
    try:
        validate_audio_file(file_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return None

    # --- Step 2: Audio -> Spectrogram Tensor ---
    try:
        tensor = audio_to_spectrogram_tensor(file_path)
    except Exception as e:
        logger.error(f"[PREPROCESSING_FAIL] Could not process '{file_name}': {e}")
        return None

    # --- Step 3: CNN Inference ---
    result = predict(model, tensor, device)

    # --- Step 4: Display results ---
    print_result(file_name, result)

    # --- Step 5: Save confidence chart ---
    if save_chart:
        chart_path = os.path.join(CHART_OUTPUT_DIR, f"{Path(file_path).stem}_prediction.png")
        plot_prediction(result, file_name, save_path=chart_path)

    return result


# ============================================================
# 4. FORMATTED OUTPUT
# ============================================================

def print_result(file_name: str, result: dict) -> None:
    """
    Prints a cleanly formatted prediction result to the console.

    Args:
        file_name (str): Name of the input file.
        result (dict): Prediction result from predict().
    """
    label = result["label"].upper()
    confidence = result["confidence"]
    probs = result["probabilities"]

    logger.info("-" * 50)
    logger.info(f"  File       : {file_name}")
    logger.info(f"  Prediction : {label}")
    logger.info(f"  Confidence : {confidence:.2%}")
    logger.info(f"  ---")
    for cls_name, prob in probs.items():
        bar = "#" * int(prob * 30)
        logger.info(f"  {cls_name:>10} : {prob:.4f}  [{bar}]")
    logger.info("-" * 50)


# ============================================================
# 5. COMMAND-LINE ARGUMENT PARSING
# ============================================================

def parse_args():
    """
    Parses command-line arguments for the prediction script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Deep-AE Audio Prediction Pipeline — Classify audio files as 'crack' or 'healthy'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict.py dataset/processed/crack/crack_001.wav\n"
            "  python predict.py file1.wav file2.mp3 file3.flac\n"
            "  python predict.py my_audio.wav --save-chart\n"
            "  python predict.py my_audio.wav --model models/deepae_cnn.pth\n"
        ),
    )
    parser.add_argument(
        "audio_files",
        nargs="+",
        help="One or more audio file paths to classify.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help=f"Path to the trained model checkpoint (default: {MODEL_PATH}).",
    )
    parser.add_argument(
        "--save-chart",
        action="store_true",
        help="Save a confidence bar chart for each prediction.",
    )

    return parser.parse_args()


# ============================================================
# 6. MAIN ENTRY POINT
# ============================================================

def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  Deep-AE Prediction Pipeline")
    logger.info("=" * 60)

    # --- Load the trained model ---
    try:
        model, device = load_model(checkpoint_path=args.model)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Device: {device}")
    logger.info(f"Files to process: {len(args.audio_files)}")
    logger.info("")

    # --- Process each audio file ---
    results = []
    for file_path in args.audio_files:
        result = predict_audio_file(
            file_path=file_path,
            model=model,
            device=device,
            save_chart=args.save_chart,
        )
        if result is not None:
            results.append({"file": file_path, **result})

    # --- Summary ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PREDICTION SUMMARY")
    logger.info("=" * 60)

    if not results:
        logger.warning("No files were successfully processed.")
    else:
        for r in results:
            fname = Path(r["file"]).name
            label = r["label"].upper()
            conf = r["confidence"]
            logger.info(f"  {fname:<30}  ->  {label:>8}  ({conf:.2%})")

    crack_count = sum(1 for r in results if r["label"] == "crack")
    healthy_count = sum(1 for r in results if r["label"] == "healthy")
    logger.info(f"\n  Total: {len(results)} | Crack: {crack_count} | Healthy: {healthy_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
