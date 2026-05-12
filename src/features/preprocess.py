import os
import glob
from typing import Optional, Tuple
import numpy as np
import librosa
import soundfile as sf

# Set up logger assuming the script is run from the project root
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_FORMATS = {".wav"}

def validate_audio_file(file_path: str, min_duration_sec: float = 0.5) -> bool:
    """
    Validates an audio file before processing.
    
    Args:
        file_path (str): Path to the audio file.
        min_duration_sec (float): Minimum required duration in seconds.
        
    Returns:
        bool: True if the file is valid.
        
    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file format is unsupported, empty, or too short.
        IOError: If file is corrupted or cannot be read.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[FILE_NOT_FOUND] No file at path: {file_path}")

    ext = os.path.splitext(file_path)[-1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"[INVALID_FORMAT] Expected .wav, got: {file_path}")

    if os.path.getsize(file_path) == 0:
        raise ValueError(f"[EMPTY_FILE] File has zero bytes: {file_path}")

    try:
        with sf.SoundFile(file_path) as f:
            duration = len(f) / f.samplerate
    except Exception as e:
        raise IOError(f"[CORRUPTED_FILE] Cannot read: {file_path} — {e}")

    if duration < min_duration_sec:
        raise ValueError(f"[AUDIO_TOO_SHORT] Duration {duration:.2f}s < {min_duration_sec}s: {file_path}")

    return True


def load_and_preprocess_audio(
    file_path: str, 
    target_sr: int = 22050, 
    target_duration: float = 2.0,
    trim_top_db: int = 20
) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file, converts to mono, resamples, trims silence, 
    pads/truncates to a fixed duration, and normalizes amplitude.
    
    Args:
        file_path (str): Path to the .wav file.
        target_sr (int): Desired sample rate (default 22050).
        target_duration (float): Target duration in seconds.
        trim_top_db (int): Threshold for silence trimming.
        
    Returns:
        Tuple[np.ndarray, int]: The processed waveform and the sample rate.
    """
    # 1. Load and convert to mono (librosa does mono conversion by default)
    # Using sr=target_sr performs resampling on load
    waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    # 2. Trim leading/trailing silence
    waveform, _ = librosa.effects.trim(waveform, top_db=trim_top_db)
    
    # 3. Standardize Length (Pad or Truncate)
    target_length = int(target_duration * target_sr)
    if len(waveform) > target_length:
        waveform = waveform[:target_length]
    elif len(waveform) < target_length:
        # Pad with zeros (silence)
        waveform = np.pad(waveform, (0, target_length - len(waveform)), mode='constant')
        
    # 4. Amplitude Normalization (Peak Normalization to [-1.0, 1.0])
    max_val = np.max(np.abs(waveform))
    if max_val > 1e-6:  # Prevent division by zero on silent audio
        waveform = waveform / max_val
        
    return waveform, sr


def save_audio(waveform: np.ndarray, sample_rate: int, output_path: str) -> None:
    """
    Saves a numpy waveform to a .wav file.
    
    Args:
        waveform (np.ndarray): Audio data.
        sample_rate (int): Sample rate.
        output_path (str): Where to save the file.
    """
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, waveform, sample_rate)


def batch_preprocess(
    input_dir: str, 
    output_dir: str, 
    target_sr: int = 22050,
    target_duration: float = 2.0
) -> None:
    """
    Processes all .wav files in an input directory and saves them to an output directory.
    Maintains the subfolder structure (e.g., classes).
    
    Args:
        input_dir (str): Root directory of raw audio files.
        output_dir (str): Root directory for processed audio files.
        target_sr (int): Target sample rate.
        target_duration (float): Target length in seconds.
    """
    logger.info(f"Starting batch preprocessing from '{input_dir}' to '{output_dir}'")
    
    # Find all .wav files recursively
    search_pattern = os.path.join(input_dir, "**", "*.wav")
    file_paths = glob.glob(search_pattern, recursive=True)
    
    if not file_paths:
        logger.warning(f"No .wav files found in {input_dir}")
        return
        
    success_count = 0
    skip_count = 0
    
    for file_path in file_paths:
        # Recreate subdirectory structure
        rel_path = os.path.relpath(file_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        try:
            # Validate
            validate_audio_file(file_path)
            
            # Preprocess
            waveform, sr = load_and_preprocess_audio(
                file_path=file_path, 
                target_sr=target_sr, 
                target_duration=target_duration
            )
            
            # Save
            save_audio(waveform, sr, output_path)
            success_count += 1
            logger.debug(f"[PROCESSED] {rel_path}")
            
        except Exception as e:
            logger.warning(f"[SKIPPED] {rel_path} — {e}")
            skip_count += 1
            
    logger.info("Batch preprocessing complete.")
    logger.info(f"Successfully processed: {success_count}/{len(file_paths)}")
    logger.info(f"Skipped files: {skip_count}")
