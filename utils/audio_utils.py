import librosa
import numpy as np
import soundfile as sf
import logging

def load_audio(file_path, target_sr=22050):
    """
    Loads an audio file, converts it to mono, and standardizes the sample rate.
    
    Args:
        file_path (str): Path to the input audio file.
        target_sr (int): Desired sample rate (default 22050 Hz).
        
    Returns:
        tuple: (audio_time_series, sample_rate)
    """
    try:
        # librosa.load automatically resamples to sr and converts to mono (mono=True by default)
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return y, sr
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        raise e

def trim_silence(y, top_db=20):
    """
    Trims leading and trailing silence from an audio signal.
    
    Args:
        y (np.ndarray): Audio time series.
        top_db (int): The threshold (in decibels) below reference to consider as silence.
        
    Returns:
        np.ndarray: Trimmed audio time series.
    """
    # librosa.effects.trim returns the trimmed signal and the index of the non-silent region
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed

def normalize_audio(y):
    """
    Normalizes the audio amplitude to the range [-1.0, 1.0] (Peak Normalization).
    
    Args:
        y (np.ndarray): Audio time series.
        
    Returns:
        np.ndarray: Normalized audio time series.
    """
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y_normalized = y / max_amp
    else:
        y_normalized = y
    return y_normalized

def save_audio(y, sr, output_path):
    """
    Saves the audio time series to a WAV file.
    
    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sample rate.
        output_path (str): Path to save the output WAV file.
    """
    # soundfile is robust and fast for saving WAV files
    sf.write(output_path, y, sr, format='WAV', subtype='PCM_16')
