import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import soundfile as sf
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.features.preprocess import batch_preprocess, load_and_preprocess_audio

# 1. Setup
# Ensure the script runs correctly from the project root
if not os.path.exists("config.yaml"):
    print("Error: Please run this script from the project root (Deep-AE).")
    exit(1)

# Load configuration
config = load_config("config.yaml")
logger = get_logger(__name__, config)

def create_dummy_data(raw_dir: str):
    """Creates a few dummy .wav files to demonstrate batch processing."""
    classes = ["healthy", "crack", "severe_crack", "noise"]
    sr = 44100  # Original sample rate (different from target 22050 to show resampling)
    
    for cls in classes:
        cls_dir = os.path.join(raw_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        # Create 2 dummy files per class
        for i in range(1, 3):
            file_path = os.path.join(cls_dir, f"sample_{i:02d}.wav")
            # Generate 2.5 seconds of random noise (simulating audio)
            audio_data = np.random.uniform(-0.5, 0.5, int(sr * 2.5))
            sf.write(file_path, audio_data, sr)
    logger.info(f"Created dummy dataset in {raw_dir}")

def main():
    logger.info("=== Deep-AE Audio Preprocessing Demo ===")
    
    # 2. Extract configuration parameters
    raw_data_dir = config["paths"]["raw_data_dir"]
    processed_data_dir = config["paths"]["processed_data_dir"]
    target_sr = config["audio"]["sample_rate"]
    target_duration = config["audio"]["duration_sec"]
    
    # 3. Create some dummy audio files to test the pipeline
    create_dummy_data(raw_data_dir)
    
    # 4. Demonstrate individual file processing
    sample_file = os.path.join(raw_data_dir, "crack", "sample_01.wav")
    logger.info(f"Demonstrating individual processing for: {sample_file}")
    
    waveform, sr = load_and_preprocess_audio(
        sample_file, 
        target_sr=target_sr, 
        target_duration=target_duration
    )
    logger.info(f"Processed Waveform Shape: {waveform.shape}")
    logger.info(f"Processed Sample Rate: {sr} Hz")
    logger.info(f"Expected Duration: {len(waveform)/sr:.2f}s")
    
    # 5. Demonstrate batch processing
    logger.info("\n--- Starting Batch Pipeline ---")
    batch_preprocess(
        input_dir=raw_data_dir,
        output_dir=processed_data_dir,
        target_sr=target_sr,
        target_duration=target_duration
    )
    
    logger.info("Demo complete. Check the 'dataset/processed/' folder.")

if __name__ == "__main__":
    main()
