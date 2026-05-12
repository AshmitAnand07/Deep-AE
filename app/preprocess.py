import os
import logging
from pathlib import Path
from tqdm import tqdm
# Add root directory to python path for package imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import audio_utils# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)

# Configuration for paths
BASE_DIR = Path("dataset")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

CATEGORIES = ["crack", "healthy"]
TARGET_SR = 22050

def setup_directories():
    """Creates the necessary input and output directories if they don't exist."""
    for category in CATEGORIES:
        (RAW_DIR / category).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / category).mkdir(parents=True, exist_ok=True)
    logging.info(f"Directory structure ensured at {BASE_DIR.absolute()}")

def process_file(input_path, output_path):
    """
    Applies the full preprocessing pipeline to a single audio file.
    Pipeline: Load (Mono + 22050Hz) -> Trim Silence -> Normalize -> Save (WAV)
    """
    try:
        # 1. Load audio, convert to mono, and standard sample rate (22050 Hz)
        y, sr = audio_utils.load_audio(str(input_path), target_sr=TARGET_SR)
        
        # 2. Trim silence
        y_trimmed = audio_utils.trim_silence(y, top_db=20)
        
        # 3. Normalize amplitude
        y_normalized = audio_utils.normalize_audio(y_trimmed)
        
        # 4. Save to processed folder in WAV format
        audio_utils.save_audio(y_normalized, sr, str(output_path))
        
        return True
    except Exception as e:
        logging.error(f"Error processing file {input_path.name}: {e}")
        return False

def main():
    logging.info("Starting Audio Preprocessing Pipeline...")
    
    # 1. Ensure directory structure exists
    setup_directories()
    
    total_processed = 0
    total_errors = 0
    
    # 2. Batch process files by category
    for category in CATEGORIES:
        input_dir = RAW_DIR / category
        output_dir = PROCESSED_DIR / category
        
        # Find all files in the category folder (supporting various formats)
        valid_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
        files = [f for f in input_dir.rglob('*') if f.is_file() and f.suffix.lower() in valid_extensions]
        
        if not files:
            logging.warning(f"No audio files found in {input_dir}. Please add data and run again.")
            continue
            
        logging.info(f"Found {len(files)} files in '{category}' category. Processing...")
        
        # Process files with a progress bar
        for file_path in tqdm(files, desc=f"Processing {category}"):
            # Enforce .wav extension for the output
            output_filename = file_path.stem + ".wav"
            output_path = output_dir / output_filename
            
            success = process_file(file_path, output_path)
            
            if success:
                total_processed += 1
            else:
                total_errors += 1
                
    logging.info("Pipeline Execution Completed.")
    logging.info(f"Successfully processed: {total_processed} files.")
    logging.info(f"Failed to process: {total_errors} files.")

if __name__ == "__main__":
    main()
