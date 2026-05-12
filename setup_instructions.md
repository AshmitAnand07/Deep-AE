# Deep-AE Setup Instructions

This document provides a comprehensive guide for setting up the Deep-AE environment from scratch on a local development machine. 

## Prerequisites

Ensure you have the following installed on your system:
- **Python 3.8+** (Recommended: Python 3.10)
- **Git**

## 1. Clone the Repository

Open your terminal or command prompt and run:
```bash
git clone https://github.com/yourusername/Deep-AE.git
cd Deep-AE
```

## 2. Create a Virtual Environment

It is highly recommended to isolate your Python dependencies.

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

Once the virtual environment is active, install the required packages. The `requirements.txt` is optimized for CPU inference out-of-the-box to ensure compatibility across all operating systems.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for GPU Users:** If you plan to *retrain* the model and have an NVIDIA GPU, you may want to install the CUDA-enabled version of PyTorch instead of the default CPU version. Visit the [PyTorch Get Started](https://pytorch.org/get-started/locally/) page for the specific pip command for your system (e.g., `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`).

## 4. Verify the Installation

To verify that everything is installed correctly, you can run a headless CLI prediction using the provided sample audio:

```bash
python app/predict.py sample_audio/test_audio.wav
```

You should see console output indicating the loaded model, the processed file, and the predicted class with a confidence score.

## 5. Launch the Web Dashboard

To start the interactive Streamlit user interface:

```bash
streamlit run app.py
```

This will automatically open your default web browser to `http://localhost:8501`. 

## 6. (Optional) Custom Dataset Setup

If you want to train the model on your own data:
1. Create the raw data folders: `dataset/raw/crack/` and `dataset/raw/healthy/`.
2. Place your raw `.wav`, `.mp3`, or `.flac` files in the respective folders.
3. Run `python app/preprocess.py` to standardize the audio.
4. Run `python app/spectrogram_generator.py` to generate the Mel Spectrograms.
5. Run `python app/train.py` to train a new model. The new `deepae_cnn.pth` will automatically be saved to the `models/` directory.
