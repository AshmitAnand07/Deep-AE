# Deep-AE: AI-Powered Acoustic Structural Health Monitoring

![Deep-AE Platform Banner](assets/banner_placeholder.png)

> **Deep-AE** is a modern, production-ready machine learning platform designed to detect structural anomalies (cracks vs. healthy states) using acoustic emission analysis and Deep Convolutional Neural Networks (CNNs).

---

## 📌 Project Overview
Structural health monitoring traditionally relies on expensive, specialized hardware. Deep-AE provides an accessible, AI-driven alternative by analyzing acoustic signatures using standard audio files. By converting raw audio waveforms into **Mel Spectrograms**, the system leverages a custom CNN to identify structural cracking with high confidence.

**Key Features:**
- 🎵 **Automated Audio DSP Pipeline**: Converts `.wav`, `.mp3`, and `.flac` files to 22050Hz, trims silence, normalizes amplitude, and generates CNN-ready Mel Spectrograms.
- 🧠 **Deep CNN Architecture**: A lightweight, PyTorch-based model optimized for edge-device deployment.
- 🚀 **Streamlit Cloud Ready**: A beautifully designed, zero-gravity "Glassmorphism" web dashboard for real-time inference and visualization.
- 📊 **Rich Analytics**: Real-time confidence scoring, waveform rendering, and Mel Spectrogram visualization.

---

## 🏗️ Architecture & Pipeline

The Deep-AE pipeline follows a strict, modular flow from raw data to inference:

1. **Preprocessing** (`app/preprocess.py`): Audio is standardized to Mono/22050Hz, silence is trimmed, and amplitude is normalized.
2. **Spectrogram Generation** (`app/spectrogram_generator.py`): Audio is converted into 128-band Mel Spectrograms (in decibel scale) and saved as 128x128 PNG images.
3. **CNN Training** (`app/train.py`): PyTorch Dataset loaders feed the spectrograms into a custom `DeepAE_CNN` model with dynamic learning rate optimization and early stopping.
4. **Inference/Dashboard** (`app.py`): The Streamlit UI accepts new audio, runs the exact preprocessing pipeline in-memory (no disk I/O), and runs a forward pass through the trained model to yield confidence scores.

---

## 📸 Screenshots

| Dashboard Home | Prediction Results |
| :---: | :---: |
| ![Dashboard Home](assets/dashboard_home_placeholder.png) | ![Prediction Results](assets/prediction_placeholder.png) |
*(Replace placeholders with actual UI screenshots in the `assets/` folder)*

---

## ⚙️ Tech Stack
- **Backend & ML**: `Python 3.x`, `PyTorch`, `Torchvision`
- **Audio DSP**: `Librosa`, `SoundFile`
- **Data Visualization**: `Matplotlib`, `NumPy`
- **Frontend / UI**: `Streamlit`, Custom CSS (Glassmorphism & CSS Animations)

---

## 🚀 Quick Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Deep-AE.git
cd Deep-AE
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎮 Usage Instructions

### 1. Running the Web Dashboard (Streamlit)
To launch the modern Glassmorphism UI locally:
```bash
streamlit run app.py
```
*Navigate to `http://localhost:8501` in your browser. You can test the platform using the audio files provided in the `sample_audio/` folder.*

### 2. Running CLI Inference
To test an audio file directly from the terminal without the UI:
```bash
python app/predict.py sample_audio/test_audio.wav --save-chart
```

### 3. Retraining the Model (Optional)
If you wish to retrain the CNN on a custom dataset:
1. Place raw audio in `dataset/raw/crack/` and `dataset/raw/healthy/`.
2. Run the preprocessing pipeline: `python app/preprocess.py`
3. Generate spectrograms: `python app/spectrogram_generator.py`
4. Train the model: `python app/train.py`

---

## 🌐 Deployment Instructions

This repository is strictly organized to be deployed directly to **Streamlit Community Cloud**:
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and select this repository.
4. Set the Main file path to `app.py`.
5. Click **Deploy**. Streamlit will automatically read `requirements.txt` and build the environment.

See `deployment_notes.md` for advanced deployment strategies (e.g., FastAPI, Docker).

---

## 🔮 Future Scope
- **Edge Deployment**: Converting the PyTorch `.pth` model to ONNX or TensorFlow Lite for Raspberry Pi integration.
- **Continuous Monitoring**: Integrating REST APIs (FastAPI) to handle live microphone streams rather than static file uploads.
- **Multi-Class Anomalies**: Expanding the dataset to identify specific *types* of structural damage (e.g., micro-fractures, delamination).

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
