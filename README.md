# 🚧 Deep-AE — AI-Powered Acoustic Structural Crack Detection

> Deep-AE is an intelligent Structural Health Monitoring (SHM) platform that detects structural crack-related acoustic emissions using Deep Learning, Mel Spectrogram analysis, and CNN-based audio classification.

---

## 🌐 Live Demo

[Launch Deep-AE Dashboard]: (https://deepae.streamlit.app)

---

## 📌 Project Overview

Modern bridges and civil infrastructures often develop internal cracks long before visible failure occurs. These fractures generate high-frequency acoustic emissions that can be captured and analyzed.

Deep-AE leverages:

* 🎧 Acoustic Emission (AE) Analysis
* 🧠 Convolutional Neural Networks (CNNs)
* 📊 Mel Spectrogram Feature Engineering
* 🌐 Interactive AI Dashboard

to identify crack-related structural anomalies from audio signals in real time.

---

# ✨ Features

✅ Audio Preprocessing Pipeline
✅ Mel Spectrogram Generation
✅ CNN-Based Crack Classification
✅ Real-Time Audio Prediction
✅ Modern AI Monitoring Dashboard
✅ WAV / MP3 / FLAC Support
✅ Interactive Visualizations
✅ Confidence Score Analytics
✅ Streamlit Web Interface
✅ Modular Production-Style Architecture

---

# 🧠 AI Pipeline

```text
Audio Input
     ↓
Preprocessing
     ↓
Mel Spectrogram Generation
     ↓
CNN Feature Extraction
     ↓
Crack / Healthy Prediction
     ↓
Interactive Dashboard Visualization
```

---

# 🏗 System Architecture

## Core Components

| Module                | Description                                 |
| --------------------- | ------------------------------------------- |
| Audio Preprocessing   | Resampling, normalization, silence trimming |
| Spectrogram Generator | Converts audio into Mel Spectrogram images  |
| CNN Model             | Learns acoustic crack patterns              |
| Prediction Engine     | Runs real-time inference                    |
| Streamlit Dashboard   | Interactive monitoring interface            |

---

# 📂 Project Structure

```text
Deep-AE/
│
├── app/
├── assets/
├── docs/
├── logs/
├── models/
│   └── deepae_cnn.pth
│
├── utils/
├── sample_audio/
│
├── app.py
├── preprocess.py
├── spectrogram_generator.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

# 🔬 Technologies Used

## Machine Learning

* PyTorch
* CNN (Convolutional Neural Networks)

## Audio Processing

* Librosa
* NumPy
* SoundFile

## Visualization

* Matplotlib
* Plotly

## Dashboard

* Streamlit

---

# 🎵 Supported Audio Formats

* WAV
* MP3
* FLAC

---

# ⚙️ Installation

## Clone Repository

```bash
git clone YOUR_REPOSITORY_LINK
cd deep-ae-acoustic-crack-detection
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Application

## Launch Dashboard

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🧪 Model Training

```bash
python train.py
```

---

# 🔍 Run Prediction

```bash
python predict.py sample_audio/test_audio.wav
```

---

# 📊 Dashboard Features

The Deep-AE dashboard provides:

* 🎧 Audio Upload & Playback
* 📈 Waveform Visualization
* 🌈 Mel Spectrogram Visualization
* 🧠 AI Prediction System
* 📊 Confidence Analytics
* ⚡ Real-Time Monitoring Interface

---

# 📈 Future Improvements

* Real-Time Live Monitoring
* Edge AI Deployment (Raspberry Pi)
* Vibration Sensor Fusion
* Thermal Imaging Integration
* Transformer-Based Audio Models
* Cloud Monitoring Infrastructure

---

# 🎯 Applications

* Structural Health Monitoring
* Bridge Monitoring
* Smart Infrastructure
* Industrial Safety Systems
* Acoustic Anomaly Detection
* Predictive Maintenance

---

# 📸 Dashboard Preview

> <img width="1343" height="662" alt="image" src="https://github.com/user-attachments/assets/617c7138-3332-414d-8392-fd694a732263" />
> <img width="1385" height="495" alt="image" src="https://github.com/user-attachments/assets/c8c217cf-275a-407f-bd0e-887fda402619" />



```text
assets/dashboard_preview.png
```

---

# 👨‍💻 Author

**Ashmit Anand**

AI/ML + Structural Monitoring Project

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Acknowledgements

Special thanks to:

* Open-source ML community
* PyTorch
* Streamlit
* Librosa
* Kaggle datasets used for experimentation

---

# 🚀 Deep-AE

### “Listening to Structural Integrity with AI”
