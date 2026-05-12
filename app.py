"""
app.py
======
Deep-AE Streamlit Dashboard (Redesigned)

This is the main entry point for the modern Streamlit UI. It provides a
sleek, professional AI dashboard for structural health monitoring.

Usage:
    streamlit run app.py

Author: Deep-AE Team
"""

import os
import tempfile
import time
import logging
from pathlib import Path

import streamlit as st
import numpy as np
import librosa

from utils.inference_utils import (
    load_model,
    audio_to_spectrogram_tensor,
    predict,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS
)
from utils.ui_utils import (
    plot_waveform,
    plot_mel_spectrogram_ui,
    plot_confidence_bar_chart
)
from utils.ui_styles import get_custom_css

# ============================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Deep-AE | AI Monitor",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize logger
logger = logging.getLogger(__name__)


# ============================================================
# 2. CACHED MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_cached_model(checkpoint_path="models/deepae_cnn.pth"):
    try:
        model, device = load_model(checkpoint_path=checkpoint_path)
        return model, device, None
    except Exception as e:
        return None, None, str(e)


# ============================================================
# 3. SIDEBAR NAVIGATION & INFO
# ============================================================

with st.sidebar:
    st.markdown('<div class="sidebar-title">⚡ Deep-AE Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Acoustic Structural Health Monitoring System")
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Status")
    with st.spinner("Loading Weights..."):
        model, device, error_msg = load_cached_model()
        
    if error_msg:
        st.error("Offline")
        st.markdown('<div class="sidebar-info-box" style="border-color: #ef4444;">Failed to load checkpoint.</div>', unsafe_allow_html=True)
    else:
        st.success("Online")
        st.markdown(f"""
        <div class="sidebar-info-box">
            <b>Architecture:</b> Lightweight CNN<br>
            <b>Parameters:</b> ~4.2M<br>
            <b>Input Size:</b> 1x128x128<br>
            <b>Device:</b> {str(device).upper()}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📁 Supported Formats")
    st.markdown("""
    <div class="sidebar-info-box">
        • WAV (Lossless)<br>
        • MP3 (Compressed)<br>
        • FLAC (Lossless)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("v1.0.0 | Deep-AE Team")

# Halt execution if model failed to load
if error_msg:
    st.error("🚨 Critical Error: Could not load the CNN Model.")
    st.error(error_msg)
    st.stop()


# ============================================================
# 4. MAIN LAYOUT: HERO SECTION
# ============================================================

st.markdown('<div class="hero-title">Deep-AE</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AI-Powered Acoustic Structural Health Monitoring</div>', unsafe_allow_html=True)


# ============================================================
# 5. MAIN LAYOUT: UPLOAD SECTION
# ============================================================

# We use a container to wrap the uploader
with st.container():
    st.markdown("### 📥 Input Data")
    uploaded_file = st.file_uploader(
        "Drag and drop an acoustic emission recording here", 
        type=["wav", "mp3", "flac"],
        label_visibility="collapsed"
    )

if uploaded_file is None:
    st.info("💡 **Tip:** Upload an audio file to initiate the automated ML pipeline.")
    st.stop() # Wait for user upload

# ============================================================
# 6. PROCESSING PIPELINE
# ============================================================

# Save uploaded file securely
file_ext = Path(uploaded_file.name).suffix.lower()
with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
    tmp_file.write(uploaded_file.getvalue())
    tmp_file_path = tmp_file.name

try:
    st.markdown("---")
    
    # Create two main columns for layout: Left (Analysis), Right (Results)
    col_visuals, col_results = st.columns([1.5, 1], gap="large")
    
    with col_visuals:
        st.markdown("### 📈 Acoustic Signature Analysis")
        
        # Audio Player
        st.audio(uploaded_file, format=f'audio/{file_ext.replace(".", "")}')
        
        with st.status("Running DSP Pipeline...", expanded=True) as status:
            st.write("Reading audio waveform...")
            # Load raw audio
            y, sr = librosa.load(tmp_file_path, sr=SAMPLE_RATE, mono=True)
            
            st.write("Generating Mel Spectrogram...")
            # Generate spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            status.update(label="Signal Processing Complete", state="complete", expanded=False)

        # Display Plots in glass cards
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Time Domain (Waveform)")
        st.pyplot(plot_waveform(y, sr, title=""))
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Frequency Domain (Mel Spectrogram)")
        st.pyplot(plot_mel_spectrogram_ui(mel_spec_db, sr, HOP_LENGTH, title=""))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        st.markdown("### 🧠 ML Inference")
        
        start_time = time.time()
        
        # --- CNN INFERENCE ---
        with st.spinner("Executing Neural Network Forward Pass..."):
            # Progress bar animation for premium feel
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.005) # Tiny delay for smooth animation
                progress_bar.progress(percent_complete + 1)
                
            # 1. Convert to tensor
            tensor = audio_to_spectrogram_tensor(tmp_file_path)
            # 2. Predict
            result = predict(model, tensor, device)
            
            progress_bar.empty()
            
        inference_time = (time.time() - start_time) * 1000 # in ms
        
        label = result["label"].upper()
        confidence = result["confidence"]
        
        # --- PREDICTION BADGE ---
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Classification Result")
        
        if label == "CRACK":
            st.markdown(f'<div class="badge-crack">⚠️ CRACK DETECTED</div>', unsafe_allow_html=True)
            alert_msg = "Structural anomalies consistent with material cracking detected."
        else:
            st.markdown(f'<div class="badge-healthy">✅ HEALTHY</div>', unsafe_allow_html=True)
            alert_msg = "Acoustic signature is nominal. No structural anomalies detected."
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(alert_msg)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- ML ANALYTICS ---
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Confidence Metrics")
        
        # Confidence Chart
        st.pyplot(plot_confidence_bar_chart(result))
        
        st.markdown("---")
        
        # Performance Metrics
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">Max Confidence</span>
                <span class="metric-value">{confidence:.1%}</span>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">Inference Time</span>
                <span class="metric-value">{inference_time:.0f}ms</span>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error("🚨 **ML Pipeline Error**")
    st.error(str(e))
    logger.error(f"Error processing upload: {e}")
    
finally:
    # Clean up temporary file
    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)
