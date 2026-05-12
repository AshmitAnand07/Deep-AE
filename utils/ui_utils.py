"""
ui_utils.py
===========
Deep-AE Streamlit Dashboard Utilities

This module contains helper functions for visualizing audio waveforms,
Mel Spectrograms, and confidence metrics optimized for a dark-themed,
modern Streamlit dashboard.

Author: Deep-AE Team
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ============================================================
# 1. MATPLOTLIB DARK THEME SETUP
# ============================================================

# Apply a dark theme globally for all matplotlib plots generated here
plt.style.use('dark_background')

# Custom color palette for the modern UI
COLOR_WAVEFORM = "#00C9FF"    # Cyan/Blue
COLOR_CRACK = "#ef4444"       # Red
COLOR_HEALTHY = "#22c55e"     # Green
COLOR_TEXT = "#e2e8f0"        # Slate 200
COLOR_BG = "#0b0f19"          # Same as CSS background
COLOR_CARD_BG = "#111928"     # Card background

# Override specific matplotlib parameters for better integration
plt.rcParams.update({
    "figure.facecolor": "none",       # Transparent background
    "axes.facecolor": "none",         # Transparent axes
    "axes.edgecolor": "#334155",      # Subtle border
    "axes.labelcolor": COLOR_TEXT,
    "text.color": COLOR_TEXT,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "grid.color": "#334155",          # Subtle grid
    "grid.alpha": 0.3,
})


# ============================================================
# 2. VISUALIZATION FUNCTIONS
# ============================================================

def plot_waveform(y: np.ndarray, sr: int, title: str = "Audio Waveform"):
    """
    Generates a dark-themed matplotlib figure of the audio waveform.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sample rate.
        title (str): Plot title.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    # Plot waveform with modern cyan color
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.85, color=COLOR_WAVEFORM)
    
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Time (s)", fontsize=10, alpha=0.8)
    ax.set_ylabel("Amplitude", fontsize=10, alpha=0.8)
    
    # Clean up aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    return fig


def plot_mel_spectrogram_ui(mel_spec_db: np.ndarray, sr: int, hop_length: int, title: str = "Mel Spectrogram"):
    """
    Generates a dark-themed matplotlib figure of the Mel Spectrogram.

    Args:
        mel_spec_db (np.ndarray): Mel Spectrogram in dB.
        sr (int): Sample rate.
        hop_length (int): Hop length used during generation.
        title (str): Plot title.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    # Using 'magma' colormap which looks excellent on dark backgrounds
    img = librosa.display.specshow(
        mel_spec_db, 
        sr=sr, 
        hop_length=hop_length, 
        x_axis="time", 
        y_axis="mel", 
        ax=ax, 
        cmap="magma"
    )
    
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Time (s)", fontsize=10, alpha=0.8)
    ax.set_ylabel("Frequency (Hz)", fontsize=10, alpha=0.8)
    
    # Customize colorbar for dark theme
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.set_ylabel('Amplitude (dB)', rotation=270, labelpad=15, alpha=0.8)
    cbar.outline.set_edgecolor('#334155')
    
    plt.tight_layout()
    return fig


def plot_confidence_bar_chart(result: dict):
    """
    Generates a dark-themed horizontal bar chart for prediction confidence.

    Args:
        result (dict): The output dictionary from the predict() function.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    probs = result["probabilities"]
    classes = list(probs.keys())
    values = list(probs.values())

    # Colors: Green for healthy, Red for crack, faded for non-predicted
    colors = []
    for cls in classes:
        if cls == result["label"]:
            colors.append(COLOR_CRACK if cls == "crack" else COLOR_HEALTHY)
        else:
            colors.append("#475569") # Slate 600 (muted)

    fig, ax = plt.subplots(figsize=(8, 2.5))
    
    # Create horizontal bars
    bars = ax.barh(classes, values, color=colors, height=0.6, alpha=0.9)
    
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Confidence Probability", fontsize=10, alpha=0.8)
    
    # Add percentage text on each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.02, 
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", 
            va="center", 
            fontsize=11, 
            fontweight="bold",
            color="#ffffff"
        )

    # Clean up aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0) # Hide y ticks
    ax.set_yticklabels([c.upper() for c in classes], fontweight="bold")
    ax.grid(True, axis='x', linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    return fig
