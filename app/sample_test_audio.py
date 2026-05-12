"""
Synthetic Bridge Structural Crack Audio Generator
-------------------------------------------------
This script generates a realistic synthetic bridge cracking sound
with environmental background noise and saves it as a WAV file.

Output:
dataset/raw/crack/bridge_crack_001.wav

Libraries Required:
- numpy
- scipy
- soundfile

Install with:
pip install numpy scipy soundfile
"""

import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, fftconvolve

# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 22050
DURATION = np.random.uniform(5, 10)  # Random duration between 5–10 sec
OUTPUT_PATH = "dataset/raw/crack/test_audio.wav"

# ============================================================
# CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ============================================================
# TIME AXIS
# ============================================================

t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_audio(audio):
    """
    Normalize audio to prevent clipping.
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio * 0.95


def butter_lowpass(cutoff, fs, order=4):
    """
    Create low-pass filter coefficients.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low')


def lowpass_filter(data, cutoff=500, fs=22050):
    """
    Apply low-pass filter.
    """
    b, a = butter_lowpass(cutoff, fs)
    return lfilter(b, a, data)


def create_reverb_impulse(length=0.25):
    """
    Create a simple synthetic reverb impulse response.
    """
    impulse_len = int(SAMPLE_RATE * length)

    impulse = np.random.randn(impulse_len)

    decay = np.exp(-np.linspace(0, 6, impulse_len))

    impulse *= decay

    return impulse


# ============================================================
# BACKGROUND AMBIENCE
# ============================================================

# Light ambient white noise
ambient_noise = np.random.normal(0, 0.01, len(t))

# Wind noise (low-frequency filtered noise)
wind = np.random.normal(0, 1, len(t))
wind = lowpass_filter(wind, cutoff=300)
wind *= 0.03

# Distant traffic rumble
traffic = np.random.normal(0, 1, len(t))
traffic = lowpass_filter(traffic, cutoff=120)
traffic *= 0.02

# Subtle environmental vibration
vibration = (
    0.01 * np.sin(2 * np.pi * 18 * t) +
    0.008 * np.sin(2 * np.pi * 26 * t)
)

# Combine ambience
background = ambient_noise + wind + traffic + vibration

# ============================================================
# STRUCTURAL CRACK GENERATION
# ============================================================

crack_audio = np.zeros(len(t))

# Number of crack events
num_events = np.random.randint(8, 18)

for _ in range(num_events):

    # Random crack start time
    start_time = np.random.uniform(0.5, DURATION - 0.5)

    start_sample = int(start_time * SAMPLE_RATE)

    # Random crack duration
    event_duration = np.random.uniform(0.05, 0.4)

    event_samples = int(event_duration * SAMPLE_RATE)

    # Time axis for event
    event_t = np.linspace(0, event_duration, event_samples)

    # ========================================================
    # METALLIC RESONANCE
    # ========================================================

    freq1 = np.random.uniform(200, 1200)
    freq2 = np.random.uniform(1200, 3500)

    metallic = (
        0.8 * np.sin(2 * np.pi * freq1 * event_t) +
        0.5 * np.sin(2 * np.pi * freq2 * event_t)
    )

    # ========================================================
    # RANDOM TRANSIENT IMPULSES
    # ========================================================

    impulses = np.random.normal(0, 1, event_samples)

    # Sharpen impulses
    impulses *= np.random.choice([0, 1], size=event_samples, p=[0.97, 0.03])

    # ========================================================
    # EXPONENTIAL DECAY ENVELOPE
    # ========================================================

    decay = np.exp(-event_t * np.random.uniform(10, 35))

    # ========================================================
    # COMBINE EVENT COMPONENTS
    # ========================================================

    crack_event = (metallic + impulses) * decay

    # Add some noise texture
    crack_event += 0.05 * np.random.randn(event_samples)

    # Random amplitude
    crack_event *= np.random.uniform(0.2, 0.8)

    # ========================================================
    # OCCASIONAL IMPACT-LIKE SOUND
    # ========================================================

    if np.random.rand() > 0.7:

        impact_len = int(0.03 * SAMPLE_RATE)

        impact = np.random.randn(impact_len)

        impact_decay = np.exp(-np.linspace(0, 20, impact_len))

        impact *= impact_decay * 0.8

        insert_len = min(impact_len, event_samples)

        crack_event[:insert_len] += impact[:insert_len]

    # ========================================================
    # INSERT EVENT INTO MAIN AUDIO
    # ========================================================

    end_sample = min(start_sample + event_samples, len(crack_audio))

    actual_len = end_sample - start_sample

    crack_audio[start_sample:end_sample] += crack_event[:actual_len]

# ============================================================
# ADD REVERB FOR REALISM
# ============================================================

reverb_impulse = create_reverb_impulse()

crack_audio = fftconvolve(crack_audio, reverb_impulse, mode='same')

# ============================================================
# FINAL MIX
# ============================================================

final_audio = background + crack_audio

# Small noise floor to avoid digital silence
final_audio += np.random.normal(0, 0.002, len(final_audio))

# Normalize audio
final_audio = normalize_audio(final_audio)

# ============================================================
# SAVE WAV FILE
# ============================================================

sf.write(
    OUTPUT_PATH,
    final_audio,
    SAMPLE_RATE
)

# ============================================================
# DONE
# ============================================================

print("======================================")
print("Synthetic bridge crack audio generated")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Sample Rate: {SAMPLE_RATE} Hz")
print(f"Duration: {DURATION:.2f} seconds")
print("======================================")