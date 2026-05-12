import numpy as np
import soundfile as sf
import os
from scipy import signal

def generate_bridge_crack_audio():
    # --- Configuration & Audio Specs ---
    fs = 22050            # Sample rate (Hz)
    duration = 8.0        # Duration in seconds
    num_samples = int(fs * duration)
    output_dir = "dataset/raw/crack"
    filename = "sample_test_audio_1.wav"
    filepath = os.path.join(output_dir, filename)

    # 1. Create Folder Structure
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Initialize empty audio buffer
    audio_out = np.zeros(num_samples)
    t = np.linspace(0, duration, num_samples)

    # --- Part A: Background Ambience ---
    print("Synthesizing background ambience...")
    
    # Light Wind (Pink Noise filtered)
    pink_noise = np.cumsum(np.random.randn(num_samples))
    pink_noise /= np.max(np.abs(pink_noise))
    
    # Distant Traffic Rumble (Low-pass filtered brown noise)
    brown_noise = np.cumsum(np.random.randn(num_samples))
    b, a = signal.butter(2, 150 / (fs / 2), btype='low')
    traffic = signal.lfilter(b, a, brown_noise)
    traffic = traffic / np.max(np.abs(traffic)) * 0.05
    
    # Combine Ambience
    audio_out += (pink_noise * 0.01) + (traffic * 0.04)

    # --- Part B: Structural Crack Events ---
    print("Generating crack propagation events...")
    
    # We define "Crack Events" as rapid bursts of energy
    num_events = np.random.randint(5, 12)
    
    for _ in range(num_events):
        # Random start time for the crack
        start_idx = np.random.randint(0, num_samples - int(fs * 0.5))
        
        # 1. Transient Impulse (The "Snap")
        # Creating a short burst of high-frequency noise
        event_len = int(fs * 0.1) # 100ms
        env = np.exp(-np.linspace(0, 10, event_len)) # Exponential decay
        crack_snap = np.random.uniform(-1, 1, event_len) * env
        
        # 2. Metallic Resonance (Ringing)
        # We use a decaying sine wave to simulate the vibration of metal beams
        res_freq = np.random.uniform(800, 3000) # Metallic ringing freq
        res_len = int(fs * 0.4) # Longer ring-out
        res_env = np.exp(-np.linspace(0, 5, res_len))
        resonance = np.sin(2 * np.pi * res_freq * np.linspace(0, 0.4, res_len)) * res_env
        
        # 3. Low-Frequency Creak
        # Simulate mechanical stress movement
        creak_len = int(fs * 0.3)
        creak_env = np.sin(np.pi * np.linspace(0, 1, creak_len)) # Hanning-like window
        creak_freq = np.random.uniform(40, 120)
        creak = np.sin(2 * np.pi * creak_freq * np.linspace(0, 0.3, creak_len)) * creak_env * 0.2

        # Apply random gain to the crack event
        gain = np.random.uniform(0.1, 0.4)
        
        # Inject into main buffer
        audio_out[start_idx : start_idx + event_len] += crack_snap * gain
        audio_out[start_idx : start_idx + res_len] += resonance * (gain * 0.3)
        audio_out[start_idx : start_idx + creak_len] += creak * gain

    # --- Part C: Post-Processing ---
    
    # 1. Apply a subtle "Room" Reverb (using a simple feedback delay)
    delay_samples = int(fs * 0.03) # 30ms delay
    decay = 0.3
    reverb = np.zeros_like(audio_out)
    reverb[delay_samples:] = audio_out[:-delay_samples] * decay
    audio_out += reverb

    # 2. Normalization
    # Ensure audio peaks at -1dB to avoid digital clipping
    if np.max(np.abs(audio_out)) > 0:
        audio_out = audio_out / np.max(np.abs(audio_out)) * 0.9

    # 3. Save File
    sf.write(filepath, audio_out, fs)
    print(f"Success! Generated: {filepath}")

if __name__ == "__main__":
    generate_bridge_crack_audio()