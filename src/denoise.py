"""
denoise.py
----------
Load the trained Wave-U-Net weights and denoise a WAV file (or, with no
argument, run on a freshly synthesized noisy example and save before/after
WAVs + a plot).

Usage:
    python src/denoise.py                      # demo on synthetic example
    python src/denoise.py path/to/noisy.wav     # denoise your own 8kHz mono wav
"""

import sys
import os
import numpy as np
from scipy.io import wavfile

sys.path.append(os.path.dirname(__file__))
from nn import Tensor
from models import WaveUNet1D
from data_generation import generate_dataset, SAMPLE_RATE
from train import frames_from_signal, denoise_conv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_wave_unet():
    model = WaveUNet1D()
    weights = np.load(os.path.join(MODEL_DIR, "wave_unet_weights.npz"))
    for i, p in enumerate(model.parameters()):
        p.data = weights[f"p{i}"]
    return model


def load_wav_mono(path, target_sr=SAMPLE_RATE):
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if data.max() > 1.0 or data.min() < -1.0:
        data = data / (np.abs(data).max() + 1e-8)
    if sr != target_sr:
        # simple resampling via linear interpolation (no scipy.signal.resample
        # dependency issues); fine for a demo CLI, not audio-quality resampling
        duration = len(data) / sr
        n_target = int(duration * target_sr)
        data = np.interp(np.linspace(0, len(data), n_target), np.arange(len(data)), data)
    return data.astype(np.float32)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_wave_unet()

    if len(sys.argv) > 1:
        path = sys.argv[1]
        noisy = load_wav_mono(path)
        print(f"Loaded {path}: {len(noisy)/SAMPLE_RATE:.2f}s @ {SAMPLE_RATE}Hz")
    else:
        print("No file given -- running on a freshly synthesized noisy example.")
        clean, noisy = generate_dataset(n_samples=1, duration_s=1.5, seed=123)
        clean, noisy = clean[0], noisy[0]
        wavfile.write(os.path.join(OUTPUT_DIR, "demo_clean.wav"), SAMPLE_RATE, clean)

    denoised = denoise_conv(model, noisy, chunk_size=256)

    out_path = os.path.join(OUTPUT_DIR, "demo_denoised.wav")
    noisy_path = os.path.join(OUTPUT_DIR, "demo_noisy.wav")
    wavfile.write(out_path, SAMPLE_RATE, denoised)
    wavfile.write(noisy_path, SAMPLE_RATE, noisy.astype(np.float32))
    print(f"Saved noisy input  -> {noisy_path}")
    print(f"Saved denoised out -> {out_path}")


if __name__ == "__main__":
    main()
