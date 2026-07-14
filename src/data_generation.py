"""
data_generation.py
-------------------
Generates synthetic "clean speech" signals and mixes them with noise at
controlled SNR levels, to create (noisy, clean) training pairs.

No internet access is available in this environment to download a real
corpus (e.g. VoiceBank-DEMAND, the standard speech-enhancement benchmark),
so clean "speech-like" signals are synthesized as sums of time-varying
sinusoids with amplitude envelopes -- mimicking the harmonic + formant
structure and non-stationary energy of real speech -- and corrupted with
white and colored (pink-ish) noise, the two noise types most commonly used
as speech-enhancement baselines.

To use a real corpus later: replace `make_clean_signal` / the noise
generator with `scipy.io.wavfile.read` calls over VoiceBank-DEMAND or
similar; the model/training code is agnostic to where the waveforms
come from.
"""

import numpy as np

SAMPLE_RATE = 8000  # 8kHz keeps chunk sizes small enough to train fast on CPU


def make_clean_signal(duration_s=1.0, sr=SAMPLE_RATE, seed=None):
    """Synthesize a speech-like signal: 3-5 harmonic partials of a slowly
    varying fundamental frequency, modulated by a smooth amplitude envelope
    with random syllable-like bursts (silence-like gaps in between)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    t = np.arange(n) / sr

    f0 = rng.uniform(90, 220)  # typical human pitch range
    f0_drift = f0 + 15 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t)

    signal = np.zeros(n)
    n_harmonics = rng.integers(3, 6)
    for h in range(1, n_harmonics + 1):
        amp = 1.0 / h
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * h * f0_drift * t + phase)

    # syllable-like amplitude envelope: smoothed random bursts
    n_syllables = rng.integers(2, 5)
    envelope = np.zeros(n)
    for _ in range(n_syllables):
        center = rng.uniform(0, duration_s)
        width = rng.uniform(0.08, 0.2)
        envelope += np.exp(-((t - center) ** 2) / (2 * width ** 2))
    envelope = envelope / (envelope.max() + 1e-8)

    sig = signal * envelope
    sig = sig / (np.abs(sig).max() + 1e-8) * 0.8
    return sig.astype(np.float32)


def make_noise(n, kind="white", seed=None):
    rng = np.random.default_rng(seed)
    if kind == "white":
        noise = rng.normal(0, 1, n)
    elif kind == "pink":
        # simple pink noise approximation via cumulative sum + normalization
        white = rng.normal(0, 1, n)
        noise = np.cumsum(white)
        noise = noise - np.mean(noise)
    else:
        raise ValueError(kind)
    noise = noise / (np.abs(noise).max() + 1e-8)
    return noise.astype(np.float32)


def mix_at_snr(clean, noise, snr_db):
    """Scale noise so that clean+noise mixture has the target SNR (dB)."""
    clean_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    noisy = clean + noise * scale
    return noisy.astype(np.float32)


def generate_dataset(n_samples=200, duration_s=1.0, sr=SAMPLE_RATE, seed=0):
    """Returns (clean_signals, noisy_signals) each shape (n_samples, n)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    clean_all = np.zeros((n_samples, n), dtype=np.float32)
    noisy_all = np.zeros((n_samples, n), dtype=np.float32)

    for i in range(n_samples):
        clean = make_clean_signal(duration_s, sr, seed=rng.integers(0, 1_000_000))
        noise_kind = rng.choice(["white", "pink"])
        noise = make_noise(n, kind=noise_kind, seed=rng.integers(0, 1_000_000))
        snr_db = rng.uniform(-5, 10)  # realistic range: hard to mild noise
        noisy = mix_at_snr(clean, noise, snr_db)
        clean_all[i] = clean
        noisy_all[i] = noisy

    return clean_all, noisy_all


if __name__ == "__main__":
    clean, noisy = generate_dataset(n_samples=5, duration_s=1.0)
    print("clean shape:", clean.shape, "noisy shape:", noisy.shape)
    print("example clean range:", clean[0].min(), clean[0].max())
    print("example noisy range:", noisy[0].min(), noisy[0].max())
