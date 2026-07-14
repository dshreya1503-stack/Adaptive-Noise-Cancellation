"""
train.py
--------
Trains DAE, U-Net1D, and Wave-U-Net1D on the same synthetic noisy/clean
dataset, evaluates each with SNR improvement, MSE, and a simplified STOI-like
intelligibility proxy, and saves comparison plots.

NOTE ON METRICS: real STOI (Short-Time Objective Intelligibility) requires
the `pystoi` package, which isn't installable in this offline environment.
`simplified_stoi_proxy()` below implements an honest approximation (windowed
correlation between clean and enhanced envelopes) -- it captures the same
intuition (does short-time structure match?) but is NOT the real STOI
algorithm and should not be reported as one. This is called out explicitly
so results aren't misrepresented.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from nn import Tensor, Adam, mse_loss
from data_generation import generate_dataset, SAMPLE_RATE
from models import DAE, UNet1D, WaveUNet1D, count_params

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def snr_db(clean, estimate):
    noise = clean - estimate
    return 10 * np.log10((np.sum(clean ** 2) + 1e-10) / (np.sum(noise ** 2) + 1e-10))


def mse(clean, estimate):
    return float(np.mean((clean - estimate) ** 2))


def simplified_stoi_proxy(clean, estimate, sr=SAMPLE_RATE, win_ms=25):
    """Windowed Pearson correlation between clean and estimate envelopes.
    NOT real STOI -- see module docstring. Ranges roughly [-1, 1], higher=better."""
    win = int(sr * win_ms / 1000)
    n_wins = len(clean) // win
    if n_wins == 0:
        return float(np.corrcoef(clean, estimate)[0, 1])
    corrs = []
    for i in range(n_wins):
        c = clean[i * win:(i + 1) * win]
        e = estimate[i * win:(i + 1) * win]
        if c.std() < 1e-6 or e.std() < 1e-6:
            continue
        corrs.append(np.corrcoef(c, e)[0, 1])
    return float(np.mean(corrs)) if corrs else 0.0


# ---------------------------------------------------------------------------
# Training: DAE (frame-based)
# ---------------------------------------------------------------------------

def frames_from_signal(sig, frame_size, hop):
    n_frames = max(1, (len(sig) - frame_size) // hop + 1)
    frames = np.stack([sig[i * hop: i * hop + frame_size] for i in range(n_frames)])
    return frames


def train_dae(clean_train, noisy_train, epochs=25, frame_size=256, hop=256, lr=2e-3, batch_size=64):
    model = DAE(frame_size=frame_size)
    opt = Adam(model.parameters(), lr=lr)

    # build frame-level dataset
    clean_frames, noisy_frames = [], []
    for c, n in zip(clean_train, noisy_train):
        clean_frames.append(frames_from_signal(c, frame_size, hop))
        noisy_frames.append(frames_from_signal(n, frame_size, hop))
    clean_frames = np.concatenate(clean_frames, axis=0)
    noisy_frames = np.concatenate(noisy_frames, axis=0)

    n_samples = clean_frames.shape[0]
    losses = []
    for epoch in range(epochs):
        idx = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, batch_size):
            b_idx = idx[start:start + batch_size]
            x = Tensor(noisy_frames[b_idx])
            target = clean_frames[b_idx]
            opt.zero_grad()
            pred = model.forward(x)
            loss = mse_loss(pred, target)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.data)
            n_batches += 1
        losses.append(epoch_loss / n_batches)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [DAE] epoch {epoch+1}/{epochs}  loss={losses[-1]:.5f}")
    return model, losses


def denoise_dae(model, noisy_sig, frame_size=256, hop=256):
    frames = frames_from_signal(noisy_sig, frame_size, hop)
    x = Tensor(frames)
    pred = model.forward(x).data
    # overlap-add reconstruction (hop == frame_size here -> simple concat)
    out = pred.reshape(-1)
    out = np.pad(out, (0, max(0, len(noisy_sig) - len(out))))[:len(noisy_sig)]
    return out


# ---------------------------------------------------------------------------
# Training: U-Net / Wave-U-Net (waveform-chunk based)
# ---------------------------------------------------------------------------

def train_conv_model(model_class, clean_train, noisy_train, epochs=25, chunk_size=256,
                      lr=1e-3, batch_size=32, **model_kwargs):
    model = model_class(**model_kwargs)
    opt = Adam(model.parameters(), lr=lr)

    clean_chunks, noisy_chunks = [], []
    for c, n in zip(clean_train, noisy_train):
        cf = frames_from_signal(c, chunk_size, chunk_size)
        nf = frames_from_signal(n, chunk_size, chunk_size)
        clean_chunks.append(cf)
        noisy_chunks.append(nf)
    clean_chunks = np.concatenate(clean_chunks, axis=0)[:, None, :]  # (N,1,L)
    noisy_chunks = np.concatenate(noisy_chunks, axis=0)[:, None, :]

    n_samples = clean_chunks.shape[0]
    losses = []
    for epoch in range(epochs):
        idx = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, batch_size):
            b_idx = idx[start:start + batch_size]
            x = Tensor(noisy_chunks[b_idx])
            target = clean_chunks[b_idx]
            opt.zero_grad()
            pred = model.forward(x)
            loss = mse_loss(pred, target)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.data)
            n_batches += 1
        losses.append(epoch_loss / n_batches)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{model_class.__name__}] epoch {epoch+1}/{epochs}  loss={losses[-1]:.5f}")
    return model, losses


def denoise_conv(model, noisy_sig, chunk_size=256):
    frames = frames_from_signal(noisy_sig, chunk_size, chunk_size)
    x = Tensor(frames[:, None, :])
    pred = model.forward(x).data[:, 0, :]
    out = pred.reshape(-1)
    out = np.pad(out, (0, max(0, len(noisy_sig) - len(out))))[:len(noisy_sig)]
    return out


# ---------------------------------------------------------------------------
# Evaluation + plotting
# ---------------------------------------------------------------------------

def evaluate_model(name, denoise_fn, clean_test, noisy_test):
    snr_improvements, mses, stoi_proxies = [], [], []
    for c, n in zip(clean_test, noisy_test):
        est = denoise_fn(n)
        L = min(len(c), len(est))
        c, n, est = c[:L], n[:L], est[:L]
        snr_before = snr_db(c, n)
        snr_after = snr_db(c, est)
        snr_improvements.append(snr_after - snr_before)
        mses.append(mse(c, est))
        stoi_proxies.append(simplified_stoi_proxy(c, est))
    return {
        "name": name,
        "snr_improvement_mean": float(np.mean(snr_improvements)),
        "snr_improvement_std": float(np.std(snr_improvements)),
        "mse_mean": float(np.mean(mses)),
        "stoi_proxy_mean": float(np.mean(stoi_proxies)),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.random.seed(42)

    print("Generating synthetic dataset...")
    clean_all, noisy_all = generate_dataset(n_samples=64, duration_s=1.0)
    split = int(0.8 * len(clean_all))
    clean_train, noisy_train = clean_all[:split], noisy_all[:split]
    clean_test, noisy_test = clean_all[split:], noisy_all[split:]
    print(f"Train: {len(clean_train)}  Test: {len(clean_test)}")

    results = []
    t0 = time.time()

    print("\nTraining DAE...")
    dae_model, dae_losses = train_dae(clean_train, noisy_train, epochs=18)
    dae_res = evaluate_model("DAE", lambda s: denoise_dae(dae_model, s), clean_test, noisy_test)
    dae_res["params"] = count_params(dae_model)
    results.append(dae_res)

    print("\nTraining U-Net1D...")
    unet_model, unet_losses = train_conv_model(UNet1D, clean_train, noisy_train, epochs=15, batch_size=48)
    unet_res = evaluate_model("U-Net", lambda s: denoise_conv(unet_model, s), clean_test, noisy_test)
    unet_res["params"] = count_params(unet_model)
    results.append(unet_res)

    print("\nTraining Wave-U-Net1D...")
    wunet_model, wunet_losses = train_conv_model(WaveUNet1D, clean_train, noisy_train, epochs=10, batch_size=48)
    wunet_res = evaluate_model("Wave-U-Net", lambda s: denoise_conv(wunet_model, s), clean_test, noisy_test)
    wunet_res["params"] = count_params(wunet_model)
    results.append(wunet_res)

    print(f"\nTotal training time: {time.time()-t0:.1f}s")

    print("\n===== RESULTS =====")
    print(f"{'Model':<12}{'ΔSNR (dB)':<14}{'MSE':<12}{'STOI-proxy':<12}{'Params':<10}")
    for r in results:
        print(f"{r['name']:<12}{r['snr_improvement_mean']:<14.2f}{r['mse_mean']:<12.5f}"
              f"{r['stoi_proxy_mean']:<12.3f}{r['params']:<10}")

    # ---- Save comparison bar chart ----
    names = [r["name"] for r in results]
    snr_imps = [r["snr_improvement_mean"] for r in results]
    snr_stds = [r["snr_improvement_std"] for r in results]

    plt.figure(figsize=(6, 4))
    plt.bar(names, snr_imps, yerr=snr_stds, color=["#4C72B0", "#55A868", "#C44E52"], capsize=5)
    plt.ylabel("SNR Improvement (dB)")
    plt.title("Denoising Performance Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_snr.png"), dpi=150)
    plt.close()

    # ---- Save training loss curves ----
    plt.figure(figsize=(6, 4))
    plt.plot(dae_losses, label="DAE")
    plt.plot(unet_losses, label="U-Net")
    plt.plot(wunet_losses, label="Wave-U-Net")
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss_curves.png"), dpi=150)
    plt.close()

    # ---- Save an example waveform comparison (best model = Wave-U-Net expected) ----
    example_clean = clean_test[0]
    example_noisy = noisy_test[0]
    example_denoised = denoise_conv(wunet_model, example_noisy)
    t = np.arange(len(example_clean)) / SAMPLE_RATE

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, sharey=True)
    axes[0].plot(t, example_noisy, color="#C44E52", linewidth=0.6)
    axes[0].set_title("Noisy input")
    axes[1].plot(t, example_denoised, color="#55A868", linewidth=0.6)
    axes[1].set_title("Wave-U-Net output (denoised)")
    axes[2].plot(t, example_clean, color="#4C72B0", linewidth=0.6)
    axes[2].set_title("Ground-truth clean")
    axes[2].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "example_waveforms.png"), dpi=150)
    plt.close()

    # ---- Save models (weights as .npz) ----
    def save_model(model, path):
        params = {f"p{i}": p.data for i, p in enumerate(model.parameters())}
        np.savez(path, **params)

    save_model(dae_model, os.path.join(MODEL_DIR, "dae_weights.npz"))
    save_model(unet_model, os.path.join(MODEL_DIR, "unet_weights.npz"))
    save_model(wunet_model, os.path.join(MODEL_DIR, "wave_unet_weights.npz"))

    print("\nSaved plots to outputs/, weights to models/")


if __name__ == "__main__":
    main()
