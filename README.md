# Adaptive Noise Cancellation — Deep Learning Speech Enhancement

Compares three deep learning architectures — **DAE**, **U-Net1D**, and **Wave-U-Net1D** — for removing noise from speech signals, evaluated with SNR improvement, MSE, and an intelligibility proxy.

**Notable constraint this was built under:** no internet access and no PyTorch/TensorFlow available. So this implements a small **automatic differentiation engine from scratch in NumPy** (real backprop, verified with numerical gradient checks — see `src/gradcheck.py`) rather than calling `torch.nn`. All three architectures — including 1D convolution, pooling, upsampling, and U-Net-style skip connections — are built on that engine.

## Results

| Model | ΔSNR (dB) | MSE | STOI-proxy | Params |
|---|---|---|---|---|
| DAE | 0.97 | 0.0380 | 0.545 | 74,272 |
| U-Net1D | 5.59 | 0.0169 | 0.799 | 19,425 |
| **Wave-U-Net1D** | **9.36** | **0.0086** | **0.814** | 136,465 |

Wave-U-Net wins decisively despite the small dataset and short training — consistent with the literature: convolutional models with skip connections dramatically outperform plain dense autoencoders on raw-waveform denoising, and depth + receptive field (Wave-U-Net's wider kernels, 3 downsampling levels) beats a shallower U-Net.

See `outputs/model_comparison_snr.png`, `outputs/training_loss_curves.png`, and `outputs/example_waveforms.png`.

## Architecture summary

1. **DAE (Denoising Autoencoder)** — fully-connected encoder→bottleneck→decoder on fixed 256-sample frames. No spatial/temporal structure awareness — treats each frame as a flat vector. Simplest baseline.
2. **U-Net1D** — convolutional encoder-decoder, 2 downsampling levels, kernel size 3, with skip connections carrying fine-grained detail from encoder to decoder.
3. **Wave-U-Net1D** — same skip-connection idea but 3 levels deep with kernel size 9. Mirrors the real Wave-U-Net design choice: since it works directly on raw audio (no STFT/frequency transform), each layer needs a much wider receptive field to "see" enough context to separate speech from noise.

## Why build an autograd engine from scratch?

Because there was no way to `pip install torch` in this environment. Rather than fake it, `src/nn.py` implements:
- A `Tensor` class with a real computation graph and reverse-mode `.backward()`
- `Conv1d` via im2col/col2im (stride=1, same padding)
- Average-pool downsampling and nearest-neighbor upsampling with correct gradients
- Channel-wise concatenation for U-Net skip connections
- Adam optimizer

**Every op is verified against numerical gradients** in `src/gradcheck.py` (max error ~1e-10, i.e. correct to floating-point precision). This also surfaced and fixed a real reference-cycle memory leak in the backward-closure design — documented in `src/nn.py`'s `backward()` method — which is a genuinely good engineering story for an interview.

## Setup

```bash
pip install -r requirements.txt
python src/gradcheck.py       # verify autograd correctness (~1 sec)
python src/train.py           # train all 3 models, ~3 minutes on CPU
python src/denoise.py         # demo: denoise a synthetic example, save WAVs
python src/denoise.py your_audio.wav   # denoise your own 8kHz-ish mono wav
```

## Project structure

```
adaptive-noise-cancellation/
├── src/
│   ├── nn.py                # from-scratch autograd engine + layers
│   ├── gradcheck.py          # numerical gradient verification
│   ├── data_generation.py    # synthetic speech + noise generator
│   ├── models.py              # DAE, UNet1D, WaveUNet1D
│   ├── train.py               # training + evaluation + plots
│   └── denoise.py             # CLI inference on WAV files
├── models/                    # saved weights (.npz)
├── outputs/                   # plots + demo WAVs
├── requirements.txt
└── README.md
```

## Notes on the dataset

No internet access was available to download a real speech corpus (e.g. VoiceBank-DEMAND, the standard speech-enhancement benchmark). `src/data_generation.py` synthesizes speech-like signals — sums of harmonic partials over a drifting fundamental frequency, shaped by syllable-like amplitude envelopes — and mixes them with white/pink noise at randomized SNR (-5 to 10 dB). This preserves the *structure* of the problem (non-stationary harmonic signal + additive noise, need to recover fine temporal detail) without needing external data.

**To use real data:** swap `data_generation.generate_dataset()` for `scipy.io.wavfile.read()` calls over VoiceBank-DEMAND or similar — the model/training code is agnostic to where waveforms come from.

## Notes on the STOI-proxy metric

Real **STOI** (Short-Time Objective Intelligibility) requires the `pystoi` package, unavailable offline here. `simplified_stoi_proxy()` in `train.py` computes windowed correlation between clean and enhanced signal envelopes as an honest approximation of the same intuition — it is explicitly *not* claimed to be real STOI, and is labeled as such in code and results.

## Tech stack

Python, NumPy (custom autograd + conv), SciPy (WAV I/O), Matplotlib

## Future improvements

- Swap in real VoiceBank-DEMAND data
- Add `pystoi`/`pesq` for real intelligibility/quality metrics
- Try masking-based output (predict a ratio mask, multiply with noisy input) instead of direct waveform regression
- Vectorize `Conv1d` backward further / add GPU support via a real framework once available
  
