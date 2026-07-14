"""
models.py
---------
Three speech-enhancement architectures, in increasing order of
sophistication, built from the from-scratch layers in nn.py:

1. DAE (Denoising Autoencoder) — operates on fixed-size waveform frames,
   fully-connected encoder -> bottleneck -> decoder. The original/simplest
   deep learning approach to denoising (Vincent et al., 2008 idea applied
   to audio frames).

2. U-Net1D — convolutional encoder-decoder with skip connections between
   matching encoder/decoder resolutions. Skip connections let the decoder
   recover fine-grained detail (transients, high-frequency content) that
   would otherwise be lost at the bottleneck. 2 downsampling levels, small
   kernel size (3) — bigger receptive field via depth, not kernel width.

3. Wave-U-Net1D — same encoder-decoder-with-skips idea but deeper (3 levels)
   and with a wider kernel (9) at every layer. This mirrors the real
   Wave-U-Net design choice: since it operates directly on raw waveform
   samples (no STFT), it needs a much larger receptive field per layer to
   "see" enough audio context to distinguish speech from noise.
"""

import sys, os
sys.path.append(os.path.dirname(__file__))
import numpy as np
from nn import Linear, Conv1d, AvgPoolDownsample1d, NearestUpsample1d, concat_channels


class DAE:
    """Fully-connected denoising autoencoder on fixed-size frames."""

    def __init__(self, frame_size=256, bottleneck=32):
        self.frame_size = frame_size
        self.enc1 = Linear(frame_size, 128)
        self.enc2 = Linear(128, bottleneck)
        self.dec1 = Linear(bottleneck, 128)
        self.dec2 = Linear(128, frame_size)

    def forward(self, x):
        # x: Tensor of shape (N, frame_size)
        h = self.enc1(x).relu()
        z = self.enc2(h).relu()
        h2 = self.dec1(z).relu()
        out = self.dec2(h2).tanh()
        return out

    def parameters(self):
        return (self.enc1.parameters() + self.enc2.parameters() +
                self.dec1.parameters() + self.dec2.parameters())


class UNet1D:
    """2-level convolutional U-Net with skip connections. Input/output
    shape: (N, 1, L) with L divisible by 4."""

    def __init__(self, base_ch=16, kernel_size=3):
        k = kernel_size
        self.enc1 = Conv1d(1, base_ch, k)
        self.down1 = AvgPoolDownsample1d()
        self.enc2 = Conv1d(base_ch, base_ch * 2, k)
        self.down2 = AvgPoolDownsample1d()

        self.bottleneck = Conv1d(base_ch * 2, base_ch * 4, k)

        self.up2 = NearestUpsample1d()
        self.dec2 = Conv1d(base_ch * 4 + base_ch * 2, base_ch * 2, k)
        self.up1 = NearestUpsample1d()
        self.dec1 = Conv1d(base_ch * 2 + base_ch, base_ch, k)

        self.out_conv = Conv1d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x).relu()          # (N, base, L)
        d1 = self.down1(e1)                # (N, base, L/2)
        e2 = self.enc2(d1).relu()          # (N, 2base, L/2)
        d2 = self.down2(e2)                # (N, 2base, L/4)

        b = self.bottleneck(d2).relu()     # (N, 4base, L/4)

        u2 = self.up2(b)                   # (N, 4base, L/2)
        c2 = concat_channels(u2, e2)       # (N, 4base+2base, L/2)
        dec2 = self.dec2(c2).relu()        # (N, 2base, L/2)

        u1 = self.up1(dec2)                # (N, 2base, L)
        c1 = concat_channels(u1, e1)       # (N, 2base+base, L)
        dec1 = self.dec1(c1).relu()        # (N, base, L)

        out = self.out_conv(dec1).tanh()   # (N, 1, L)
        return out

    def parameters(self):
        ps = []
        for layer in [self.enc1, self.enc2, self.bottleneck, self.dec2, self.dec1, self.out_conv]:
            ps += layer.parameters()
        return ps


class WaveUNet1D:
    """3-level, wide-kernel U-Net variant -- mirrors real Wave-U-Net's design
    choice of a large receptive field per layer since it works on raw audio."""

    def __init__(self, base_ch=12, kernel_size=9):
        k = kernel_size
        self.enc1 = Conv1d(1, base_ch, k)
        self.down1 = AvgPoolDownsample1d()
        self.enc2 = Conv1d(base_ch, base_ch * 2, k)
        self.down2 = AvgPoolDownsample1d()
        self.enc3 = Conv1d(base_ch * 2, base_ch * 4, k)
        self.down3 = AvgPoolDownsample1d()

        self.bottleneck = Conv1d(base_ch * 4, base_ch * 8, k)

        self.up3 = NearestUpsample1d()
        self.dec3 = Conv1d(base_ch * 8 + base_ch * 4, base_ch * 4, k)
        self.up2 = NearestUpsample1d()
        self.dec2 = Conv1d(base_ch * 4 + base_ch * 2, base_ch * 2, k)
        self.up1 = NearestUpsample1d()
        self.dec1 = Conv1d(base_ch * 2 + base_ch, base_ch, k)

        self.out_conv = Conv1d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x).relu()
        d1 = self.down1(e1)
        e2 = self.enc2(d1).relu()
        d2 = self.down2(e2)
        e3 = self.enc3(d2).relu()
        d3 = self.down3(e3)

        b = self.bottleneck(d3).relu()

        u3 = self.up3(b)
        c3 = concat_channels(u3, e3)
        dec3 = self.dec3(c3).relu()

        u2 = self.up2(dec3)
        c2 = concat_channels(u2, e2)
        dec2 = self.dec2(c2).relu()

        u1 = self.up1(dec2)
        c1 = concat_channels(u1, e1)
        dec1 = self.dec1(c1).relu()

        out = self.out_conv(dec1).tanh()
        return out

    def parameters(self):
        ps = []
        for layer in [self.enc1, self.enc2, self.enc3, self.bottleneck,
                      self.dec3, self.dec2, self.dec1, self.out_conv]:
            ps += layer.parameters()
        return ps


def count_params(model):
    return sum(p.data.size for p in model.parameters())


if __name__ == "__main__":
    from nn import Tensor
    x_frame = Tensor(np.random.randn(4, 256))
    dae = DAE()
    out = dae.forward(x_frame)
    print("DAE out shape:", out.shape, "| params:", count_params(dae))

    x_wave = Tensor(np.random.randn(2, 1, 256))
    unet = UNet1D()
    out2 = unet.forward(x_wave)
    print("UNet1D out shape:", out2.shape, "| params:", count_params(unet))

    wunet = WaveUNet1D()
    out3 = wunet.forward(x_wave)
    print("WaveUNet1D out shape:", out3.shape, "| params:", count_params(wunet))
