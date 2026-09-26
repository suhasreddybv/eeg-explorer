"""Compact architecture-family adaptations, not exact paper reproductions."""

import numpy as np
from scipy import signal
import torch
from torch import nn
from .model import EEGNet

FAMILIES = ["eegnet", "shallow", "channel_cnn", "wavelet_cnn", "graph", "transformer"]


class Base(nn.Module):
    def constrain(self):
        pass


class Shallow(Base):
    def __init__(self, c, dropout):
        super().__init__()
        self.temporal = nn.Conv2d(1, 20, (1, 25))
        self.spatial = nn.Conv2d(20, 20, (c, 1), bias=False)
        self.bn = nn.BatchNorm2d(20)
        self.pool = nn.AvgPool2d((1, 75), (1, 15))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(20, 1)

    def forward(self, x):
        x = self.bn(self.spatial(self.temporal(x[:, None]))).square()
        x = self.pool(x).clamp_min(1e-6).log().mean((-1, -2))
        return self.head(self.drop(x)).squeeze(-1)


class ChannelCNN(Base):
    def __init__(self, c, dropout):
        super().__init__()
        layers = []
        for a, b, k in [(c, 21, 20), (21, 42, 10), (42, 42, 10), (42, 64, 5)]:
            layers.extend(
                [
                    nn.Conv1d(a, b, k, padding=k // 2),
                    nn.BatchNorm1d(b),
                    nn.LeakyReLU(0.1),
                    nn.MaxPool1d(4),
                ]
            )
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


class WaveletCNN(Base):
    def __init__(self, c, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Graph(Base):
    """Learned four-head adjacency and two-step message passing over electrodes."""

    def __init__(self, c, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 25, stride=4, padding=12),
            nn.GELU(),
            nn.Conv1d(16, 32, 15, stride=4, padding=7),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.q = nn.Linear(32, 32)
        self.k = nn.Linear(32, 32)
        self.v = nn.Linear(32, 32)
        self.msg = nn.Linear(64, 32)
        self.norm = nn.LayerNorm(32)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        b, c, t = x.shape
        h = self.encoder(x.reshape(b * c, 1, t)).reshape(b, c, 32)
        q = self.q(h).reshape(b, c, 4, 8).transpose(1, 2)
        k = self.k(h).reshape(b, c, 4, 8).transpose(1, 2)
        v = self.v(h).reshape(b, c, 4, 8).transpose(1, 2)
        a = (q @ k.transpose(-1, -2) / np.sqrt(8)).softmax(-1)
        one = a @ v
        two = a @ one
        msg = torch.cat([one, two], -1).transpose(1, 2).reshape(b, c, 64)
        return self.head(self.norm(h + self.msg(msg)).mean(1)).squeeze(-1)


class Transformer(Base):
    def __init__(self, c, dropout):
        super().__init__()
        self.embed = nn.Linear(5, 32)
        self.position = nn.Parameter(torch.randn(1, c, 32) * 0.02)
        layer = nn.TransformerEncoderLayer(32, 4, 64, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        return self.head(self.encoder(self.embed(x) + self.position).mean(1)).squeeze(-1)


def make(name, c=17, dropout=0.5):
    return {
        "eegnet": lambda: EEGNet(channels=c, dropout=dropout),
        "shallow": lambda: Shallow(c, dropout),
        "channel_cnn": lambda: ChannelCNN(c, dropout),
        "wavelet_cnn": lambda: WaveletCNN(c, dropout),
        "graph": lambda: Graph(c, dropout),
        "transformer": lambda: Transformer(c, dropout),
    }[name]()


def spectral_tokens(x):
    f, p = signal.welch(x, fs=256, nperseg=512, axis=-1)
    features = [
        np.log10(
            np.maximum(
                np.trapezoid(p[..., (f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)], axis=-1), 1e-12
            )
        )
        for lo, hi in [(1, 4), (4, 8), (8, 13), (13, 30)]
    ]
    q = p[..., (f >= 1) & (f <= 40)]
    q = q / np.maximum(q.sum(-1, keepdims=True), 1e-20)
    features.append(-(q * np.log(q + 1e-20)).sum(-1) / np.log(q.shape[-1]))
    return np.stack(features, -1).astype("float32")


def represent(x, name):
    if name == "transformer":
        return spectral_tokens(x)
    if name == "wavelet_cnn":
        # Morlet scalograms: 12 frequencies, 64 time bins, all shared channels.
        out = []
        for f in np.geomspace(2, 35, 12):
            sigma = 5 / (2 * np.pi * f)
            t = np.arange(-3.5 * sigma, 3.5 * sigma, 1 / 256)
            wave = np.exp(2j * np.pi * f * t) * np.exp(-t * t / (2 * sigma * sigma))
            wave /= np.sqrt(np.sum(abs(wave) ** 2))
            z = signal.fftconvolve(x, wave[None, None, :], mode="same", axes=-1)
            out.append(
                np.log10(np.maximum((abs(z) ** 2).reshape(*x.shape[:2], 64, 16).mean(-1), 1e-12))
            )
        return np.stack(out, 2).astype("float32")
    return x


def periodic_features(x):
    """Transparent spectral parameterization approximation, not specparam/FOOOF."""
    f, p = signal.welch(x, fs=256, nperseg=512, axis=-1)
    sel = (f >= 2) & (f <= 35)
    f = f[sel]
    y = np.log10(np.maximum(p[..., sel].mean(0), 1e-12))
    a = np.stack([np.ones_like(f), np.log10(f)], -1)
    coef = np.linalg.lstsq(a, y.T, rcond=None)[0]
    res = y - (a @ coef).T
    # One lower-residual refit reduces influence of oscillatory peaks.
    for c in range(len(y)):
        keep = res[c] <= np.quantile(res[c], 0.7)
        coef[:, c] = np.linalg.lstsq(a[keep], y[c, keep], rcond=None)[0]
    res = y - (a @ coef).T
    features = [coef[0], -coef[1]]
    for lo, hi in [(2, 4), (4, 8), (8, 13), (13, 30)]:
        features.append(np.maximum(res[:, (f >= lo) & (f < hi)], 0).mean(-1))
    features.extend([f[res.argmax(-1)], res.max(-1)])
    return np.stack(features, -1).ravel()
