"""Small EEGNet-style classifier; not a pretrained or validated diagnostic model."""

import torch
from torch import nn


class EEGNet(nn.Module):
    def __init__(self, channels=18, f1=8, depth=2, dropout=0.5):
        super().__init__()
        f2 = f1 * depth
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, 128), padding=(0, 64), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, (channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(f2, f2, (1, 16), padding=(0, 8), groups=f2, bias=False),
            nn.Conv2d(f2, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(f2, 1)

    def forward(self, x):
        return self.classifier(self.features(x[:, None]).flatten(1)).squeeze(-1)

    @torch.no_grad()
    def constrain(self):
        # EEGNet spatial filters are bounded after each optimizer update.
        weight = self.features[2].weight
        norms = weight.flatten(1).norm(dim=1).clamp_min(1e-8)
        weight.mul_((1 / norms).clamp_max(1).view(-1, 1, 1, 1))


def device(name="auto"):
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name not in {"cpu", "mps"}:
        raise ValueError("Only the approved local CPU/MPS devices are allowed")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this process")
    return torch.device(name)
