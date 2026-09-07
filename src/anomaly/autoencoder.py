"""Small convolutional autoencoder for reconstruction-based sonar anomalies."""

from __future__ import annotations

import torch
from torch import nn


class SonarAutoencoder(nn.Module):
    """Encode and reconstruct a normalized grayscale sonar patch."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def reconstruction_error(model: nn.Module, patches: torch.Tensor) -> torch.Tensor:
    """Return one mean-squared reconstruction error per patch."""
    with torch.no_grad():
        reconstructed = model(patches)
        return ((reconstructed - patches) ** 2).flatten(1).mean(dim=1)
