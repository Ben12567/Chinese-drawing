from __future__ import annotations

import torch
from torch import nn


class StyleReferenceEncoder(nn.Module):
    """Fallback style encoder for experiments before CLIP/IP-Adapter integration."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x).flatten(1)
        return self.proj(h)
