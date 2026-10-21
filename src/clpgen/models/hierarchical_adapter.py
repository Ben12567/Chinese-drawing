from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=max(1, out_channels // 16), num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, out_channels // 16), num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HierarchicalStructureAdapter(nn.Module):
    """Encodes 4-channel structure maps into multi-scale control features."""

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        levels: int = 4,
        token_dim: int = 512,
        cross_attention_dim: int = 2048,
        pooled_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        if pooled_sizes is None:
            pooled_sizes = [16, 8, 4, 2]
        if len(pooled_sizes) != levels:
            raise ValueError("pooled_sizes must have the same length as levels.")

        self.stem = ConvBlock(in_channels, base_channels, downsample=False)
        channels = base_channels
        self.levels = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.pooled_sizes = pooled_sizes
        self.cross_attention_dim = cross_attention_dim
        for _ in range(levels):
            next_channels = channels * 2
            self.levels.append(ConvBlock(channels, next_channels, downsample=True))
            self.projections.append(nn.Conv2d(next_channels, token_dim, kernel_size=1))
            channels = next_channels
        self.cross_attn_projection = nn.Linear(token_dim, cross_attention_dim)

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        feats: list[torch.Tensor] = []
        tokens: list[torch.Tensor] = []
        x = self.stem(x)
        for pooled_size, level, projection in zip(self.pooled_sizes, self.levels, self.projections):
            x = level(x)
            feats.append(x)
            pooled = F.adaptive_avg_pool2d(x, output_size=(pooled_size, pooled_size))
            token_map = projection(pooled)
            tokens.append(token_map.flatten(2).transpose(1, 2))
        merged_tokens = torch.cat(tokens, dim=1)
        cross_attention_tokens = self.cross_attn_projection(merged_tokens)
        return {
            "features": feats,
            "tokens": tokens,
            "cross_attention_tokens": cross_attention_tokens,
        }
