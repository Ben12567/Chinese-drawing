from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoraConfig:
    rank: int = 16
    alpha: int = 16
    target_modules: list[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"])
    train_text_encoder: bool = True


@dataclass
class HierarchicalAdapterConfig:
    in_channels: int = 4
    base_channels: int = 64
    token_dim: int = 256
    levels: int = 4


@dataclass
class ProjectConfig:
    raw: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(raw=data)

    def get(self, *keys: str, default: Any = None) -> Any:
        value: Any = self.raw
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value
