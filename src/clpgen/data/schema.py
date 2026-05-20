from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class PromptBundle:
    short_zh: str
    structured_zh: str
    structured_en: str
    dense_zh: str
    dense_en: str


@dataclass
class PaintingSample:
    sample_id: str
    image_path: str
    structure_map_path: str
    style_reference_path: str = ""
    width: int = 0
    height: int = 0
    source: str = "unknown"
    painter: str = "unknown"
    era: str = "unknown"
    title: str = ""
    object_id: str = ""
    object_url: str = ""
    image_url: str = ""
    license: str = ""
    style_label: str = "unknown"
    brushwork_label: str = "unknown"
    culture: str = ""
    department: str = ""
    medium: str = ""
    period: str = ""
    dynasty: str = ""
    tags: list[str] = field(default_factory=list)
    width_raw: int = 0
    height_raw: int = 0
    dense_caption_zh: str = ""
    dense_caption_en: str = ""
    prompt_short_zh: str = ""
    prompt_structured_zh: str = ""
    prompt_structured_en: str = ""
    structure_channels: list[str] = field(
        default_factory=lambda: [
            "lineart",
            "quantized_depth",
            "blank_space_mask",
            "salient_composition_mask",
        ]
    )
    split: str = "train"
    quality_score: float = 0.0
    quality_flags: list[str] = field(default_factory=list)
    duplicate_group: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaintingSample":
        return cls(**data)


def load_manifest(path: str | Path) -> list[PaintingSample]:
    samples: list[PaintingSample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(PaintingSample.from_dict(json.loads(line)))
    return samples


def save_manifest(samples: list[PaintingSample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample.to_json() + "\n")
