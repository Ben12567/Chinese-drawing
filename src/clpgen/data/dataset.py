from __future__ import annotations

from pathlib import Path
from typing import Literal
import warnings

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from clpgen.data.schema import PaintingSample, load_manifest
from clpgen.data.structure_maps import STRUCTURE_CHANNELS


PromptMode = Literal["short", "structured", "dense"]


class ChineseLandscapeDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path | None = None,
        split: str | None = None,
        prompt_mode: PromptMode = "structured",
        image_size: int = 768,
        structure_channels_active: list[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir) if root_dir else self.manifest_path.parent
        self.prompt_mode = prompt_mode
        self.image_size = image_size
        self.structure_channels_active = structure_channels_active or list(STRUCTURE_CHANNELS)
        self.samples = load_manifest(self.manifest_path)
        if split:
            self.samples = [sample for sample in self.samples if sample.split == split]
        self._warned_corrupt_paths: set[Path] = set()

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve(self, rel_path: str) -> Path:
        return (self.root_dir / rel_path).resolve()

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _load_structure_map(self, path: Path) -> torch.Tensor:
        try:
            image = Image.open(path).convert("RGBA").resize((self.image_size, self.image_size))
        except Exception:
            if path not in self._warned_corrupt_paths:
                warnings.warn(
                    f"Falling back to an empty structure map because the file could not be read: {path}",
                    RuntimeWarning,
                )
                self._warned_corrupt_paths.add(path)
            image = Image.new("RGBA", (self.image_size, self.image_size), (0, 0, 0, 0))
        array = np.asarray(image).astype(np.float32) / 255.0
        inactive = [index for index, name in enumerate(STRUCTURE_CHANNELS) if name not in self.structure_channels_active]
        if inactive:
            array[..., inactive] = 0.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _prompt(self, sample: PaintingSample) -> str:
        if self.prompt_mode == "short":
            return sample.prompt_short_zh
        if self.prompt_mode == "dense":
            return sample.dense_caption_zh
        return sample.prompt_structured_zh

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = self._load_image(self._resolve(sample.image_path))
        structure = self._load_structure_map(self._resolve(sample.structure_map_path))
        style_reference = None
        if sample.style_reference_path:
            style_reference = self._load_image(self._resolve(sample.style_reference_path))
        return {
            "sample_id": sample.sample_id,
            "pixel_values": image,
            "structure_map": structure,
            "style_reference": style_reference if style_reference is not None else torch.zeros_like(image),
            "prompt": self._prompt(sample),
            "style_label": sample.style_label,
            "brushwork_label": sample.brushwork_label,
        }
