"""Shared helpers for the revision figures (PLOS: TIFF, RGB, 300 dpi, width <= 7.5 in)."""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.linewidth": 0.6})


def save_tiff(fig, out: str | Path, dpi: int = 300) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    image.save(out, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
    image.save(out.with_suffix(".png"))


def channel_images(structure_map_path: str | Path) -> list[np.ndarray]:
    """The four channels of a stored structure map, rendered for display (line art dark on white)."""
    arr = np.asarray(Image.open(structure_map_path).convert("RGBA"))
    return [arr[..., i] for i in range(4)]
