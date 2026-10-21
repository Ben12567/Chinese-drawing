from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


STRUCTURE_CHANNELS = [
    "lineart",
    "quantized_depth",
    "blank_space_mask",
    "salient_composition_mask",
]


@dataclass
class StructureMapResult:
    lineart: np.ndarray
    quantized_depth: np.ndarray
    blank_space_mask: np.ndarray
    salient_composition_mask: np.ndarray

    def stack(self) -> np.ndarray:
        return np.stack(
            [
                self.lineart,
                self.quantized_depth,
                self.blank_space_mask,
                self.salient_composition_mask,
            ],
            axis=-1,
        )


def load_rgb(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)


def _normalize_u8(array: np.ndarray) -> np.ndarray:
    array = np.clip(array, 0, 255)
    return array.astype(np.uint8)


def extract_lineart(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    inv = 255 - edges
    return _normalize_u8(inv)


def extract_blank_space_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_32F)
    variance = cv2.GaussianBlur(np.abs(variance), (9, 9), 0)
    texture = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX)
    bright = gray > 190
    smooth = texture < 32
    blank = np.where(bright & smooth, 255, 0).astype(np.uint8)
    blank = cv2.medianBlur(blank, 5)
    return blank


def extract_salient_composition_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    mag = cv2.GaussianBlur(mag, (15, 15), 0)
    sal = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    threshold = np.percentile(sal, 72)
    mask = np.where(sal >= threshold, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def extract_quantized_depth(rgb: np.ndarray, blank_space_mask: np.ndarray | None = None) -> np.ndarray:
    """A composition-aware pseudo-depth map for landscape paintings.

    This is not physical depth. It is a hierarchical depth proxy built from:
    - vertical position prior
    - local contrast
    - ink density
    - blank-space suppression
    """

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    vertical = np.linspace(0.2, 1.0, h, dtype=np.float32).reshape(h, 1)
    vertical = np.repeat(vertical, w, axis=1)
    contrast = cv2.Laplacian(gray, cv2.CV_32F)
    contrast = np.abs(cv2.GaussianBlur(contrast, (9, 9), 0))
    contrast = cv2.normalize(contrast, None, 0, 1, cv2.NORM_MINMAX)
    ink_density = 1.0 - (gray / 255.0)
    score = 0.45 * vertical + 0.35 * ink_density + 0.20 * contrast
    if blank_space_mask is not None:
        score = score * (1.0 - blank_space_mask.astype(np.float32) / 255.0)

    bins = np.quantile(score, [0.25, 0.5, 0.75])
    quantized = np.digitize(score, bins=bins, right=False).astype(np.uint8)
    quantized = quantized * 85
    return quantized


def extract_structure_maps(rgb: np.ndarray) -> StructureMapResult:
    blank = extract_blank_space_mask(rgb)
    return StructureMapResult(
        lineart=extract_lineart(rgb),
        quantized_depth=extract_quantized_depth(rgb, blank_space_mask=blank),
        blank_space_mask=blank,
        salient_composition_mask=extract_salient_composition_mask(rgb),
    )


def save_structure_map(result: StructureMapResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.stack(), mode="RGBA").save(path)
