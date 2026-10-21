from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import cv2
import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError


Image.MAX_IMAGE_PIXELS = None


@dataclass
class ImageQualityReport:
    width: int
    height: int
    aspect_ratio: float
    blur_score: float
    entropy: float
    border_score: float
    text_like_score: float
    blank_ratio: float
    mean_brightness: float
    phash: str
    quality_score: float
    flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def load_image_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def phash_hex(rgb: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized)
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low[1:, 1:])
    bits = dct_low > median
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_hex(a: str, b: str) -> int:
    return int(bin(int(a, 16) ^ int(b, 16)).count("1"))


def shannon_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256]).ravel()
    prob = hist / max(hist.sum(), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def border_score(gray: np.ndarray, border_frac: float = 0.08) -> float:
    h, w = gray.shape
    bx = max(1, int(w * border_frac))
    by = max(1, int(h * border_frac))
    mask = np.zeros_like(gray, dtype=bool)
    mask[:by, :] = True
    mask[-by:, :] = True
    mask[:, :bx] = True
    mask[:, -bx:] = True
    border = gray[mask]
    center = gray[~mask]
    if len(center) == 0:
        return 0.0
    return float(abs(border.mean() - center.mean()) / 255.0)


def blank_space_ratio(gray: np.ndarray) -> float:
    return float((gray > 235).mean())


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def text_like_score(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    merged = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    score = 0.0
    area = gray.shape[0] * gray.shape[1]
    for idx in range(1, num_labels):
        _, _, w, h, comp_area = stats[idx]
        if w > h * 2 and comp_area > area * 0.002:
            score += comp_area / area
    return float(score)


def evaluate_image_quality(path: str | Path) -> ImageQualityReport:
    rgb = load_image_rgb(path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    width = int(rgb.shape[1])
    height = int(rgb.shape[0])
    aspect = width / max(height, 1)
    blur = blur_score(gray)
    entropy = shannon_entropy(gray)
    border = border_score(gray)
    text_score = text_like_score(gray)
    blank_ratio = blank_space_ratio(gray)
    brightness = float(gray.mean())
    p_hash = phash_hex(rgb)

    flags: list[str] = []
    if min(width, height) < 512:
        flags.append("low_resolution")
    if aspect < 0.3 or aspect > 3.5:
        flags.append("extreme_aspect_ratio")
    if blur < 15:
        flags.append("too_blurry")
    if entropy < 4.0:
        flags.append("low_information")
    if border > 0.22:
        flags.append("heavy_border")
    if text_score > 0.03:
        flags.append("possible_inscription_or_text")
    if blank_ratio > 0.8:
        flags.append("too_much_blank")

    quality_score = (
        min(1.0, blur / 120.0) * 0.25
        + min(1.0, entropy / 7.5) * 0.2
        + (1.0 - min(1.0, border / 0.35)) * 0.2
        + (1.0 - min(1.0, text_score / 0.08)) * 0.15
        + (1.0 - abs(blank_ratio - 0.35)) * 0.1
        + (1.0 - abs(brightness / 255.0 - 0.6)) * 0.1
    )
    quality_score = float(max(0.0, min(1.0, quality_score)))

    return ImageQualityReport(
        width=width,
        height=height,
        aspect_ratio=float(aspect),
        blur_score=float(blur),
        entropy=float(entropy),
        border_score=float(border),
        text_like_score=float(text_score),
        blank_ratio=float(blank_ratio),
        mean_brightness=brightness,
        phash=p_hash,
        quality_score=quality_score,
        flags=flags,
    )


def deduplicate_reports(reports: list[dict[str, Any]], hamming_threshold: int = 6) -> list[dict[str, Any]]:
    reports = sorted(reports, key=lambda item: item["quality"]["quality_score"], reverse=True)
    kept: list[dict[str, Any]] = []
    for report in reports:
        current_hash = report["quality"]["phash"]
        if not current_hash:
            report["duplicate_of"] = ""
            continue
        duplicate = False
        for existing in kept:
            existing_hash = existing["quality"]["phash"]
            if not existing_hash:
                continue
            if hamming_hex(current_hash, existing_hash) <= hamming_threshold:
                report["duplicate_of"] = existing["sample_id"]
                duplicate = True
                break
        if not duplicate:
            report["duplicate_of"] = ""
            kept.append(report)
    return reports


def save_reports(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
