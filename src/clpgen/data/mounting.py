"""Heuristics for museum-scan artefacts: mounting borders and red seal impressions.

Mountings (silk/paper borders of hanging scrolls and handscrolls, photographic background
around album leaves and fans) are removed by cropping to the painted area. Seals are not
removed from training images, because they are part of the pictorial format; they are
detected so that their prevalence can be reported and so that a seal-masked variant of the
fidelity metrics can be computed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class CropResult:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in full-resolution pixels
    cropped: bool
    kept_area_fraction: float
    sides: dict[str, int]  # pixels removed per side at full resolution

    def to_dict(self) -> dict:
        return asdict(self)


def _edge_density(mag: np.ndarray, thr: float) -> float:
    return float((mag > thr).mean()) if mag.size else 0.0


def _trim_uniform_margins(lum: np.ndarray, max_frac: float = 0.25, std_thr: float = 4.0) -> tuple[int, int, int, int]:
    h, w = lum.shape
    y0, y1, x0, x1 = 0, h, 0, w
    while y0 < int(h * max_frac) and lum[y0, x0:x1].std() < std_thr:
        y0 += 1
    while y1 > h - int(h * max_frac) and lum[y1 - 1, x0:x1].std() < std_thr:
        y1 -= 1
    while x0 < int(w * max_frac) and lum[y0:y1, x0].std() < std_thr:
        x0 += 1
    while x1 > w - int(w * max_frac) and lum[y0:y1, x1 - 1].std() < std_thr:
        x1 -= 1
    return x0, y0, x1, y1


def _boundary_lines(grad: np.ndarray, coverage_thr: float, grad_thr: float) -> np.ndarray:
    """Lines along which a brightness step of CONSISTENT sign exceeds grad_thr over at least
    coverage_thr of the positions (1-px tolerance). Mount and margin boundaries are straight,
    span the whole picture and have one sign; ink strokes inside a painting do not."""
    # Coverage is measured over the central 60% of each line, so that the edge of a painting
    # mounted with margins on both sides (which spans only the painting's width) still counts.
    n = grad.shape[1]
    grad = grad[:, int(0.2 * n) : max(int(0.8 * n), int(0.2 * n) + 1)]
    coverage = np.zeros(grad.shape[0])
    for strong in (grad > grad_thr, grad < -grad_thr):
        strong = cv2.dilate(strong.astype(np.uint8), np.ones((3, 1), np.uint8))
        coverage = np.maximum(coverage, strong.mean(axis=1))
    return coverage >= coverage_thr


def detect_painting_box(
    rgb: np.ndarray,
    analysis_side: int = 800,
    band_tb: float = 0.45,
    band_lr: float = 0.20,
    coverage_thr: float = 0.85,
    grad_thr: float = 10.0,
    min_keep: float = 0.35,
) -> CropResult:
    h_full, w_full = rgb.shape[:2]
    scale = min(1.0, analysis_side / max(h_full, w_full))
    small = cv2.resize(rgb, (max(1, round(w_full * scale)), max(1, round(h_full * scale))), interpolation=cv2.INTER_AREA)
    lum = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    lum = cv2.GaussianBlur(lum, (3, 3), 0)
    h, w = lum.shape

    x0, y0, x1, y1 = _trim_uniform_margins(lum)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    mag = np.hypot(gx, gy)

    def inner_density(ax0: int, ay0: int, ax1: int, ay1: int) -> float:
        return _edge_density(mag[ay0:ay1, ax0:ax1], 25.0)

    # Horizontal mount boundaries (top/bottom).
    rows = _boundary_lines(gy[:, x0:x1], coverage_thr, grad_thr)
    band = int((y1 - y0) * band_tb)
    top_candidates = [y for y in range(y0 + 1, y0 + band) if rows[y]]
    bottom_candidates = [y for y in range(y1 - band, y1 - 1) if rows[y]]
    new_y0, new_y1 = y0, y1
    if top_candidates:
        cand = max(top_candidates) + 2
        if inner_density(x0, y0, x1, cand) < 0.6 * inner_density(x0, cand, x1, y1):
            new_y0 = cand
    if bottom_candidates:
        cand = min(bottom_candidates) - 1
        if inner_density(x0, cand, x1, y1) < 0.6 * inner_density(x0, new_y0, x1, cand):
            new_y1 = cand
    if (new_y1 - new_y0) >= min_keep * h:
        y0, y1 = new_y0, new_y1

    # Vertical mount boundaries (left/right).
    cols = _boundary_lines(gx[y0:y1, :].T, coverage_thr, grad_thr)
    band = int((x1 - x0) * band_lr)
    left_candidates = [x for x in range(x0 + 1, x0 + band) if cols[x]]
    right_candidates = [x for x in range(x1 - band, x1 - 1) if cols[x]]
    new_x0, new_x1 = x0, x1
    if left_candidates:
        cand = max(left_candidates) + 2
        if inner_density(x0, y0, cand, y1) < 0.6 * inner_density(cand, y0, x1, y1):
            new_x0 = cand
    if right_candidates:
        cand = min(right_candidates) - 1
        if inner_density(cand, y0, x1, y1) < 0.6 * inner_density(new_x0, y0, cand, y1):
            new_x1 = cand
    if (new_x1 - new_x0) >= min_keep * w:
        x0, x1 = new_x0, new_x1

    inv = 1.0 / scale
    fx0, fy0 = int(round(x0 * inv)), int(round(y0 * inv))
    fx1, fy1 = min(w_full, int(round(x1 * inv))), min(h_full, int(round(y1 * inv)))
    kept = (fx1 - fx0) * (fy1 - fy0) / float(w_full * h_full)
    sides = {"left": fx0, "top": fy0, "right": w_full - fx1, "bottom": h_full - fy1}
    cropped = any(v > 0.01 * max(w_full, h_full) for v in sides.values())
    if not cropped:
        return CropResult((0, 0, w_full, h_full), False, 1.0, {k: 0 for k in sides})
    return CropResult((fx0, fy0, fx1, fy1), True, float(kept), sides)


def red_seal_mask(rgb: np.ndarray, min_area_frac: float = 3e-4) -> tuple[np.ndarray, int]:
    """Binary mask of compact saturated-red regions (seal impressions) and their count."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((hue <= 10) | (hue >= 165)) & (sat >= 90) & (val >= 50)
    red = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
    area_min = min_area_frac * rgb.shape[0] * rgb.shape[1]
    mask = np.zeros(red.shape, dtype=np.uint8)
    count = 0
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if area < area_min:
            continue
        aspect = w / max(h, 1)
        fill = area / float(max(w * h, 1))
        if 0.3 <= aspect <= 3.3 and fill >= 0.35:
            mask[labels == idx] = 255
            count += 1
    return mask, count


def inpaint_seals(rgb: np.ndarray) -> np.ndarray:
    mask, count = red_seal_mask(rgb)
    if count == 0:
        return rgb
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
