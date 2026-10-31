"""Structure-adherence metrics with the extraction rules applied to the output (v8).

The v7 blank-space IoU thresholded the output at grey > 220, whereas the blank channel of the
condition is defined as grey > 190 with low texture; under that mismatch even a real painting
scores only ~0.16 against its own map. Here every output is resized to the size of its condition
map and passed through the same fixed rules that produced the map (structure_maps.py), and each
channel is compared with the corresponding condition channel:

  edge_iou      IoU of line-art edges (unchanged v7 definition, Canny 40/120)
  blank_iou_re  IoU of blank-space masks
  layer_acc     fraction of pixels whose quantised layer level (4 levels) agrees
  salience_iou  IoU of salience masks

Every metric is also computed against the condition map of a different test painting (a fixed
random derangement, identical for all methods). The matched-minus-mismatched difference measures
how much of the agreement is specific to the condition the model actually received, rather than
generic layout priors (e.g. light upper parts, dark lower parts).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from clpgen.data.schema import load_manifest
from clpgen.data.structure_maps import extract_structure_maps

METRICS = ["edge_iou", "blank_iou_re", "layer_acc", "salience_iou"]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def compare_all(cond: np.ndarray, out: np.ndarray) -> dict[str, float]:
    maps = extract_structure_maps(out)
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    return {
        # v7 edge IoU definition (Canny 40/120 on the output without blur)
        "edge_iou": _iou((255 - cond[..., 0].astype(np.int32)) > 127, cv2.Canny(gray, 40, 120) > 0),
        "blank_iou_re": _iou(cond[..., 2] > 127, maps.blank_space_mask > 127),
        "layer_acc": float((np.round(cond[..., 1] / 85.0) == np.round(maps.quantized_depth / 85.0)).mean()),
        "salience_iou": _iou(cond[..., 3] > 127, maps.salient_composition_mask > 127),
    }


def derangement(ids: list[str], seed: int = 2026) -> dict[str, str]:
    rng = random.Random(seed)
    order = sorted(ids)
    while True:
        perm = order[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(order, perm)):
            return dict(zip(order, perm))


def main(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest_path)
    root = manifest.parent
    samples = {s.sample_id: s for s in load_manifest(manifest)}
    split_ids = [sid for sid, s in samples.items() if s.split == args.split]
    partner = derangement(split_ids)
    groups: dict[str, list[Path]] = defaultdict(list)
    if args.real:
        for sid in split_ids:
            groups[sid].append(root / samples[sid].image_path)
    else:
        for p in sorted(Path(args.prediction_dir).glob("*.png")):
            sid = p.stem.split("__seed")[0]
            if sid in partner:
                groups[sid].append(p)
    maps: dict[str, np.ndarray] = {}

    def cond_map(sid: str) -> np.ndarray:
        if sid not in maps:
            maps[sid] = np.asarray(Image.open(root / samples[sid].structure_map_path).convert("RGBA"))
        return maps[sid]

    rows = []
    for sid in sorted(groups):
        for path in groups[sid]:
            img = Image.open(path).convert("RGB")
            row = {"sample_id": sid, "path": str(path)}
            for tag, other in [("", sid), ("_mismatch", partner[sid])]:
                cond = cond_map(other)
                out = np.asarray(img.resize((cond.shape[1], cond.shape[0]), Image.BICUBIC))
                for k, v in compare_all(cond, out).items():
                    row[k + tag] = v
            rows.append(row)
    df = pd.DataFrame(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "structure_v8_per_image.csv", index=False)
    per_sample = df.drop(columns=["path"]).groupby("sample_id").mean()
    for m in METRICS:
        per_sample[m + "_specific"] = per_sample[m] - per_sample[m + "_mismatch"]
    per_sample.to_csv(out_dir / "structure_v8_per_sample.csv")
    summary = {c: float(per_sample[c].mean()) for c in per_sample.columns}
    summary["n_samples"] = int(len(per_sample))
    summary["n_images"] = int(len(df))
    (out_dir / "structure_v8_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default="data/processed/clp_v7/manifest.jsonl")
    parser.add_argument("--prediction-dir", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--real", action="store_true", help="score the real paintings (reference row)")
    main(parser.parse_args())
