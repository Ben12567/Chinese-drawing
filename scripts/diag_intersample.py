"""Inter-sample diversity of a generation run (LoRA-only diagnosis, R1 major issue 4).

For the images generated with one seed, compute the mean LPIPS between images generated for
DIFFERENT test samples, and the fraction of pairs that are near-duplicates (LPIPS < 0.1).
A model that ignores its per-sample inputs, or receives identical inputs, scores near zero.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from clpgen.evaluation.metrics import load_rgb


@torch.no_grad()
def main(pred_dir: str, seed: str, out: str) -> None:
    import lpips

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    paths = sorted(Path(pred_dir).glob(f"*__seed{seed}.png"))
    tensors = []
    for p in paths:
        arr = np.asarray(load_rgb(p, image_size=256)).astype(np.float32) / 127.5 - 1.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    batch = torch.stack(tensors).to(device)
    values = []
    pairs = list(combinations(range(len(paths)), 2))
    for start in range(0, len(pairs), 256):
        chunk = pairs[start : start + 256]
        a = batch[[i for i, _ in chunk]]
        b = batch[[j for _, j in chunk]]
        values.extend(metric(a, b).flatten().cpu().tolist())
    values = np.array(values)
    result = {"n_images": len(paths), "n_pairs": len(values), "inter_sample_lpips": float(values.mean()),
              "near_duplicate_fraction": float((values < 0.1).mean())}
    Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--seed", default="42")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.pred_dir, args.seed, args.out)
