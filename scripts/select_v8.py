"""Apply the pre-registered v8 selection rule (configs/revision/v8_preregistration.md) to
validation results: mean rank over KID, CMMD (lower is better) and the three condition-specific
structure metrics (higher is better); ties broken by KID."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

LOWER = ["kid", "cmmd"]
HIGHER = ["edge_iou_specific", "blank_iou_re_specific", "layer_acc_specific"]


def load(eval_dir: Path) -> dict:
    s = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    t = json.loads((eval_dir / "structure_v8_summary.json").read_text(encoding="utf-8"))
    return {"kid": s["plain"]["kid"], "cmmd": s["plain"]["cmmd"], "fid": s["plain"]["fid"], **{k: t[k] for k in HIGHER}}


def main(args: argparse.Namespace) -> None:
    rows = {name: load(Path(args.eval_root) / name) for name in args.candidates}
    df = pd.DataFrame(rows).T
    ranks = pd.concat([df[LOWER].rank(ascending=True), df[HIGHER].rank(ascending=False)], axis=1)
    df["mean_rank"] = ranks.mean(axis=1)
    df = df.sort_values(["mean_rank", "kid"])
    winner = df.index[0]
    print(df.to_string())
    print("selected:", winner)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"))
    out.write_text(json.dumps({"selected": winner, "candidates": args.candidates}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="outputs/revision/eval_val")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
