from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scipy.stats import ttest_rel, wilcoxon


def load_metric_map(csv_path: Path, metric: str) -> dict[str, float]:
    rows = csv.DictReader(csv_path.open("r", encoding="utf-8"))
    result: dict[str, float] = {}
    for row in rows:
        sample_id = row["sample_id"]
        try:
            value = float(row[metric])
        except Exception:
            continue
        if value != value:
            continue
        result[sample_id] = value
    return result


def main(left_csv: str, right_csv: str, metrics: list[str], output_csv: str) -> None:
    left_path = Path(left_csv)
    right_path = Path(right_csv)
    rows: list[dict[str, str]] = []
    for metric in metrics:
        left = load_metric_map(left_path, metric)
        right = load_metric_map(right_path, metric)
        shared = sorted(set(left) & set(right))
        if not shared:
            continue
        left_vals = [left[key] for key in shared]
        right_vals = [right[key] for key in shared]
        try:
            t_stat, t_p = ttest_rel(left_vals, right_vals)
        except Exception:
            t_stat, t_p = float("nan"), float("nan")
        try:
            w_stat, w_p = wilcoxon(left_vals, right_vals)
        except Exception:
            w_stat, w_p = float("nan"), float("nan")
        rows.append(
            {
                "metric": metric,
                "n": str(len(shared)),
                "left_mean": f"{sum(left_vals) / len(left_vals):.6f}",
                "right_mean": f"{sum(right_vals) / len(right_vals):.6f}",
                "paired_t_p": f"{t_p:.6g}",
                "wilcoxon_p": f"{w_p:.6g}",
            }
        )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "n", "left_mean", "right_mean", "paired_t_p", "wilcoxon_p"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote significance report to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-csv", required=True)
    parser.add_argument("--right-csv", required=True)
    parser.add_argument("--metrics", default="clipscore,edge_consistency,blank_space_iou,blank_space_ssim")
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    main(args.left_csv, args.right_csv, metrics, args.output_csv)
