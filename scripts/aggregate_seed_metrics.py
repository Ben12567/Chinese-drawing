from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean, pstdev


METRIC_FIELDS = [
    "fid",
    "kid",
    "clipscore",
    "image_reward",
    "pickscore",
    "hpsv2",
    "lpips_diversity",
    "style_accuracy",
    "edge_consistency",
    "blank_space_iou",
    "blank_space_ssim",
]


def _safe_float(value) -> float | None:
    try:
        value = float(value)
    except Exception:
        return None
    if value != value:
        return None
    return value


def _group_name(path: Path, strip_seed_suffix: bool) -> str:
    name = path.name
    if strip_seed_suffix:
        name = re.sub(r"[_-]seed\d+$", "", name)
    return name


def summarize_group(group_name: str, summaries: list[Path]) -> dict[str, str]:
    grouped: dict[str, list[float]] = {field: [] for field in METRIC_FIELDS}
    for path in summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        for field in METRIC_FIELDS:
            value = _safe_float(data.get(field))
            if value is not None:
                grouped[field].append(value)
    row = {"Method": group_name, "NumRuns": str(len(summaries))}
    for field, values in grouped.items():
        if not values:
            row[f"{field}_mean"] = ""
            row[f"{field}_std"] = ""
            continue
        row[f"{field}_mean"] = f"{mean(values):.6f}"
        row[f"{field}_std"] = f"{pstdev(values):.6f}" if len(values) > 1 else "0.000000"
    return row


def main(results_root: str, output_csv: str, strip_seed_suffix: bool) -> None:
    grouped_summaries: dict[str, list[Path]] = {}
    for summary_path in sorted(Path(results_root).glob("*/evaluation*/summary_metrics.json")):
        group = _group_name(summary_path.parent.parent, strip_seed_suffix=strip_seed_suffix)
        grouped_summaries.setdefault(group, []).append(summary_path)
    rows: list[dict[str, str]] = []
    for group_name, summaries in sorted(grouped_summaries.items()):
        rows.append(summarize_group(group_name, summaries))
    fieldnames = ["Method", "NumRuns"] + [f"{field}_{suffix}" for field in METRIC_FIELDS for suffix in ("mean", "std")]
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote seed-aggregated metrics to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--strip-seed-suffix", action="store_true")
    args = parser.parse_args()
    main(args.results_root, args.output_csv, args.strip_seed_suffix)
