from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from clpgen.evaluation.metrics import cronbach_alpha


DIMENSIONS = [
    "PromptConsistency",
    "CompositionStructure",
    "BrushworkCharm",
    "ArtisticQuality",
    "Creativity",
]


def _safe_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed:
        return None
    return parsed


def _icc_two_way_random_absolute(matrix: np.ndarray) -> float:
    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    mean_targets = matrix.mean(axis=1, keepdims=True)
    mean_raters = matrix.mean(axis=0, keepdims=True)
    grand_mean = matrix.mean()
    msr = k * np.square(mean_targets - grand_mean).sum() / (n - 1)
    msc = n * np.square(mean_raters - grand_mean).sum() / (k - 1)
    mse = np.square(matrix - mean_targets - mean_raters + grand_mean).sum() / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + (k * (msc - mse) / n)
    if denom <= 0:
        return float("nan")
    return float((msr - mse) / denom)


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _summarize_dimension(rows: list[dict[str, str]], dimension: str) -> dict[str, str]:
    method_values: dict[str, list[float]] = defaultdict(list)
    sample_rater_values: dict[str, dict[str, float]] = defaultdict(dict)
    raters: list[str] = []
    samples: list[str] = []

    for row in rows:
        method = row.get("Method", "unknown")
        sample_id = row.get("SampleID", "")
        rater_id = row.get("RaterID", "")
        value = _safe_float(row.get(dimension, ""))
        if value is None or not sample_id or not rater_id:
            continue
        method_values[method].append(value)
        sample_rater_values[sample_id][rater_id] = value
        if rater_id not in raters:
            raters.append(rater_id)
        if sample_id not in samples:
            samples.append(sample_id)

    matrix_rows: list[list[float]] = []
    for sample_id in samples:
        values = sample_rater_values[sample_id]
        if all(rater in values for rater in raters):
            matrix_rows.append([values[rater] for rater in raters])
    matrix = np.asarray(matrix_rows, dtype=np.float32) if matrix_rows else np.zeros((0, 0), dtype=np.float32)

    summary = {
        "Dimension": dimension,
        "NumRatings": str(sum(len(values) for values in method_values.values())),
        "CronbachAlpha": f"{cronbach_alpha(matrix):.6f}" if matrix.size else "",
        "ICC2_1": f"{_icc_two_way_random_absolute(matrix):.6f}" if matrix.size else "",
    }
    for method, values in sorted(method_values.items()):
        summary[f"{method}_Mean"] = f"{float(np.mean(values)):.6f}"
        summary[f"{method}_Std"] = f"{float(np.std(values, ddof=0)):.6f}"
    return summary


def main(input_csv: str, output_csv: str) -> None:
    rows = _load_rows(Path(input_csv))
    methods = sorted({row.get("Method", "unknown") for row in rows if row.get("Method")})
    summaries = [_summarize_dimension(rows, dimension) for dimension in DIMENSIONS]

    fieldnames = ["Dimension", "NumRatings", "CronbachAlpha", "ICC2_1"]
    for method in methods:
        fieldnames.extend([f"{method}_Mean", f"{method}_Std"])

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote expert-score analysis to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    main(args.input_csv, args.output_csv)
