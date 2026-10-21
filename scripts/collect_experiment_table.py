from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main(results_root: str, output_csv: str) -> None:
    rows: list[dict[str, str | float]] = []
    summary_candidates: dict[str, Path] = {}
    for summary_path in Path(results_root).glob("*/evaluation*/summary_metrics.json"):
        method = summary_path.parent.parent.name
        current = summary_candidates.get(method)
        if current is None:
            summary_candidates[method] = summary_path
            continue
        current_name = current.parent.name
        candidate_name = summary_path.parent.name
        if candidate_name == "evaluation_multiseed" and current_name != "evaluation_multiseed":
            summary_candidates[method] = summary_path
            continue
        if current_name == "evaluation_multiseed" and candidate_name != "evaluation_multiseed":
            continue
        if summary_path.stat().st_mtime > current.stat().st_mtime:
            summary_candidates[method] = summary_path
    for method, summary_path in sorted(summary_candidates.items()):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "Method": method,
                "FID": summary.get("fid", ""),
                "KID": summary.get("kid", ""),
                "CLIPScore": summary.get("clipscore", ""),
                "ImageReward": summary.get("image_reward", ""),
                "PickScore": summary.get("pickscore", ""),
                "HPSv2": summary.get("hpsv2", ""),
                "LPIPS_Diversity": summary.get("lpips_diversity", ""),
                "StructureConsistency": summary.get("edge_consistency", ""),
                "StyleAccuracy": summary.get("style_accuracy", ""),
                "Notes": f"blank_iou={summary.get('blank_space_iou', '')}",
            }
        )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Method",
                "FID",
                "KID",
                "CLIPScore",
                "ImageReward",
                "PickScore",
                "HPSv2",
                "LPIPS_Diversity",
                "StructureConsistency",
                "StyleAccuracy",
                "Notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} experiment rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    main(args.results_root, args.output_csv)
