from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main(manifest_path: str, output_path: str) -> None:
    manifest = Path(manifest_path)
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    painters: set[str] = set()
    widths: list[int] = []
    heights: list[int] = []

    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            split_counts[row.get("split", "")] += 1
            source_counts[row.get("source", "")] += 1
            style_counts[row.get("style_label", "")] += 1
            painter = row.get("painter", "")
            if painter and painter != "unknown":
                painters.add(painter)
            widths.append(int(row.get("width", 0) or 0))
            heights.append(int(row.get("height", 0) or 0))

    summary = {
        "manifest_path": str(manifest),
        "num_samples": sum(split_counts.values()),
        "num_sources": len(source_counts),
        "num_styles": len(style_counts),
        "num_painters": len(painters),
        "split_counts": dict(split_counts),
        "source_counts": dict(source_counts),
        "style_counts": dict(style_counts),
        "mean_width": sum(widths) / max(len(widths), 1),
        "mean_height": sum(heights) / max(len(heights), 1),
        "min_width": min(widths) if widths else 0,
        "max_width": max(widths) if widths else 0,
        "min_height": min(heights) if heights else 0,
        "max_height": max(heights) if heights else 0,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote dataset summary to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    main(manifest_path=args.manifest_path, output_path=args.output_path)
