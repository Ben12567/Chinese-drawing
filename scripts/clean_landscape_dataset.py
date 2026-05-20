from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from clpgen.data.cleaning import deduplicate_reports, evaluate_image_quality, save_reports
from clpgen.data.collection import load_jsonl


def should_keep(flags: list[str], quality_score: float) -> bool:
    hard_drop = {"low_resolution", "extreme_aspect_ratio"}
    if any(flag in hard_drop for flag in flags):
        return False
    return quality_score >= 0.42


def main(
    candidate_manifest: str,
    cleaned_dir: str,
    report_path: str,
    copy_kept: bool = True,
) -> None:
    rows = load_jsonl(candidate_manifest)
    cleaned_root = Path(cleaned_dir)
    kept_dir = cleaned_root / "kept_images"
    rejected_dir = cleaned_root / "rejected_images"
    reports: list[dict] = []

    for index, row in enumerate(rows):
        image_path = row.get("download_path") or row.get("image_path")
        if not image_path:
            continue
        path = Path(image_path)
        if not path.exists():
            continue
        try:
            quality = evaluate_image_quality(path).to_dict()
        except Exception as exc:  # noqa: BLE001
            reports.append(
                {
                    "sample_id": row.get("source", "src") + "_" + row.get("source_id", str(index)),
                    "source": row.get("source", ""),
                    "source_id": row.get("source_id", ""),
                    "title": row.get("title", ""),
                    "image_path": str(path),
                    "quality": {
                        "width": 0,
                        "height": 0,
                        "aspect_ratio": 0.0,
                        "blur_score": 0.0,
                        "entropy": 0.0,
                        "border_score": 0.0,
                        "text_like_score": 0.0,
                        "blank_ratio": 0.0,
                        "mean_brightness": 0.0,
                        "phash": "",
                        "quality_score": 0.0,
                        "flags": ["unreadable_file"],
                    },
                    "metadata": row,
                    "keep": False,
                    "reason": f"unreadable_file: {exc}",
                    "duplicate_of": "",
                }
            )
            continue
        sample_id = row.get("source", "src") + "_" + row.get("source_id", str(index))
        reports.append(
            {
                "sample_id": sample_id,
                "source": row.get("source", ""),
                "source_id": row.get("source_id", ""),
                "title": row.get("title", ""),
                "image_path": str(path),
                "quality": quality,
                "metadata": row,
            }
        )

    reports = deduplicate_reports(reports)
    kept_count = 0
    for report in reports:
        duplicate = bool(report.get("duplicate_of"))
        flags = report["quality"]["flags"]
        keep = (not duplicate) and should_keep(flags, report["quality"]["quality_score"])
        report["keep"] = keep
        report["reason"] = "duplicate" if duplicate else ("pass" if keep else "quality_filter")
        src = Path(report["image_path"])
        if copy_kept:
            dst_root = kept_dir if keep else rejected_dir
            dst = dst_root / report["source"] / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            report["copied_path"] = str(dst)
        if keep:
            kept_count += 1

    save_reports(reports, report_path)
    print(f"Cleaned {len(reports)} candidates, kept {kept_count}. Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--cleaned-dir", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--copy-kept", action="store_true")
    args = parser.parse_args()
    main(
        candidate_manifest=args.candidate_manifest,
        cleaned_dir=args.cleaned_dir,
        report_path=args.report_path,
        copy_kept=args.copy_kept,
    )
