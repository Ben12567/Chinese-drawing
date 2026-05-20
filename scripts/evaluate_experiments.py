from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import shutil

from clpgen.data.schema import load_manifest
from clpgen.evaluation.metrics import (
    aggregate_structure_consistency,
    compute_clipscore,
    compute_fid_kid,
    compute_hpsv2,
    compute_image_reward,
    compute_lpips_diversity,
    compute_pickscore,
    compute_style_accuracy,
    group_prediction_paths,
    structure_consistency,
)


PROMPT_FIELD_MAP = {
    "short": "prompt_short_zh",
    "structured": "prompt_structured_zh",
    "dense": "dense_caption_zh",
}


def _cleanup_accelerator_state() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _copy_primary_predictions(groups: dict[str, list[Path]], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary: dict[str, Path] = {}
    for sample_id, paths in groups.items():
        src = sorted(paths)[0]
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        primary[sample_id] = dst
    return primary


def _collect_rows(
    manifest_path: str,
    prediction_dir: str,
    split: str,
    prompt_mode: str,
) -> tuple[list[dict], dict[str, list[Path]]]:
    samples = [sample for sample in load_manifest(manifest_path) if sample.split == split]
    prompt_field = PROMPT_FIELD_MAP[prompt_mode]
    prediction_groups = group_prediction_paths(prediction_dir)
    rows: list[dict] = []
    for sample in samples:
        pred_paths = prediction_groups.get(sample.sample_id, [])
        if not pred_paths:
            continue
        for pred_path in pred_paths:
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "prediction_path": pred_path,
                    "reference_image_path": Path(manifest_path).parent / sample.image_path,
                    "reference_structure_path": Path(manifest_path).parent / sample.structure_map_path,
                    "prompt": getattr(sample, prompt_field),
                    "style_label": sample.style_label,
                }
            )
    return rows, prediction_groups


def main(
    manifest_path: str,
    prediction_dir: str,
    split: str,
    prompt_mode: str,
    output_dir: str,
) -> None:
    rows, prediction_groups = _collect_rows(
        manifest_path=manifest_path,
        prediction_dir=prediction_dir,
        split=split,
        prompt_mode=prompt_mode,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_root / "per_sample_metrics.csv"
    summary_path = output_root / "summary_metrics.json"
    primary_dir = output_root / "primary_predictions"

    primary_predictions = _copy_primary_predictions(prediction_groups, primary_dir)
    reference_dir = output_root / "reference_images"
    reference_dir.mkdir(parents=True, exist_ok=True)
    seen_refs: set[str] = set()
    primary_image_paths: list[Path] = []
    primary_prompts: list[str] = []
    primary_style_labels: list[str] = []
    structure_pairs: list[tuple[Path, Path]] = []
    primary_rows: list[dict] = []

    for row in rows:
        sample_id = row["sample_id"]
        primary_original = sorted(prediction_groups.get(sample_id, []))[0] if prediction_groups.get(sample_id) else None
        if primary_original != row["prediction_path"]:
            continue
        ref_path = Path(row["reference_image_path"])
        ref_copy = reference_dir / ref_path.name
        if ref_path.name not in seen_refs:
            shutil.copy2(ref_path, ref_copy)
            seen_refs.add(ref_path.name)
        primary_image_paths.append(row["prediction_path"])
        primary_prompts.append(row["prompt"])
        primary_style_labels.append(row["style_label"])
        structure_pairs.append((Path(row["reference_structure_path"]), row["prediction_path"]))
        primary_rows.append(row)

    clipscore = compute_clipscore(primary_image_paths, primary_prompts)
    _cleanup_accelerator_state()
    image_reward = compute_image_reward(primary_image_paths, primary_prompts)
    _cleanup_accelerator_state()
    pickscore = compute_pickscore(primary_image_paths, primary_prompts)
    _cleanup_accelerator_state()
    hpsv2 = compute_hpsv2(primary_image_paths, primary_prompts)
    _cleanup_accelerator_state()
    structure_summary = aggregate_structure_consistency(structure_pairs)
    fid_kid = compute_fid_kid(reference_dir, primary_dir)
    _cleanup_accelerator_state()
    lpips_diversity = compute_lpips_diversity(prediction_groups)
    _cleanup_accelerator_state()

    manifest = load_manifest(manifest_path)
    train_samples = [sample for sample in manifest if sample.split == "train"]
    train_image_paths = [Path(manifest_path).parent / sample.image_path for sample in train_samples]
    train_style_labels = [sample.style_label for sample in train_samples]
    style_accuracy = compute_style_accuracy(
        reference_image_paths=train_image_paths,
        reference_labels=train_style_labels,
        prediction_image_paths=primary_image_paths,
        target_labels=primary_style_labels,
    )
    _cleanup_accelerator_state()

    per_image_clip = clipscore.get("per_image", [])
    per_image_reward = image_reward.get("per_image", [])
    per_image_pickscore = pickscore.get("per_image", [])
    per_image_hpsv2 = hpsv2.get("per_image", [])
    per_image_style = style_accuracy.get("per_image", [])
    per_sample_rows: list[dict[str, str | float]] = []
    for index, row in enumerate(primary_rows):
        structure = structure_consistency(Path(row["reference_structure_path"]), row["prediction_path"])
        per_sample_rows.append(
            {
                "sample_id": row["sample_id"],
                "prediction_path": str(row["prediction_path"]),
                "clipscore": per_image_clip[index] if index < len(per_image_clip) else float("nan"),
                "image_reward": per_image_reward[index] if index < len(per_image_reward) else float("nan"),
                "pickscore": per_image_pickscore[index] if index < len(per_image_pickscore) else float("nan"),
                "hpsv2": per_image_hpsv2[index] if index < len(per_image_hpsv2) else float("nan"),
                "style_correct": per_image_style[index] if index < len(per_image_style) else float("nan"),
                **structure,
            }
        )

    with per_sample_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "prediction_path",
                "clipscore",
                "image_reward",
                "pickscore",
                "hpsv2",
                "style_correct",
                "edge_consistency",
                "blank_space_iou",
                "blank_space_ssim",
            ],
        )
        writer.writeheader()
        writer.writerows(per_sample_rows)

    summary = {
        "num_primary_predictions": len(primary_image_paths),
        "fid": fid_kid["fid"],
        "kid": fid_kid["kid"],
        "clipscore": clipscore["clipscore"],
        "image_reward": image_reward["image_reward"],
        "pickscore": pickscore["pickscore"],
        "hpsv2": hpsv2["hpsv2"],
        "lpips_diversity": lpips_diversity["lpips_diversity"],
        "style_accuracy": style_accuracy["style_accuracy"],
        **structure_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote per-sample metrics to {per_sample_path}")
    print(f"Wrote summary metrics to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--prompt-mode", choices=["short", "structured", "dense"], default="structured")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(
        manifest_path=args.manifest_path,
        prediction_dir=args.prediction_dir,
        split=args.split,
        prompt_mode=args.prompt_mode,
        output_dir=args.output_dir,
    )
