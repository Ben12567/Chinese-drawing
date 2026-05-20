from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random

from PIL import Image

from clpgen.data.collection import load_jsonl
from clpgen.data.prompts import build_prompt_bundle
from clpgen.data.schema import PaintingSample, save_manifest
from clpgen.data.structure_maps import extract_structure_maps, load_rgb, save_structure_map


STYLE_KEYWORDS = {
    "mi fu": "mi_style_cloud_mountains",
    "mi youren": "mi_style_cloud_mountains",
    "dong yuan": "dong_yuan_style",
    "light color": "light_reddish_landscape",
    "slight color": "light_reddish_landscape",
    "ink and color": "blue_green_landscape",
    "color on silk": "blue_green_landscape",
    "color on paper": "blue_green_landscape",
    "ink on silk": "ink_wash_freehand",
    "ink on paper": "ink_wash_freehand",
}


def infer_style(row: dict) -> str:
    text = " ".join(
        [
            str(row.get("title", "")),
            str(row.get("artist", "")),
            str(row.get("medium", "")),
            " ".join(row.get("tags", []) or []),
        ]
    ).lower()
    for keyword, style in STYLE_KEYWORDS.items():
        if keyword in text:
            return style
    return "ink_wash_freehand"


def infer_prompt_fields(row: dict) -> dict:
    text = " ".join(
        [
            str(row.get("title", "")),
            str(row.get("medium", "")),
            str(row.get("period", "")),
            str(row.get("dynasty", "")),
            " ".join(row.get("tags", []) or []),
        ]
    ).lower()
    season = "autumn"
    weather = "cloud and mist"
    if "snow" in text or "winter" in text:
        season, weather = "winter", "snow scene"
    elif "spring" in text:
        season, weather = "spring", "light haze"
    elif "summer" in text:
        season, weather = "summer", "after rain"

    palette = "ink wash"
    if any(token in text for token in ["light color", "slight color"]):
        palette = "light reddish color"
    elif any(token in text for token in ["ink and color", "color on silk", "color on paper", "green", "blue"]):
        palette = "blue-green color"

    foreground = ["pine trees", "waterside pavilion"]
    midground = ["mountain path", "riverbank"]
    background = ["distant mountains", "clouds and mist"]
    if "waterfall" in text:
        midground = ["waterfall", "mountain path"]
    if "boat" in text:
        foreground = ["fishing boat", "pine trees"]
    if "temple" in text or "pavilion" in text:
        foreground = ["pavilion", "old trees"]
    return {
        "subject": "Chinese landscape painting",
        "season": season,
        "weather": weather,
        "foreground": foreground,
        "midground": midground,
        "background": background,
        "ink_tone": "light ink wash",
        "palette": palette,
        "blankness": "ample blank space",
        "mood": "serene",
    }


def assign_split(groups: list[tuple[str, list[dict]]], seed: int = 42) -> dict[str, str]:
    rng = random.Random(seed)
    total = sum(len(rows) for _, rows in groups)
    if total <= 1:
        targets = {"train": 1, "val": 0, "test": 0}
    elif total == 2:
        targets = {"train": 1, "val": 1, "test": 0}
    else:
        train_target = max(1, int(total * 0.8))
        val_target = max(1, int(total * 0.1))
        targets = {
            "train": train_target,
            "val": val_target,
            "test": max(1, total - train_target - val_target),
        }

    shuffled = list(groups)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda item: len(item[1]), reverse=True)

    split_map: dict[str, str] = {}
    counts = {name: 0 for name in targets}
    for _, rows in shuffled:
        group_size = len(rows)
        deficits = {name: targets[name] - counts[name] for name in targets}
        eligible = [name for name, deficit in deficits.items() if deficit >= group_size]
        if eligible:
            split = max(eligible, key=lambda name: (deficits[name], -counts[name], 1 if name == "test" else 0))
        else:
            split = min(targets, key=lambda name: counts[name] / max(targets[name], 1))
        counts[split] += group_size
        for row in rows:
            split_map[row["sample_id"]] = split
    return split_map


def assign_style_reference_paths(samples: list[PaintingSample]) -> None:
    by_style_split: dict[tuple[str, str], list[PaintingSample]] = defaultdict(list)
    by_style: dict[str, list[PaintingSample]] = defaultdict(list)
    ordered = sorted(samples, key=lambda sample: sample.sample_id)
    for sample in ordered:
        by_style_split[(sample.style_label, sample.split)].append(sample)
        by_style[sample.style_label].append(sample)

    for sample in ordered:
        candidates = [
            candidate
            for candidate in by_style_split[(sample.style_label, sample.split)]
            if candidate.sample_id != sample.sample_id
        ]
        if not candidates:
            candidates = [candidate for candidate in by_style[sample.style_label] if candidate.sample_id != sample.sample_id]
        if not candidates:
            candidates = [candidate for candidate in ordered if candidate.sample_id != sample.sample_id]
        if not candidates:
            sample.style_reference_path = sample.image_path
            continue
        offset = int(sample.sample_id.split("_")[-1]) % len(candidates)
        sample.style_reference_path = candidates[offset].image_path


def main(
    cleaned_report: str,
    output_dir: str,
    split_name: str = "clp4k_v2",
    target_size: int = 1536,
    seed: int = 42,
) -> None:
    rows = [row for row in load_jsonl(cleaned_report) if row.get("keep")]
    grouped: dict[str, list[dict]] = {}
    for index, row in enumerate(rows):
        meta = row.get("metadata", {})
        painter = str(meta.get("artist", "")).strip()
        title = str(meta.get("title", "")).strip()
        group_key = painter or title or str(meta.get("source", "")).strip() or "unknown"
        row["sample_id"] = f"{split_name}_{index:06d}"
        grouped.setdefault(group_key, []).append(row)
    split_map = assign_split(list(grouped.items()), seed=seed)

    output_root = Path(output_dir)
    images_dir = output_root / "images"
    structure_dir = output_root / "structure_maps"
    samples: list[PaintingSample] = []

    for row in rows:
        meta = row["metadata"]
        sample_id = row["sample_id"]
        split = split_map[sample_id]
        src = Path(row["image_path"])
        image_out = images_dir / f"{sample_id}.png"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(src).convert("RGB")
        width_raw, height_raw = image.size
        scale = min(target_size / max(width_raw, height_raw), 1.0)
        image = image.resize((int(width_raw * scale), int(height_raw * scale)))
        image.save(image_out)

        structure_out = structure_dir / f"{sample_id}.png"
        structure = extract_structure_maps(load_rgb(image_out))
        save_structure_map(structure, structure_out)

        prompt_fields = infer_prompt_fields(meta)
        prompts = build_prompt_bundle(**prompt_fields)
        samples.append(
            PaintingSample(
                sample_id=sample_id,
                image_path=str(image_out.relative_to(output_root)),
                structure_map_path=str(structure_out.relative_to(output_root)),
                width=image.size[0],
                height=image.size[1],
                width_raw=width_raw,
                height_raw=height_raw,
                source=str(meta.get("source", "")),
                painter=str(meta.get("artist", "")) or "unknown",
                era=str(meta.get("date_display", "")),
                title=str(meta.get("title", "")),
                object_id=str(meta.get("source_id", "")),
                object_url=str(meta.get("object_url", "")),
                image_url=str(meta.get("image_url", "")),
                license=str(meta.get("license", "")),
                style_label=infer_style(meta),
                brushwork_label="light ink texture strokes",
                culture=str(meta.get("culture", "")),
                department=str(meta.get("department", "")),
                medium=str(meta.get("medium", "")),
                period=str(meta.get("period", "")),
                dynasty=str(meta.get("dynasty", "")),
                tags=list(meta.get("tags", []) or []),
                dense_caption_zh=prompts.dense_zh,
                dense_caption_en=prompts.dense_en,
                prompt_short_zh=prompts.short_zh,
                prompt_structured_zh=prompts.structured_zh,
                prompt_structured_en=prompts.structured_en,
                split=split,
                quality_score=float(row["quality"]["quality_score"]),
                quality_flags=list(row["quality"]["flags"]),
                duplicate_group=str(row.get("duplicate_group", "")),
            )
        )

    assign_style_reference_paths(samples)
    save_manifest(samples, output_root / "manifest.jsonl")
    split_dir = output_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        ids = [sample.sample_id for sample in samples if sample.split == split]
        (split_dir / f"{split}.txt").write_text("\n".join(ids), encoding="utf-8")
    print(f"Built {len(samples)} cleaned paper-ready samples at {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="clp4k_v2")
    parser.add_argument("--target-size", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        cleaned_report=args.cleaned_report,
        output_dir=args.output_dir,
        split_name=args.split_name,
        target_size=args.target_size,
        seed=args.seed,
    )
