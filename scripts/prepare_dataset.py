from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

from clpgen.data.prompts import build_prompt_bundle
from clpgen.data.schema import PaintingSample, save_manifest
from clpgen.data.structure_maps import extract_structure_maps, load_rgb, save_structure_map


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def infer_basic_tags(path: Path) -> dict[str, str | list[str]]:
    stem = path.stem.lower()
    season = "秋"
    if "spring" in stem or "春" in stem:
        season = "春"
    elif "summer" in stem or "夏" in stem:
        season = "夏"
    elif "winter" in stem or "冬" in stem:
        season = "冬"

    weather = "云雾"
    if "rain" in stem or "雨" in stem:
        weather = "烟雨"
    elif "snow" in stem or "雪" in stem:
        weather = "雪景"

    style = "水墨写意式"
    if "mishi" in stem or "米" in stem:
        style = "米氏云山式"
    elif "dongyuan" in stem or "董源" in stem:
        style = "董源式"
    elif "qianjiang" in stem or "浅绛" in stem:
        style = "浅绛设色式"

    return {
        "subject": "山水",
        "season": season,
        "weather": weather,
        "foreground": ["松树", "溪亭"],
        "midground": ["山径", "水岸"],
        "background": ["远山", "云岚"],
        "ink_tone": "淡墨",
        "palette": "水墨" if "color" not in stem else "浅绛",
        "blankness": "大留白",
        "mood": "空灵",
        "style_label": style,
        "brushwork_label": "淡墨皴擦",
    }


def save_resized_image(src: Path, dst: Path, max_side: int = 1536) -> tuple[int, int]:
    image = Image.open(src).convert("RGB")
    width, height = image.size
    scale = min(max_side / max(width, height), 1.0)
    resized = image.resize((int(width * scale), int(height * scale)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    resized.save(dst)
    return resized.size


def split_name(index: int, train_cutoff: int, val_cutoff: int) -> str:
    if index < train_cutoff:
        return "train"
    if index < val_cutoff:
        return "val"
    return "test"


def main(images_dir: str, output_dir: str, split_name_prefix: str = "clp4k_v1", seed: int = 42) -> None:
    rng = random.Random(seed)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    image_paths = sorted([path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES])
    rng.shuffle(image_paths)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {images_dir}.")

    if len(image_paths) == 1:
        train_cutoff, val_cutoff = 1, 1
    elif len(image_paths) == 2:
        train_cutoff, val_cutoff = 1, 2
    else:
        train_cutoff = max(1, int(len(image_paths) * 0.8))
        val_cutoff = max(train_cutoff + 1, int(len(image_paths) * 0.9))
    samples: list[PaintingSample] = []

    for index, image_path in enumerate(image_paths):
        sample_id = f"{split_name_prefix}_{index:06d}"
        split = split_name(index, train_cutoff, val_cutoff)
        image_out = output_dir / "images" / f"{sample_id}.png"
        structure_out = output_dir / "structure_maps" / f"{sample_id}.png"
        width, height = save_resized_image(image_path, image_out)
        structure = extract_structure_maps(load_rgb(image_out))
        save_structure_map(structure, structure_out)

        tags = infer_basic_tags(image_path)
        prompts = build_prompt_bundle(
            subject=str(tags["subject"]),
            season=str(tags["season"]),
            weather=str(tags["weather"]),
            foreground=list(tags["foreground"]),
            midground=list(tags["midground"]),
            background=list(tags["background"]),
            ink_tone=str(tags["ink_tone"]),
            palette=str(tags["palette"]),
            blankness=str(tags["blankness"]),
            mood=str(tags["mood"]),
        )
        samples.append(
            PaintingSample(
                sample_id=sample_id,
                image_path=str(image_out.relative_to(output_dir)),
                structure_map_path=str(structure_out.relative_to(output_dir)),
                width=width,
                height=height,
                source="local_import",
                style_label=str(tags["style_label"]),
                brushwork_label=str(tags["brushwork_label"]),
                dense_caption_zh=prompts.dense_zh,
                dense_caption_en=prompts.dense_en,
                prompt_short_zh=prompts.short_zh,
                prompt_structured_zh=prompts.structured_zh,
                prompt_structured_en=prompts.structured_en,
                split=split,
            )
        )

    save_manifest(samples, output_dir / "manifest.jsonl")
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        ids = [sample.sample_id for sample in samples if sample.split == split]
        (split_dir / f"{split}.txt").write_text("\n".join(ids), encoding="utf-8")

    print(f"Prepared {len(samples)} samples at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="clp4k_v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.images_dir, args.output_dir, split_name_prefix=args.split_name, seed=args.seed)
