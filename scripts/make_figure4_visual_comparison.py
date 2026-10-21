from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from clpgen.data.schema import load_manifest


SAMPLES = [
    {
        "id": "clp4k_v6_union_en_000001",
        "prompt": "Misty mountains with river pavilion",
        "style": "ink wash",
        "labels": {
            "lora": "weak layout",
            "controlnet": "rigid contour",
            "ip_adapter": "style drift",
            "ours": "balanced output",
        },
    },
    {
        "id": "clp4k_v6_union_en_000110",
        "prompt": "Blue-green landscape with layered peaks",
        "style": "blue-green",
        "labels": {
            "lora": "loose hierarchy",
            "controlnet": "hard edges",
            "ip_adapter": "layout drift",
            "ours": "coherent style",
        },
    },
    {
        "id": "clp4k_v6_union_en_000061",
        "prompt": "Ink-wash river valley with blank space",
        "style": "freehand ink",
        "labels": {
            "lora": "weak blank",
            "controlnet": "stiff strokes",
            "ip_adapter": "space drift",
            "ours": "controlled blank",
        },
    },
    {
        "id": "clp4k_v6_union_en_000300",
        "prompt": "Light-red autumn distant mountains",
        "style": "light red",
        "labels": {
            "lora": "unstable tone",
            "controlnet": "rigid shape",
            "ip_adapter": "soft layout",
            "ours": "balanced tone",
        },
    },
]

METHODS = [
    (
        "lora",
        "LoRA-only",
        Path("outputs/paper_suite_v6/paper_lora_only_817/predictions_multiseed"),
    ),
    (
        "controlnet",
        "ControlNet",
        Path("outputs/paper_suite_strong/paper_controlnet_817_seed42/predictions_multiseed"),
    ),
    (
        "ip_adapter",
        "IP-Adapter only",
        Path("outputs/paper_suite_strong/paper_ip_adapter_only_817_seed42/predictions_multiseed"),
    ),
    (
        "ours",
        "Ours",
        Path("outputs/paper_suite_v6/paper_ours_817/predictions_multiseed"),
    ),
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_image(path: Path, size: tuple[int, int], fill: str = "white") -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, fill)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _structure_preview(path: Path, size: tuple[int, int]) -> Image.Image:
    rgba = Image.open(path).convert("RGBA")
    arr = np.asarray(rgba)
    lineart = Image.fromarray(arr[..., 0]).convert("RGB")
    lineart = ImageOps.autocontrast(lineart)
    return _fit_image_from_image(lineart, size)


def _fit_image_from_image(image: Image.Image, size: tuple[int, int], fill: str = "white") -> Image.Image:
    image = image.convert("RGB")
    image = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, fill)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _draw_centered(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], font, fill) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - tw) // 2
    y = box[1] + (box[3] - box[1] - th) // 2
    draw.text((x, y), text, font=font, fill=fill)


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    width_chars: int,
    font,
    fill,
    line_gap: int = 4,
) -> int:
    y = xy[1]
    for line in textwrap.wrap(text, width=width_chars):
        draw.text((xy[0], y), line, font=font, fill=fill)
        bbox = draw.textbbox((xy[0], y), line, font=font)
        y += bbox[3] - bbox[1] + line_gap
    return y


def build_figure(manifest_path: Path, output_prefix: Path) -> None:
    root = manifest_path.parent
    samples_by_id = {sample.sample_id: sample for sample in load_manifest(manifest_path)}

    margin = 28
    gutter = 14
    header_h = 54
    row_h = 278
    condition_w = 285
    image_w = 226
    image_h = 226
    label_h = 34
    columns = [("condition", "Prompt / condition", None)] + [
        (key, title, directory) for key, title, directory in METHODS
    ]
    widths = [condition_w] + [image_w] * len(METHODS)
    total_w = margin * 2 + sum(widths) + gutter * (len(widths) - 1)
    total_h = margin * 2 + header_h + row_h * len(SAMPLES)

    bg = "#FFFFFF"
    border = "#D7D2CA"
    text = "#1F2933"
    muted = "#667085"
    accent = "#B5332E"

    canvas = Image.new("RGB", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)
    header_font = _font(17, bold=True)
    prompt_font = _font(15, bold=True)
    small_font = _font(12)
    tag_font = _font(13, bold=True)

    x = margin
    for col_idx, (_, title, _) in enumerate(columns):
        _draw_centered(draw, title, (x, margin, x + widths[col_idx], margin + header_h), header_font, text)
        x += widths[col_idx] + gutter

    y0 = margin + header_h
    for row_index, sample_cfg in enumerate(SAMPLES):
        sample = samples_by_id[sample_cfg["id"]]
        y = y0 + row_index * row_h
        x = margin

        draw.rounded_rectangle((x, y + 8, x + condition_w, y + row_h - 8), radius=4, outline=border, width=1, fill=bg)
        prompt_y = _draw_wrapped(draw, sample_cfg["prompt"], (x + 14, y + 18), 30, prompt_font, text, line_gap=3)
        draw.text((x + 14, prompt_y + 4), f"style: {sample_cfg['style']}", font=small_font, fill=muted)
        structure_path = root / sample.structure_map_path
        sketch = _structure_preview(structure_path, (condition_w - 28, 142))
        canvas.paste(sketch, (x + 14, y + row_h - 160))
        draw.rectangle((x + 14, y + row_h - 160, x + condition_w - 14, y + row_h - 18), outline=border, width=1)

        x += condition_w + gutter
        for key, _, directory in METHODS:
            image_path = directory / f"{sample.sample_id}__seed42.png"
            image = _fit_image(image_path, (image_w, image_h))
            draw.rectangle((x, y + 8, x + image_w, y + 8 + image_h), outline=border, width=1, fill=bg)
            canvas.paste(image, (x, y + 8))
            label = sample_cfg["labels"][key]
            label_box = (x, y + 8 + image_h, x + image_w, y + 8 + image_h + label_h)
            fill = accent if key == "ours" else muted
            _draw_centered(draw, label, label_box, tag_font, fill)
            x += image_w + gutter

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_prefix.with_suffix(".png"))
    canvas.save(output_prefix.with_suffix(".pdf"), "PDF", resolution=300.0)
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default="data/processed/clp4k_v6_union_en/manifest.jsonl")
    parser.add_argument(
        "--output-prefix",
        default="reports/final_results/figures/figure4_main_visual_comparison",
    )
    args = parser.parse_args()
    build_figure(Path(args.manifest_path), Path(args.output_prefix))


if __name__ == "__main__":
    main()
