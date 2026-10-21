from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/processed/clp4k_v6_union_en"
OUTPUT_ROOT = ROOT / "reports/final_results/figures"
PREDICTION_ROOT = ROOT / "outputs/paper_suite_v6/paper_ours_817/predictions_multiseed"

UNIFIED_PROMPT = (
    "Chinese landscape painting; pine trees and waterside pavilion; "
    "misty distant mountains; ink wash; ample poetic blank space."
)


@dataclass(frozen=True)
class CaseSpec:
    tag: str
    title: str
    structure_id: str
    output_file: str
    blue_label: str
    orange_label: str
    blue_box: tuple[float, float, float, float]
    orange_box: tuple[float, float, float, float]


CASES = [
    CaseSpec(
        tag="A",
        title="Large upper blank space",
        structure_id="clp4k_v6_union_en_000685",
        output_file="clp4k_v6_union_en_000685__seed42.png",
        blue_label="blank space",
        orange_label="low composition",
        blue_box=(0.06, 0.04, 0.82, 0.48),
        orange_box=(0.18, 0.48, 0.92, 0.88),
    ),
    CaseSpec(
        tag="B",
        title="Dominant central peak",
        structure_id="clp4k_v6_union_en_000673",
        output_file="clp4k_v6_union_en_000673__seed42.png",
        blue_label="spatial hierarchy",
        orange_label="main peak",
        blue_box=(0.10, 0.08, 0.86, 0.88),
        orange_box=(0.36, 0.08, 0.74, 0.82),
    ),
    CaseSpec(
        tag="C",
        title="Left-heavy mountain mass",
        structure_id="clp4k_v6_union_en_000740",
        output_file="clp4k_v6_union_en_000740__seed42.png",
        blue_label="open right",
        orange_label="left mass",
        blue_box=(0.42, 0.18, 0.92, 0.82),
        orange_box=(0.06, 0.16, 0.48, 0.84),
    ),
    CaseSpec(
        tag="D",
        title="Lower-right foreground",
        structure_id="clp4k_v6_union_en_000001",
        output_file="clp4k_v6_union_en_000001__seed42.png",
        blue_label="depth layers",
        orange_label="foreground",
        blue_box=(0.08, 0.10, 0.74, 0.58),
        orange_box=(0.58, 0.46, 0.92, 0.88),
    ),
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _structure_preview(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    lineart = image.split()[0].convert("L")
    lineart = ImageOps.autocontrast(lineart)
    preview = ImageOps.contain(lineart.convert("RGB"), size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(preview, ((size[0] - preview.width) // 2, (size[1] - preview.height) // 2))
    return canvas


def _image_preview(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    preview = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#FBFAF7")
    canvas.paste(preview, ((size[0] - preview.width) // 2, (size[1] - preview.height) // 2))
    return canvas


def _draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    color: str,
    width: int = 2,
    dash: int = 8,
    gap: int = 5,
) -> None:
    x0, y0, x1, y1 = box
    for x in range(x0, x1, dash + gap):
        draw.line((x, y0, min(x + dash, x1), y0), fill=color, width=width)
        draw.line((x, y1, min(x + dash, x1), y1), fill=color, width=width)
    for y in range(y0, y1, dash + gap):
        draw.line((x0, y, x0, min(y + dash, y1)), fill=color, width=width)
        draw.line((x1, y, x1, min(y + dash, y1)), fill=color, width=width)


def _scaled_box(origin: tuple[int, int], size: tuple[int, int], box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x, y = origin
    w, h = size
    return (
        int(x + box[0] * w),
        int(y + box[1] * h),
        int(x + box[2] * w),
        int(y + box[3] * h),
    )


def _label_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), label, font=font)
    pad_x, pad_y = 5, 3
    draw.rounded_rectangle(
        (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y),
        radius=4,
        fill="#FFFFFF",
        outline=fill,
        width=1,
    )
    draw.text((x, y), label, font=font, fill=fill)


def _annotate_panel(
    canvas: Image.Image,
    origin: tuple[int, int],
    size: tuple[int, int],
    case: CaseSpec,
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(canvas)
    blue = "#3A8DCC"
    orange = "#C66A2E"
    blue_box = _scaled_box(origin, size, case.blue_box)
    orange_box = _scaled_box(origin, size, case.orange_box)
    _draw_dashed_rectangle(draw, blue_box, blue)
    _draw_dashed_rectangle(draw, orange_box, orange)
    _label_box(draw, (blue_box[0] + 8, blue_box[1] + 8), case.blue_label, font, blue)
    _label_box(draw, (orange_box[0] + 8, orange_box[1] + 8), case.orange_label, font, orange)


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) // 2
    y = box[1] + (box[3] - box[1] - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, font=font, fill=fill)


def build_figure(output_prefix: Path = OUTPUT_ROOT / "figure5_structure_controllability") -> None:
    margin = 34
    gutter = 20
    header_h = 112
    col_header_h = 36
    row_h = 260
    structure_w = 410
    output_w = 590
    panel_inner = (350, 190)
    output_inner = (350, 190)
    total_w = margin * 2 + structure_w + gutter + output_w
    total_h = margin * 2 + header_h + col_header_h + row_h * len(CASES)

    bg = "#FFFFFF"
    paper = "#FBFAF7"
    border = "#D9D4CB"
    text = "#202833"
    muted = "#667085"
    accent = "#9A241F"

    canvas = Image.new("RGB", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)
    title_font = _font(23, bold=True)
    prompt_font = _font(15)
    header_font = _font(16, bold=True)
    case_font = _font(15, bold=True)
    small_font = _font(11)

    draw.text((margin, margin), "Figure 5. Structure Controllability Visualization", font=title_font, fill=text)
    prompt_box = (margin, margin + 44, total_w - margin, margin + 84)
    draw.rounded_rectangle(prompt_box, radius=8, fill="#F7F8FA", outline=border, width=1)
    draw.text(
        (prompt_box[0] + 14, prompt_box[1] + 11),
        f"Unified prompt: {UNIFIED_PROMPT}",
        font=prompt_font,
        fill=text,
    )

    table_y = margin + header_h
    _draw_centered(draw, "Input structure", (margin, table_y, margin + structure_w, table_y + col_header_h), header_font, text)
    _draw_centered(
        draw,
        "Generated output (Ours)",
        (margin + structure_w + gutter, table_y, total_w - margin, table_y + col_header_h),
        header_font,
        text,
    )

    start_y = table_y + col_header_h
    for idx, case in enumerate(CASES):
        row_y = start_y + idx * row_h
        left_x = margin
        right_x = margin + structure_w + gutter

        for x, width in [(left_x, structure_w), (right_x, output_w)]:
            draw.rounded_rectangle((x, row_y + 8, x + width, row_y + row_h - 8), radius=7, fill=paper, outline=border, width=1)

        case_text = f"Case {case.tag}: {case.title}"
        draw.text((left_x + 14, row_y + 18), case_text, font=case_font, fill=text)
        draw.text((left_x + 14, row_y + 40), case.structure_id.replace("clp4k_v6_union_en_", "test #"), font=small_font, fill=muted)

        structure_path = DATA_ROOT / "structure_maps" / f"{case.structure_id}.png"
        output_path = PREDICTION_ROOT / case.output_file
        if not structure_path.exists():
            raise FileNotFoundError(structure_path)
        if not output_path.exists():
            raise FileNotFoundError(output_path)

        structure = _structure_preview(structure_path, panel_inner)
        output = _image_preview(output_path, output_inner)

        structure_origin = (left_x + (structure_w - panel_inner[0]) // 2, row_y + 56)
        output_origin = (right_x + (output_w - output_inner[0]) // 2, row_y + 44)
        canvas.paste(structure, structure_origin)
        canvas.paste(output, output_origin)
        draw.rectangle(
            (
                structure_origin[0],
                structure_origin[1],
                structure_origin[0] + panel_inner[0],
                structure_origin[1] + panel_inner[1],
            ),
            outline=border,
            width=1,
        )
        draw.rectangle(
            (
                output_origin[0],
                output_origin[1],
                output_origin[0] + output_inner[0],
                output_origin[1] + output_inner[1],
            ),
            outline=border,
            width=1,
        )

        _annotate_panel(canvas, structure_origin, panel_inner, case, small_font)
        _annotate_panel(canvas, output_origin, output_inner, case, small_font)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    canvas.save(png_path, dpi=(300, 300))
    canvas.save(pdf_path, "PDF", resolution=300.0)
    print(f"Wrote {png_path.relative_to(ROOT)}")
    print(f"Wrote {pdf_path.relative_to(ROOT)}")


if __name__ == "__main__":
    build_figure()
