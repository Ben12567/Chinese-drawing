from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/final_results"
FIGURE_ROOT = REPORT_ROOT / "figures"
TABLE_ROOT = ROOT / "paper/tables"
DATA_ROOT = ROOT / "data/processed/clp4k_v6_union_en"

METHOD_LABELS = {
    "paper_controlnet_817": "ControlNet",
    "paper_ip_adapter_only_817": "IP-Adapter only",
    "paper_lora_only_817": "LoRA-only",
    "paper_ours_817": "Ours",
}

METHOD_ORDER = [
    "paper_controlnet_817",
    "paper_ip_adapter_only_817",
    "paper_lora_only_817",
    "paper_ours_817",
]

COLORS = {
    "ControlNet": "#3E7771",
    "IP-Adapter only": "#C7823A",
    "LoRA-only": "#707B84",
    "Ours": "#B5332E",
    "blue": "#2F6E8E",
    "green": "#496B3D",
    "red": "#A33A31",
    "paper": "#FBFAF7",
}


def _setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#E8E2DA",
        }
    )


def _save_fig(fig: plt.Figure, stem: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ROOT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_ROOT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote reports/final_results/figures/{stem}.png")
    print(f"Wrote reports/final_results/figures/{stem}.pdf")


def _load_summary() -> dict:
    return json.loads((ROOT / "outputs/dataset_report/clp4k_v6_union_en_summary.json").read_text(encoding="utf-8"))


def _load_manifest() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with (DATA_ROOT / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows[item["sample_id"]] = item
    return rows


def _structure_channel(path: Path, index: int) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    channel = Image.fromarray(np.asarray(image)[..., index]).convert("L")
    return ImageOps.autocontrast(channel)


def _fit_image(image: Image.Image, size: tuple[int, int], fill: str = "white") -> Image.Image:
    image = image.convert("RGB")
    image = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, fill)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def make_figure3_dataset_structure() -> None:
    summary = _load_summary()
    manifest = _load_manifest()
    sample_id = "clp4k_v6_union_en_000001"
    sample = manifest[sample_id]

    fig = plt.figure(figsize=(10.6, 5.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.25], height_ratios=[1.0, 1.15], wspace=0.36, hspace=0.44)

    ax_source = fig.add_subplot(gs[0, 0])
    source_items = sorted(summary["source_counts"].items(), key=lambda item: item[1], reverse=True)
    ax_source.bar([item[0].upper() for item in source_items], [item[1] for item in source_items], color="#496B3D")
    ax_source.set_title("Source distribution")
    ax_source.set_ylabel("Images")
    ax_source.grid(axis="y")

    ax_split = fig.add_subplot(gs[0, 1])
    split_order = ["train", "val", "test"]
    ax_split.bar(split_order, [summary["split_counts"][key] for key in split_order], color="#2F6E8E")
    ax_split.set_title("Data split")
    ax_split.set_ylabel("Images")
    ax_split.grid(axis="y")

    ax_style = fig.add_subplot(gs[0, 2])
    style_items = sorted(summary["style_counts"].items(), key=lambda item: item[1], reverse=True)
    style_name_map = {
        "blue_green_landscape": "blue-green",
        "ink_wash_freehand": "ink-wash",
        "light_reddish_landscape": "light-red",
        "mi_style_cloud_mountains": "Mi-style",
        "dong_yuan_style": "Dong Yuan",
    }
    style_labels = [style_name_map.get(item[0], item[0].replace("_", " ")) for item in style_items]
    ax_style.barh(style_labels, [item[1] for item in style_items], color="#C7823A")
    ax_style.set_title("Style-label distribution")
    ax_style.set_xlabel("Images")
    ax_style.grid(axis="x")
    ax_style.invert_yaxis()

    ax_sample = fig.add_subplot(gs[1, 0])
    ax_sample.imshow(_fit_image(Image.open(DATA_ROOT / sample["image_path"]), (360, 230), "#FBFAF7"))
    ax_sample.set_title("Representative painting")
    ax_sample.axis("off")

    ax_channels = fig.add_subplot(gs[1, 1:])
    channel_names = ["lineart", "quantized depth", "blank mask", "saliency mask"]
    channel_imgs = [_fit_image(_structure_channel(DATA_ROOT / sample["structure_map_path"], idx), (190, 190)) for idx in range(4)]
    canvas = Image.new("RGB", (820, 235), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, image in enumerate(channel_imgs):
        x = idx * 200
        canvas.paste(image, (x + 10, 0))
        draw.rectangle((x + 10, 0, x + 200, 190), outline=(210, 210, 210), width=1)
        draw.text((x + 18, 200), channel_names[idx], fill=(45, 45, 45))
    ax_channels.imshow(canvas)
    ax_channels.set_title("Four-channel hierarchical structure")
    ax_channels.axis("off")

    fig.suptitle("Dataset and structural annotation overview", y=1.02, fontsize=12, fontweight="bold")
    _save_fig(fig, "figure3_dataset_and_structure_overview")


def make_figure8_failures() -> None:
    cases = [
        (
            "Text-heavy source",
            DATA_ROOT / "structure_maps/clp4k_v6_union_en_000038.png",
            ROOT / "outputs/paper_suite_v6/paper_ours_817/predictions_multiseed/clp4k_v6_union_en_000038__seed42.png",
            "Calligraphic regions may dominate visual content.",
        ),
        (
            "Weak layout",
            DATA_ROOT / "structure_maps/clp4k_v6_union_en_000001.png",
            ROOT / "outputs/paper_suite_v6/paper_lora_only_817/predictions_multiseed/clp4k_v6_union_en_000001__seed42.png",
            "LoRA-only learns tone but lacks explicit composition control.",
        ),
        (
            "Rigid contour",
            DATA_ROOT / "structure_maps/clp4k_v6_union_en_000110.png",
            ROOT / "outputs/paper_suite_strong/paper_controlnet_817_seed42/predictions_multiseed/clp4k_v6_union_en_000110__seed42.png",
            "Hard structural adherence may reduce brush-and-ink naturalness.",
        ),
        (
            "Layout drift",
            DATA_ROOT / "structure_maps/clp4k_v6_union_en_000061.png",
            ROOT / "outputs/paper_suite_strong/paper_ip_adapter_only_817_seed42/predictions_multiseed/clp4k_v6_union_en_000061__seed42.png",
            "Reference-style control can weaken spatial correspondence.",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.3), facecolor="white")
    for ax, (issue, structure_path, output_path, note) in zip(axes.ravel(), cases):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes, fill=False, ec="#D7D2CA", lw=0.9))
        ax.text(0.05, 0.90, issue, color=COLORS["red"], fontsize=10, fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.79, note, color="#30343B", fontsize=7.2, transform=ax.transAxes, wrap=True)
        structure = _fit_image(_structure_channel(structure_path, 0), (180, 150))
        output = _fit_image(Image.open(output_path), (180, 150), "#FBFAF7")
        left = ax.inset_axes([0.06, 0.10, 0.38, 0.48])
        right = ax.inset_axes([0.54, 0.10, 0.38, 0.48])
        left.imshow(structure)
        right.imshow(output)
        for subax in (left, right):
            subax.set_xticks([])
            subax.set_yticks([])
            for spine in subax.spines.values():
                spine.set_color("#D7D2CA")
                spine.set_linewidth(0.7)
        ax.text(0.06, 0.61, "Structure", fontsize=7, color="#666666", transform=ax.transAxes)
        ax.text(0.54, 0.61, "Generated", fontsize=7, color="#666666", transform=ax.transAxes)
    fig.suptitle("Representative limitations and failure cases", y=1.02, fontsize=12, fontweight="bold")
    _save_fig(fig, "figure8_failure_cases")


def _normalize(series: pd.Series, lower: bool = False) -> pd.Series:
    values = series.astype(float)
    lo = values.min()
    hi = values.max()
    if math.isclose(float(lo), float(hi)):
        return pd.Series([1.0] * len(values), index=series.index)
    if lower:
        return (hi - values) / (hi - lo)
    return (values - lo) / (hi - lo)


def make_figure9_ablation_generalization() -> None:
    ablation = pd.read_csv(ROOT / "outputs/paper_suite/ablation_summary.csv")
    bench = json.loads((REPORT_ROOT / "benchmark_summary_v6.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), facecolor="white", gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    labels = ["Full", "w/o structure", "w/o style"]
    x = np.arange(len(labels))
    fid_norm = _normalize(ablation["FID"], lower=True)
    kid_norm = _normalize(ablation["KID"], lower=True)
    lpips_norm = _normalize(ablation["LPIPS_Diversity"], lower=False)
    width = 0.24
    ax.bar(x - width, fid_norm, width, label="FID", color="#496B3D")
    ax.bar(x, kid_norm, width, label="KID", color="#2F6E8E")
    ax.bar(x + width, lpips_norm, width, label="LPIPS", color="#C7823A")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Normalized score")
    ax.set_title("Pilot component ablation")
    ax.text(0.01, -0.30, "Early protocol; use as appendix evidence.", transform=ax.transAxes, fontsize=7, color="#666666")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    methods = ["LoRA-only", "Ours"]
    tifa = [bench["paper_lora_only_817"]["tifa_subset"]["tifa_average"], bench["paper_ours_817"]["tifa_subset"]["tifa_average"]]
    t2i = [
        bench["paper_lora_only_817"]["t2i_compbench_non_spatial"]["score_avg"],
        bench["paper_ours_817"]["t2i_compbench_non_spatial"]["score_avg"],
    ]
    x = np.arange(len(methods))
    ax.bar(x - 0.18, tifa, 0.36, label="TIFA subset", color="#707B84")
    ax.bar(x + 0.18, t2i, 0.36, label="T2I-CompBench non-spatial", color="#B5332E")
    ax.set_xticks(x, methods)
    ax.set_title("General benchmark boundary")
    ax.set_ylabel("Score")
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=7)
    ax.text(0.01, -0.30, "Ours is domain-oriented, not universally stronger.", transform=ax.transAxes, fontsize=7, color="#666666")

    fig.suptitle("Ablation evidence and claim boundary", y=1.02, fontsize=12, fontweight="bold")
    _save_fig(fig, "figure9_ablation_and_generalization")


def _escape_tex(text: str) -> str:
    return (
        str(text)
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _tex_cell(text: str) -> str:
    text = str(text)
    if "\\" in text or "$" in text:
        return text
    return _escape_tex(text)


def _write_table(name: str, caption: str, label: str, columns: list[str], rows: list[list[str]], *, wide: bool = False) -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    env = "table*" if wide else "table"
    col_spec = "l" + "c" * (len(columns) - 1)
    lines = [
        f"\\begin{{{env}}}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{col}}}" for col in columns) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_tex_cell(cell) for cell in row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""])
    path = TABLE_ROOT / f"{name}.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}")


def write_tables() -> None:
    summary = _load_summary()
    main = pd.read_csv(REPORT_ROOT / "main_results_mean_std.csv")
    main = main[main["Method"].isin(METHOD_ORDER)].copy()
    main["Label"] = main["Method"].map(METHOD_LABELS)
    main["Order"] = main["Method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    main = main.sort_values("Order")

    _write_table(
        "table1_dataset_overview",
        r"\textbf{Dataset overview.} Summary statistics of the cleaned Chinese landscape painting benchmark.",
        "tab:dataset-overview",
        ["Item", "Value"],
        [
            ["Total images", str(summary["num_samples"])],
            ["Train / validation / test", f"{summary['split_counts']['train']} / {summary['split_counts']['val']} / {summary['split_counts']['test']}"],
            ["Sources", ", ".join(f"{k.upper()}={v}" for k, v in summary["source_counts"].items())],
            ["Style categories", str(summary["num_styles"])],
            ["Painter/author entries", str(summary["num_painters"])],
            ["Width range", f"{summary['min_width']}--{summary['max_width']} px"],
            ["Height range", f"{summary['min_height']}--{summary['max_height']} px"],
        ],
    )

    style_rows = []
    for style, count in sorted(summary["style_counts"].items(), key=lambda item: item[1], reverse=True):
        style_rows.append([style, str(count), f"{count / summary['num_samples'] * 100:.1f}"])
    _write_table(
        "table2_dataset_style_distribution",
        r"\textbf{Style-label distribution.} The benchmark is long-tailed, with rare historical-style categories retained for analysis.",
        "tab:dataset-style-distribution",
        ["Style label", "Count", "Percent"],
        style_rows,
    )

    _write_table(
        "table3_experimental_protocol",
        r"\textbf{Experimental protocol.} Main settings used for final quantitative evaluation.",
        "tab:experimental-protocol",
        ["Setting", "Value"],
        [
            ["Base model", "SDXL 1.0"],
            ["Training split", "653 images"],
            ["Test split", "83 images"],
            ["Training resolution", "768"],
            ["Inference resolution", "1024"],
            ["Maximum training steps", "500"],
            ["Training seeds", "42, 52, 62 for Ours and LoRA-only"],
            ["Inference seeds", "42, 52, 62"],
            ["Generated test images", "1992 in the main evaluation"],
        ],
    )

    _write_table(
        "table4_method_configuration",
        r"\textbf{Compared method configuration.} Condition branches enabled for each method.",
        "tab:method-configuration",
        ["Method", "LoRA", "Structure control", "Style reference", "Runs"],
        [
            ["ControlNet", "No", "lineart ControlNet", "No", "1"],
            ["IP-Adapter only", "No", "No", "IP-Adapter", "1"],
            ["LoRA-only", "Yes", "No", "No", "3"],
            ["Ours", "Yes", "hierarchical adapter", "IP-Adapter", "3"],
        ],
    )

    def fmt_mean(row: pd.Series, mean_col: str, std_col: str, decimals: int = 4) -> str:
        mean = float(row[mean_col])
        std = float(row[std_col])
        if int(row["NumRuns"]) > 1:
            return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"
        return f"{mean:.{decimals}f}"

    _write_table(
        "table5_main_quantitative_results",
        r"\textbf{Main quantitative results.} Fidelity, alignment, preference, and diversity comparison.",
        "tab:main-quantitative-results",
        ["Method", "FID$\\downarrow$", "KID$\\downarrow$", "CLIP$\\uparrow$", "Pick$\\uparrow$", "HPSv2$\\uparrow$", "LPIPS$\\uparrow$"],
        [
            [
                row["Label"],
                fmt_mean(row, "fid_mean", "fid_std", 2),
                fmt_mean(row, "kid_mean", "kid_std", 4),
                fmt_mean(row, "clipscore_mean", "clipscore_std", 4),
                fmt_mean(row, "pickscore_mean", "pickscore_std", 2),
                fmt_mean(row, "hpsv2_mean", "hpsv2_std", 4),
                fmt_mean(row, "lpips_diversity_mean", "lpips_diversity_std", 4),
            ]
            for _, row in main.iterrows()
        ],
        wide=True,
    )

    _write_table(
        "table6_style_structure_results",
        r"\textbf{Style and structure metrics.} Domain-specific evaluation of style labels, edge adherence, and blank-space organization.",
        "tab:style-structure-results",
        ["Method", "Style Acc.$\\uparrow$", "Edge Cons.$\\uparrow$", "Blank IoU$\\uparrow$", "Blank SSIM$\\uparrow$"],
        [
            [
                row["Label"],
                fmt_mean(row, "style_accuracy_mean", "style_accuracy_std", 4),
                fmt_mean(row, "edge_consistency_mean", "edge_consistency_std", 4),
                fmt_mean(row, "blank_space_iou_mean", "blank_space_iou_std", 4),
                fmt_mean(row, "blank_space_ssim_mean", "blank_space_ssim_std", 4),
            ]
            for _, row in main.iterrows()
        ],
    )

    sig = pd.read_csv(REPORT_ROOT / "significance_ours_vs_lora.csv")
    metric_labels = {
        "clipscore": "CLIPScore",
        "pickscore": "PickScore",
        "hpsv2": "HPSv2",
        "style_correct": "Style correct",
        "edge_consistency": "Edge consistency",
        "blank_space_iou": "Blank IoU",
        "blank_space_ssim": "Blank SSIM",
    }
    sig_rows = []
    for _, row in sig.iterrows():
        sig_rows.append([
            metric_labels.get(row["metric"], row["metric"]),
            str(int(row["n"])),
            f"{float(row['left_mean']):.4f}",
            f"{float(row['right_mean']):.4f}",
            f"{float(row['paired_t_p']):.2e}",
            f"{float(row['wilcoxon_p']):.2e}",
        ])
    _write_table(
        "table7_significance_ours_vs_lora",
        r"\textbf{Paired significance analysis.} Per-sample comparison between the proposed method and LoRA-only on the 83-image test split.",
        "tab:significance-ours-vs-lora",
        ["Metric", "N", "Ours", "LoRA-only", "Paired t-test", "Wilcoxon"],
        sig_rows,
        wide=True,
    )

    ablation = pd.read_csv(ROOT / "outputs/paper_suite/ablation_summary.csv")
    _write_table(
        "table8_pilot_ablation_and_general_benchmark",
        r"\textbf{Pilot component ablation and general benchmark boundary.} The ablation rows are from an early protocol and should be interpreted as appendix evidence; the benchmark rows show that the method is not universally stronger on open-domain compositional tests.",
        "tab:pilot-ablation-general-benchmark",
        ["Experiment", "FID", "KID", "CLIP/TIFA", "LPIPS/T2I"],
        [
            ["Full model (pilot)", f"{ablation.loc[0, 'FID']:.2f}", f"{ablation.loc[0, 'KID']:.4f}", f"{ablation.loc[0, 'CLIPScore']:.4f}", f"{ablation.loc[0, 'LPIPS_Diversity']:.4f}"],
            ["w/o structure (pilot)", f"{ablation.loc[1, 'FID']:.2f}", f"{ablation.loc[1, 'KID']:.4f}", f"{ablation.loc[1, 'CLIPScore']:.4f}", f"{ablation.loc[1, 'LPIPS_Diversity']:.4f}"],
            ["w/o style ref. (pilot)", f"{ablation.loc[2, 'FID']:.2f}", f"{ablation.loc[2, 'KID']:.4f}", f"{ablation.loc[2, 'CLIPScore']:.4f}", f"{ablation.loc[2, 'LPIPS_Diversity']:.4f}"],
            ["LoRA-only general bench.", "-", "-", "TIFA 0.7783", "T2I 0.3122"],
            ["Ours general bench.", "-", "-", "TIFA 0.7566", "T2I 0.3040"],
        ],
        wide=True,
    )


def write_experiment_extension_plan() -> None:
    path = ROOT / "docs/extended_experiment_plan.md"
    text = """# Extended Experiment Package

This package expands the paper result presentation without fabricating unrun experiments.

## Recommended Final Figure Set

1. Figure 1: Motivation overview, editable PowerPoint.
2. Figure 2: Method framework, still needs final editable diagram.
3. Figure 3: Dataset and structure overview, generated from real dataset statistics.
4. Figure 4: Main qualitative comparison, editable PowerPoint.
5. Figure 5: Structure controllability visualization, editable PowerPoint.
6. Figure 6: Quantitative results dashboard.
7. Figure 7: Fidelity-style trade-off and significance.
8. Figure 8: Representative failure cases and limitations.

Optional appendix figure:

- Figure 9: Pilot ablation and general benchmark boundary.

## Recommended Final Table Set

1. Dataset overview.
2. Style-label distribution.
3. Experimental protocol.
4. Compared method configuration.
5. Main quantitative results.
6. Style and structure metrics.
7. Paired significance analysis.
8. Pilot ablation and general benchmark boundary.

## Ablation Policy

The existing component ablation is usable only as a pilot/appendix result because it was produced under an earlier protocol. For a strong method paper, rerun the following under the final 817-image protocol:

- Full model.
- w/o hierarchical structure adapter.
- lineart only.
- lineart + quantized depth.
- w/o blank-space mask.
- w/o saliency mask.
- w/o style reference.
- short prompt vs structured prompt.
- LoRA U-Net only vs U-Net + text encoder.

Report all final ablations with the same test split, resolution, inference seeds, and metrics as the main table.
"""
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}")


def main() -> None:
    _setup()
    make_figure3_dataset_structure()
    make_figure8_failures()
    make_figure9_ablation_generalization()
    write_tables()
    write_experiment_extension_plan()


if __name__ == "__main__":
    main()
