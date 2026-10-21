from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METHOD_LABELS = {
    "paper_controlnet_817": "ControlNet",
    "paper_ip_adapter_only_817": "IP-Adapter",
    "paper_lora_only_817": "SDXL+LoRA",
    "paper_ours_817": "Ours",
}

METHOD_ORDER = [
    "paper_lora_only_817",
    "paper_controlnet_817",
    "paper_ip_adapter_only_817",
    "paper_ours_817",
]

COLORS = {
    "SDXL+LoRA": "#6B7280",
    "ControlNet": "#2F6B5F",
    "IP-Adapter": "#B56B2D",
    "Ours": "#B5332E",
}


def _setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_main(results_root: Path) -> pd.DataFrame:
    df = pd.read_csv(results_root / "main_results_mean_std.csv")
    df = df[df["Method"].isin(METHOD_ORDER)].copy()
    df["label"] = df["Method"].map(METHOD_LABELS)
    df["order"] = df["Method"].map({name: index for index, name in enumerate(METHOD_ORDER)})
    return df.sort_values("order")


def plot_fidelity(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    labels = df["label"].tolist()
    x = np.arange(len(labels))
    colors = [COLORS[label] for label in labels]

    axes[0].bar(x, df["fid_mean"], yerr=df["fid_std"], color=colors, capsize=3)
    axes[0].set_title("Distribution fidelity")
    axes[0].set_ylabel("FID (lower is better)")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")

    axes[1].bar(x, df["kid_mean"], yerr=df["kid_std"], color=colors, capsize=3)
    axes[1].set_title("Kernel distance")
    axes[1].set_ylabel("KID (lower is better)")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")

    for ax in axes:
        ax.grid(axis="y", color="#E5E7EB")
        ax.grid(axis="x", visible=False)
    _save(fig, output_dir / "fig_main_fidelity")


def plot_quality_profile(df: pd.DataFrame, output_dir: Path) -> None:
    metrics = {
        "FID": ("fid_mean", "lower"),
        "KID": ("kid_mean", "lower"),
        "HPSv2": ("hpsv2_mean", "higher"),
        "LPIPS": ("lpips_diversity_mean", "higher"),
        "StyleAcc": ("style_accuracy_mean", "higher"),
        "BlankIoU": ("blank_space_iou_mean", "higher"),
    }
    values = []
    for _, row in df.iterrows():
        item = {"Method": row["label"]}
        for label, (column, direction) in metrics.items():
            series = df[column].astype(float)
            min_v = float(series.min())
            max_v = float(series.max())
            if math.isclose(max_v, min_v):
                score = 1.0
            elif direction == "lower":
                score = (max_v - float(row[column])) / (max_v - min_v)
            else:
                score = (float(row[column]) - min_v) / (max_v - min_v)
            item[label] = score
        values.append(item)
    heat = pd.DataFrame(values).set_index("Method")

    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    sns.heatmap(
        heat,
        ax=ax,
        cmap=sns.light_palette("#B5332E", as_cmap=True),
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Normalized score"},
        annot=True,
        fmt=".2f",
    )
    ax.set_title("Metric profile after direction-aware normalization")
    ax.set_xlabel("")
    ax.set_ylabel("")
    _save(fig, output_dir / "fig_metric_profile")


def plot_improvements(df: pd.DataFrame, output_dir: Path) -> None:
    rows = df.set_index("Method")
    ours = rows.loc["paper_ours_817"]
    lora = rows.loc["paper_lora_only_817"]
    improvements = {
        "FID reduction": (lora["fid_mean"] - ours["fid_mean"]) / lora["fid_mean"] * 100,
        "KID reduction": (lora["kid_mean"] - ours["kid_mean"]) / lora["kid_mean"] * 100,
        "LPIPS gain": (ours["lpips_diversity_mean"] - lora["lpips_diversity_mean"])
        / lora["lpips_diversity_mean"]
        * 100,
        "StyleAcc gain": (ours["style_accuracy_mean"] - lora["style_accuracy_mean"])
        / lora["style_accuracy_mean"]
        * 100,
    }
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    labels = list(improvements)
    vals = list(improvements.values())
    bars = ax.barh(labels, vals, color="#B5332E")
    ax.set_xlabel("Relative improvement over SDXL+LoRA (%)")
    ax.set_title("Main gains of the proposed method")
    ax.grid(axis="x", color="#E5E7EB")
    ax.grid(axis="y", visible=False)
    for bar, value in zip(bars, vals):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
    _save(fig, output_dir / "fig_ours_vs_lora_gains")


def plot_significance(results_root: Path, output_dir: Path) -> None:
    sig = pd.read_csv(results_root / "significance_ours_vs_lora.csv")
    sig["neg_log10_p"] = sig["wilcoxon_p"].apply(lambda p: -math.log10(max(float(p), 1e-300)))
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    labels = [
        "CLIP",
        "Pick",
        "HPSv2",
        "Style",
        "Edge",
        "Blank IoU",
        "Blank SSIM",
    ]
    x = np.arange(len(sig))
    ax.bar(x, sig["neg_log10_p"], color="#2F6B5F")
    ax.axhline(-math.log10(0.05), color="#B5332E", linestyle="--", linewidth=1.0, label="p=0.05")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("-log10(Wilcoxon p)")
    ax.set_title("Sample-level significance: Ours vs SDXL+LoRA")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E5E7EB")
    ax.grid(axis="x", visible=False)
    _save(fig, output_dir / "fig_significance_ours_vs_lora")


def plot_tradeoff(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for _, row in df.iterrows():
        label = row["label"]
        ax.scatter(
            row["fid_mean"],
            row["style_accuracy_mean"],
            s=120 + 360 * row["lpips_diversity_mean"],
            color=COLORS[label],
            edgecolor="white",
            linewidth=1,
            alpha=0.95,
        )
        ax.text(row["fid_mean"] + 3, row["style_accuracy_mean"], label, va="center", fontsize=8)
    ax.invert_xaxis()
    ax.set_xlabel("FID (lower is better; axis inverted)")
    ax.set_ylabel("Style accuracy")
    ax.set_title("Fidelity-style trade-off")
    ax.grid(color="#E5E7EB")
    _save(fig, output_dir / "fig_fidelity_style_tradeoff")


def write_tables(df: pd.DataFrame, results_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    main = pd.DataFrame(
        {
            "Method": df["label"],
            "Runs": df["NumRuns"],
            "FID": [f"{m:.2f} +/- {s:.2f}" for m, s in zip(df["fid_mean"], df["fid_std"])],
            "KID": [f"{m:.4f} +/- {s:.4f}" for m, s in zip(df["kid_mean"], df["kid_std"])],
            "PickScore": [
                f"{m:.2f} +/- {s:.2f}" for m, s in zip(df["pickscore_mean"], df["pickscore_std"])
            ],
            "HPSv2": [f"{m:.4f} +/- {s:.4f}" for m, s in zip(df["hpsv2_mean"], df["hpsv2_std"])],
            "LPIPS": [
                f"{m:.4f} +/- {s:.4f}"
                for m, s in zip(df["lpips_diversity_mean"], df["lpips_diversity_std"])
            ],
            "StyleAcc": [
                f"{m:.4f} +/- {s:.4f}"
                for m, s in zip(df["style_accuracy_mean"], df["style_accuracy_std"])
            ],
        }
    )
    main.to_csv(output_dir / "table_main_results_clean.csv", index=False)
    (output_dir / "table_main_results_clean.tex").write_text(
        main.to_latex(index=False, escape=False, column_format="lcllllll"),
        encoding="utf-8",
    )

    sig = pd.read_csv(results_root / "significance_ours_vs_lora.csv")
    sig_table = sig[["metric", "n", "left_mean", "right_mean", "paired_t_p", "wilcoxon_p"]].copy()
    sig_table.columns = ["Metric", "N", "Ours", "SDXL+LoRA", "Paired t p", "Wilcoxon p"]
    sig_table.to_csv(output_dir / "table_significance_ours_vs_lora.csv", index=False)
    (output_dir / "table_significance_ours_vs_lora.tex").write_text(
        sig_table.to_latex(index=False, escape=False),
        encoding="utf-8",
    )


def main(results_root: str, output_root: str) -> None:
    _setup_style()
    results_root_path = Path(results_root)
    output_root_path = Path(output_root)
    figure_dir = output_root_path / "figures"
    table_dir = output_root_path / "tables"
    df = _load_main(results_root_path)

    plot_fidelity(df, figure_dir)
    plot_quality_profile(df, figure_dir)
    plot_improvements(df, figure_dir)
    plot_significance(results_root_path, figure_dir)
    plot_tradeoff(df, figure_dir)
    write_tables(df, results_root_path, table_dir)
    print(f"Wrote figures to {figure_dir}")
    print(f"Wrote tables to {table_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="reports/final_results")
    parser.add_argument("--output-root", default="reports/final_results")
    args = parser.parse_args()
    main(results_root=args.results_root, output_root=args.output_root)
