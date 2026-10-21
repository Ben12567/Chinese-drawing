from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "reports/final_results"
FIGURE_ROOT = RESULTS_ROOT / "figures"

METHOD_LABELS = {
    "paper_lora_only_817": "LoRA-only",
    "paper_controlnet_817": "ControlNet",
    "paper_ip_adapter_only_817": "IP-Adapter only",
    "paper_ours_817": "Ours",
}

METHOD_ORDER = [
    "paper_lora_only_817",
    "paper_controlnet_817",
    "paper_ip_adapter_only_817",
    "paper_ours_817",
]

PALETTE = {
    "LoRA-only": "#707B84",
    "ControlNet": "#3E7771",
    "IP-Adapter only": "#C7823A",
    "Ours": "#B5332E",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 450,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#B8B8B8",
            "axes.linewidth": 0.8,
            "grid.color": "#E8E2DA",
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_main() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_ROOT / "main_results_mean_std.csv")
    df = df[df["Method"].isin(METHOD_ORDER)].copy()
    df["method"] = pd.Categorical(df["Method"], categories=METHOD_ORDER, ordered=True)
    df["label"] = df["Method"].map(METHOD_LABELS)
    return df.sort_values("method")


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    png = FIGURE_ROOT / f"{stem}.png"
    pdf = FIGURE_ROOT / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png.relative_to(ROOT)}")
    print(f"Wrote {pdf.relative_to(ROOT)}")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def bar_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    title: str,
    ylabel: str,
    lower_is_better: bool,
) -> None:
    labels = df["label"].tolist()
    x = np.arange(len(labels))
    colors = [PALETTE[label] for label in labels]
    y = df[mean_col].astype(float).to_numpy()
    yerr = df[std_col].astype(float).to_numpy()
    bars = ax.bar(x, y, color=colors, width=0.68, edgecolor="#2A2A2A", linewidth=0.45)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#202020", elinewidth=0.9, capsize=3, capthick=0.9)
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, labels, rotation=16, ha="right")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    best_index = int(np.argmin(y) if lower_is_better else np.argmax(y))
    for idx, (bar, value) in enumerate(zip(bars, y)):
        color = "#B5332E" if idx == best_index else "#4B5563"
        weight = "bold" if idx == best_index else "normal"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(y) * 0.025,
            f"{value:.2f}" if value > 1 else f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
            fontweight=weight,
        )


def normalized_metric_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metrics: list[tuple[str, str, str]],
    title: str,
) -> None:
    labels = df["label"].tolist()
    x = np.arange(len(metrics))
    width = 0.18

    for method_index, (_, row) in enumerate(df.iterrows()):
        scores = []
        for _, column, direction in metrics:
            values = df[column].astype(float)
            min_v = float(values.min())
            max_v = float(values.max())
            if math.isclose(max_v, min_v):
                score = 1.0
            elif direction == "lower":
                score = (max_v - float(row[column])) / (max_v - min_v)
            else:
                score = (float(row[column]) - min_v) / (max_v - min_v)
            scores.append(score)
        offset = (method_index - (len(labels) - 1) / 2) * width
        ax.bar(
            x + offset,
            scores,
            width=width,
            label=row["label"],
            color=PALETTE[row["label"]],
            edgecolor="#2A2A2A",
            linewidth=0.35,
        )

    ax.set_title(title, pad=8)
    ax.set_ylabel("Normalized score")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x, [name for name, _, _ in metrics])
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def make_quantitative_dashboard(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.0))
    fig.patch.set_facecolor("white")

    bar_metric(
        axes[0, 0],
        df,
        "fid_mean",
        "fid_std",
        "Target-domain fidelity",
        "FID (lower)",
        lower_is_better=True,
    )
    add_panel_label(axes[0, 0], "a")

    bar_metric(
        axes[0, 1],
        df,
        "kid_mean",
        "kid_std",
        "Distribution distance",
        "KID (lower)",
        lower_is_better=True,
    )
    add_panel_label(axes[0, 1], "b")

    normalized_metric_panel(
        axes[1, 0],
        df,
        [
            ("CLIP", "clipscore_mean", "higher"),
            ("Pick", "pickscore_mean", "higher"),
            ("HPSv2", "hpsv2_mean", "higher"),
            ("LPIPS", "lpips_diversity_mean", "higher"),
        ],
        "General alignment / preference / diversity",
    )
    add_panel_label(axes[1, 0], "c")

    normalized_metric_panel(
        axes[1, 1],
        df,
        [
            ("Style", "style_accuracy_mean", "higher"),
            ("Edge", "edge_consistency_mean", "higher"),
            ("BlankIoU", "blank_space_iou_mean", "higher"),
            ("BlankSSIM", "blank_space_ssim_mean", "higher"),
        ],
        "Domain-specific style and structure",
    )
    add_panel_label(axes[1, 1], "d")

    handles = [Patch(facecolor=PALETTE[label], edgecolor="#2A2A2A", label=label) for label in df["label"]]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Quantitative evaluation on the Chinese landscape painting benchmark", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "figure6_quantitative_results_dashboard")


def make_evidence_map(df: pd.DataFrame) -> None:
    sig = pd.read_csv(RESULTS_ROOT / "significance_ours_vs_lora.csv")
    fig = plt.figure(figsize=(9.3, 3.8), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.32)
    ax_trade = fig.add_subplot(gs[0, 0])
    ax_sig = fig.add_subplot(gs[0, 1])

    for _, row in df.iterrows():
        label = row["label"]
        ax_trade.scatter(
            row["fid_mean"],
            row["style_accuracy_mean"],
            s=120 + 520 * row["lpips_diversity_mean"],
            color=PALETTE[label],
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
        )
        ax_trade.text(
            row["fid_mean"] + 4,
            row["style_accuracy_mean"] + 0.004,
            label,
            fontsize=8,
            color="#222222",
        )
    ax_trade.invert_xaxis()
    ax_trade.set_xlabel("FID (lower; axis inverted)")
    ax_trade.set_ylabel("Style accuracy")
    ax_trade.set_title("Fidelity-style trade-off", pad=8)
    ax_trade.grid(True)
    add_panel_label(ax_trade, "a")

    metric_labels = {
        "clipscore": "CLIP",
        "pickscore": "Pick",
        "hpsv2": "HPSv2",
        "style_correct": "Style",
        "edge_consistency": "Edge",
        "blank_space_iou": "Blank IoU",
        "blank_space_ssim": "Blank SSIM",
    }
    sig["metric_label"] = sig["metric"].map(metric_labels)
    sig["neg_log10_p"] = sig["wilcoxon_p"].apply(lambda value: -math.log10(max(float(value), 1e-300)))
    sig["delta"] = sig["left_mean"] - sig["right_mean"]
    colors = ["#B5332E" if delta > 0 else "#707B84" for delta in sig["delta"]]

    y = np.arange(len(sig))
    bars = ax_sig.barh(y, sig["neg_log10_p"], color=colors, edgecolor="#2A2A2A", linewidth=0.35)
    ax_sig.axvline(-math.log10(0.05), color="#1F2933", linestyle="--", linewidth=1.0)
    ax_sig.text(
        -math.log10(0.05) + 0.1,
        len(sig) - 0.35,
        "p = 0.05",
        fontsize=7,
        color="#1F2933",
        va="center",
    )
    ax_sig.set_yticks(y, sig["metric_label"])
    ax_sig.set_xlabel(r"$-\log_{10}$ Wilcoxon p")
    ax_sig.set_title("Paired significance: Ours vs LoRA-only", pad=8)
    ax_sig.grid(axis="x")
    ax_sig.grid(axis="y", visible=False)
    ax_sig.invert_yaxis()
    add_panel_label(ax_sig, "b")

    for bar, p_value in zip(bars, sig["wilcoxon_p"]):
        if p_value < 0.001:
            marker = "***"
        elif p_value < 0.01:
            marker = "**"
        elif p_value < 0.05:
            marker = "*"
        else:
            marker = "n.s."
        ax_sig.text(
            bar.get_width() + 0.12,
            bar.get_y() + bar.get_height() / 2,
            marker,
            va="center",
            fontsize=8,
            color="#222222",
        )

    legend_handles = [
        Patch(facecolor="#B5332E", edgecolor="#2A2A2A", label="Ours higher"),
        Patch(facecolor="#707B84", edgecolor="#2A2A2A", label="LoRA-only higher"),
    ]
    ax_sig.legend(handles=legend_handles, loc="lower right", frameon=False)
    fig.tight_layout()
    save_figure(fig, "figure7_tradeoff_and_significance")


def main() -> None:
    setup_style()
    df = load_main()
    make_quantitative_dashboard(df)
    make_evidence_map(df)


if __name__ == "__main__":
    main()
