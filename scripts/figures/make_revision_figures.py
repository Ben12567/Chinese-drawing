"""Figures 3-8 and supporting figures for the revision, built from outputs/revision.

Qualitative examples are drawn at random (numpy Generator with the seed given below) from the
test set; no example is hand-picked. Every generated image is annotated with its per-image
edge IoU (E) and blank-space IoU (B); no evaluative labels are used.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common import channel_images, save_tiff

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/clp_v7"
EVAL = ROOT / "outputs/revision/eval"
GEN = ROOT / "outputs/revision/gen"
TRAIN = ROOT / "outputs/revision/train"
OUT = ROOT / "outputs/revision/figures"
SAMPLING_SEED = 2026

MAIN = [
    ("SDXL", ["sdxl_base"]),
    ("LoRA-only", ["lora_only_s42", "lora_only_s52", "lora_only_s62"]),
    ("ControlNet", ["controlnet"]),
    ("T2I-Adapter", ["t2i_adapter"]),
    ("IP-Adapter-only", ["ip_adapter_only"]),
    ("LoRA+ControlNet", ["lora_controlnet_s42", "lora_controlnet_s52", "lora_controlnet_s62"]),
    ("LoRA+IP-Adapter", ["lora_ip_adapter_s42", "lora_ip_adapter_s52", "lora_ip_adapter_s62"]),
    ("Ours w/o style ref.", ["ours_no_style_s42", "ours_no_style_s52", "ours_no_style_s62"]),
    ("Ours", ["ours_s42", "ours_s52", "ours_s62"]),
]


def manifest() -> dict:
    rows = [json.loads(l) for l in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["sample_id"]: r for r in rows}


def summary(run: str) -> dict:
    return json.loads((EVAL / run / "summary.json").read_text(encoding="utf-8"))


def per_image(run: str) -> pd.DataFrame:
    return pd.read_csv(EVAL / run / "per_image_metrics.csv")


def show(ax, path, title=None, note=None, gray=False):
    ax.imshow(Image.open(path).convert("L" if gray else "RGB"), cmap="gray" if gray else None)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    if title:
        ax.set_title(title, fontsize=7, pad=2)
    if note:
        ax.set_xlabel(note, fontsize=6, labelpad=1.5)


def fig3_fidelity(out: Path) -> None:
    real = summary("real_test")["plain"]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9))
    for ax, key, label in zip(axes, ["kid", "cmmd", "fid"], ["KID ↓", "CMMD ↓", "FID ↓"]):
        names, means, lo, hi, sds = [], [], [], [], []
        for name, runs in MAIN:
            vals = [summary(r)["plain"] for r in runs]
            names.append(name)
            means.append(np.mean([v[key] for v in vals]))
            lo.append(vals[0][f"{key}_ci"][0])
            hi.append(vals[0][f"{key}_ci"][1])
            sds.append(np.std([v[key] for v in vals], ddof=1) if len(vals) > 1 else np.nan)
        y = np.arange(len(names))[::-1]
        colors = ["#c44e52" if n == "Ours" else "#4c72b0" for n in names]
        ax.barh(y, means, color=colors, height=0.65)
        err_lo = np.array(means) - np.array(lo)
        err_hi = np.array(hi) - np.array(means)
        ax.errorbar(means, y, xerr=[np.clip(err_lo, 0, None), np.clip(err_hi, 0, None)], fmt="none", ecolor="black", lw=0.7, capsize=1.5)
        for yy, m, sd in zip(y, means, sds):
            if not np.isnan(sd):
                ax.plot([m - sd, m + sd], [yy + 0.38, yy + 0.38], color="#ff9f1c", lw=1.2)
        ax.axvline(real[key], color="gray", ls="--", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(names if key == "kid" else [], fontsize=6.5)
        ax.set_title(label, fontsize=8)
        ax.tick_params(axis="x", labelsize=6)
    fig.tight_layout()
    save_tiff(fig, out)


def pick_random_tests(n: int, seed: int = SAMPLING_SEED) -> list[str]:
    m = manifest()
    test = sorted(s for s, r in m.items() if r["split"] == "test")
    rng = np.random.default_rng(seed)
    return list(rng.choice(test, n, replace=False))


def annotate(df: pd.DataFrame, path: str) -> str:
    row = df[df.path.str.replace("\\", "/").str.endswith(Path(path).as_posix().split("/")[-1])]
    if row.empty:
        return ""
    r = row.iloc[0]
    return f"E {r.edge_iou:.3f}  B {r.blank_iou:.3f}"


def fig4_qualitative(out: Path, runs: list[tuple[str, str]], n: int = 4, seed_tag: str = "42") -> None:
    m = manifest()
    ids = pick_random_tests(n)
    cols = 2 + len(runs)
    fig, axes = plt.subplots(n, cols, figsize=(7.5, 7.5 * n / cols * 1.12))
    dfs = {name: per_image(run) for name, run in runs}
    for i, sid in enumerate(ids):
        r = m[sid]
        show(axes[i, 0], DATA / r["image_path"], "Real test painting" if i == 0 else None)
        axes[i, 0].set_ylabel(textwrap.fill(r["extra"]["prompt_fields"]["subject"], 22), fontsize=5.5)
        show(axes[i, 1], channel_images(DATA / r["structure_map_path"])[0] if False else DATA / r["structure_map_path"], "Line-art channel" if i == 0 else None, gray=True)
        axes[i, 1].images[0].set_data(channel_images(DATA / r["structure_map_path"])[0])
        for j, (name, run) in enumerate(runs):
            path = GEN / run / f"{sid}__seed{seed_tag}.png"
            show(axes[i, 2 + j], path, name if i == 0 else None, annotate(dfs[name], str(path)))
    fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.4)
    save_tiff(fig, out)


def fig5_structure(out: Path, n: int = 4) -> None:
    df = pd.read_csv(ROOT / "outputs/revision/controllability/structure_controllability.csv")
    m = manifest()
    rng = np.random.default_rng(SAMPLING_SEED)
    prompt_id = df.prompt_sample.iloc[0]
    rows = df[df.prompt_sample == prompt_id].iloc[:n]
    fig, axes = plt.subplots(n, 5, figsize=(7.5, 7.5 * n / 5 * 1.05))
    names = ["Line art", "Layer map", "Blank-space mask", "Salience mask", "Output (Ours)"]
    for i, (_, row) in enumerate(rows.iterrows()):
        chans = channel_images(DATA / m[row.structure_sample]["structure_map_path"])
        for j in range(4):
            axes[i, j].imshow(chans[j], cmap="gray", vmin=0, vmax=255)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(names[j], fontsize=7)
        show(axes[i, 4], row.path, names[4] if i == 0 else None,
             f"E {row.edge_iou_used:.3f} (others {row.edge_iou_unused:.3f})\nB {row.blank_iou_used:.3f} (others {row.blank_iou_unused:.3f})")
    fig.tight_layout(pad=0.3)
    save_tiff(fig, out)
    return prompt_id


def fig6_style(out: Path) -> None:
    df = pd.read_csv(ROOT / "outputs/revision/controllability/style_controllability.csv")
    m = manifest()
    prompt_id = df.prompt_sample.iloc[0]
    rows = df[df.prompt_sample == prompt_id]
    fig, axes = plt.subplots(2, len(rows), figsize=(7.5, 5.2))
    for j, (_, row) in enumerate(rows.iterrows()):
        show(axes[0, j], DATA / m[row.reference_sample]["image_path"], f"Reference: {row.reference_group.replace('_', ' ')}",
             f"saturation {row.ref_mean_saturation:.2f}")
        show(axes[1, j], row.path, None, f"predicted: {row.predicted_group.replace('_', ' ')}; saturation {row.mean_saturation:.2f}")
    axes[0, 0].set_ylabel("Style reference\n(real, training split)", fontsize=6.5)
    axes[1, 0].set_ylabel("Output (Ours)", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    save_tiff(fig, out)


def fig8_failures(out: Path, run: str = "ours_s42", n: int = 4) -> None:
    df = per_image(run)
    m = manifest()
    df = df[df.path.str.contains("seed42")]
    picks = list(df.nsmallest(n // 2, "edge_iou").path) + list(df[df.style_correct == 0].sample(n - n // 2, random_state=SAMPLING_SEED).path)
    fig, axes = plt.subplots(3, n, figsize=(7.5, 6.2))
    for j, path in enumerate(picks):
        sid = Path(path).stem.split("__seed")[0]
        r = m[sid]
        row = df[df.path == path].iloc[0]
        show(axes[0, j], DATA / r["image_path"], "Real test painting" if j == 0 else None)
        axes[1, j].imshow(channel_images(DATA / r["structure_map_path"])[0], cmap="gray"); axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        show(axes[2, j], path, None, f"E {row.edge_iou:.3f} B {row.blank_iou:.3f}\ntarget {row.target_style}\npred. {row.pred_style}")
    fig.tight_layout(pad=0.3)
    save_tiff(fig, out)


def s_training_curves(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6))
    for ax, prefix in zip(axes, ["ours", "lora_only"]):
        for seed in [42, 52, 62]:
            log = TRAIN / f"{prefix}_s{seed}" / "train_log.csv"
            if log.exists():
                d = pd.read_csv(log)
                ax.plot(d.step, d.loss.rolling(20, min_periods=1).mean(), lw=0.8, label=f"seed {seed}")
        ax.set_title({"ours": "Ours (LoRA + adapter)", "lora_only": "LoRA-only"}[prefix], fontsize=8)
        ax.set_xlabel("optimiser step"); ax.set_ylabel("training loss (20-step mean)")
        ax.legend(fontsize=6)
    fig.tight_layout()
    save_tiff(fig, out)


def main(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "3" in args.which:
        fig3_fidelity(OUT / "Fig3.tif")
    if "4" in args.which:
        fig4_qualitative(OUT / "Fig4.tif", [("LoRA-only", "lora_only_s42"), ("ControlNet", "controlnet"), ("T2I-Adapter", "t2i_adapter"),
                                             ("IP-Adapter-only", "ip_adapter_only"), ("LoRA+ControlNet", "lora_controlnet_s42"),
                                             ("LoRA+IP-Adapter", "lora_ip_adapter_s42"), ("Ours", "ours_s42")])
    if "5" in args.which:
        fig5_structure(OUT / "Fig5.tif")
    if "6" in args.which:
        fig6_style(OUT / "Fig6.tif")
    if "7" in args.which:
        fig4_qualitative(OUT / "Fig7.tif", [("w/o structure\n(LoRA+IP-Adapter)", "lora_ip_adapter_s42"), ("w/o style ref.", "ours_no_style_s42"),
                                             ("w/o both\n(LoRA-only)", "lora_only_s42"), ("Full model", "ours_s42")], n=3)
    if "8" in args.which:
        fig8_failures(OUT / "Fig8.tif")
    if "S" in args.which:
        s_training_curves(OUT / "S4_Fig_training_curves.tif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", default="345678S")
    main(parser.parse_args())
