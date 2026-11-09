"""Figures 3-8 and supporting figures for the revision, built from outputs/revision.

Qualitative examples are drawn at random (numpy Generator with the seed given below) from the
test set; no example is hand-picked. Every generated image is annotated with its per-image
line-art IoU (E) and blank-space IoU (B) of structure_metrics_v8.py (structure re-extracted from
the output and compared with the input map); no evaluative labels are used.
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
    ("Token adapter", ["ours_s42", "ours_s52", "ours_s62"]),
    ("Ours w/o style ref.", ["ours_v8_no_style_s42", "ours_v8_no_style_s52", "ours_v8_no_style_s62"]),
    ("Ours, line art only", ["ours_v8_lineart_s42", "ours_v8_lineart_s52", "ours_v8_lineart_s62"]),
    ("Ours", ["ours_v8_s42", "ours_v8_s52", "ours_v8_s62"]),
]
CTRL = ROOT / "outputs/revision/controllability_v8_ours"


def num(x: float) -> str:
    """Score for an annotation; IoU is undefined (n/a) when both masks are empty."""
    return "n/a" if pd.isna(x) else f"{x:.3f}"


def manifest() -> dict:
    rows = [json.loads(l) for l in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["sample_id"]: r for r in rows}


def summary(run: str) -> dict:
    return json.loads((EVAL / run / "summary.json").read_text(encoding="utf-8"))


def per_image(run: str) -> pd.DataFrame:
    """Per-image metrics of a run: evaluate_v7 columns plus the v8 structure scores (blank_iou_re etc.)."""
    df = pd.read_csv(EVAL / run / "per_image_metrics.csv")
    v8 = pd.read_csv(EVAL / run / "structure_v8_per_image.csv")[["path", "edge_iou", "blank_iou_re", "layer_acc", "salience_iou"]]
    df = df.drop(columns=["edge_iou"]).merge(v8, on="path", how="left")
    # style_v8_per_image.csv rows follow the sorted image paths of the run (style_v8.py)
    style = pd.read_csv(EVAL / run / "style_v8_per_image.csv")
    paths = sorted(df.path, key=lambda p: Path(p).name)
    if len(paths) == len(style):
        style["path"] = paths
        df = df.merge(style[["path", "target", "pred", "correct"]].rename(
            columns={"target": "target_colour", "pred": "pred_colour", "correct": "colour_correct"}), on="path", how="left")
    return df


def show(ax, path, title=None, note=None, gray=False):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path  # result tables store paths relative to the repository root
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
        if key != "fid":  # bootstrap FID is biased upwards by duplicated images; no interval shown
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
    return f"E {num(r.edge_iou)}  B {num(r.blank_iou_re)}"


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
    df = pd.read_csv(CTRL / "structure_controllability.csv")
    m = manifest()
    prompt_id = df.prompt_sample.iloc[0]
    rows = df[df.prompt_sample == prompt_id].iloc[:n]
    fig, axes = plt.subplots(n, 5, figsize=(7.5, 7.5 * n / 5 * 1.12))
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
    fig.tight_layout(pad=0.3, h_pad=1.6)
    fig.subplots_adjust(bottom=0.06)
    save_tiff(fig, out)
    return prompt_id


def fig6_style(out: Path) -> None:
    df = pd.read_csv(CTRL / "style_controllability.csv")
    m = manifest()
    prompt_id = df.prompt_sample.iloc[0]
    rows = df[df.prompt_sample == prompt_id]
    fig, axes = plt.subplots(2, len(rows), figsize=(7.5, 5.2))
    for j, (_, row) in enumerate(rows.iterrows()):
        show(axes[0, j], DATA / m[row.reference_sample]["image_path"], f"Reference: {row.reference_group.replace('_', ' ')}",
             f"saturation {row.ref_mean_saturation:.2f}")
        show(axes[1, j], row.path, None, f"saturation {row.mean_saturation:.2f}")
    axes[0, 0].set_ylabel("Style reference\n(real, training split)", fontsize=6.5)
    axes[1, 0].set_ylabel("Output (Ours)", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    save_tiff(fig, out)


def fig8_failures(out: Path, run: str = "ours_v8_s42", n: int = 4) -> None:
    """Two lowest line-art IoU outputs and two random colour-treatment misclassifications (seed-42 images)."""
    df = per_image(run)
    m = manifest()
    df = df[df.path.str.contains("seed42")]
    lowest = list(df.nsmallest(n // 2, "edge_iou").path)
    wrong = df[(df.colour_correct == 0) & ~df.path.isin(lowest)]
    picks = lowest + list(wrong.sample(n - n // 2, random_state=SAMPLING_SEED).path)
    fig, axes = plt.subplots(3, n, figsize=(7.5, 6.8))
    for j, path in enumerate(picks):
        sid = Path(path).stem.split("__seed")[0]
        r = m[sid]
        row = df[df.path == path].iloc[0]
        show(axes[0, j], DATA / r["image_path"], "Real test painting" if j == 0 else None)
        axes[1, j].imshow(channel_images(DATA / r["structure_map_path"])[0], cmap="gray"); axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        show(axes[2, j], path, None, f"E {num(row.edge_iou)} B {num(row.blank_iou_re)}\ntarget {row.target_colour}\npred. {row.pred_colour}")
    fig.tight_layout(pad=0.3)
    save_tiff(fig, out)


def fig2_framework(out: Path) -> None:
    """Schematic of the implemented model (v8) with the inputs and output of one random test painting."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    m = manifest()
    sid = pick_random_tests(1)[0]
    r = m[sid]
    ref = m[r["extra"]["style_reference_id"]]
    fig = plt.figure(figsize=(7.5, 3.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(x, y, w, h, text, colour, trained=False, size=6.3, inner=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", lw=1.2 if trained else 0.7,
                                    ec="#c44e52" if trained else ("#999999" if inner else "#555555"), fc=colour,
                                    ls="-" if trained or inner else "--"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, wrap=True)

    def arrow(x0, y0, x1, y1, text=None, at=None, ha="center", va="bottom"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="#333333"))
        if text:
            tx, ty = at if at else ((x0 + x1) / 2, (y0 + y1) / 2 + 1.2)
            ax.text(tx, ty, text, ha=ha, va=va, fontsize=5.3, color="#333333")

    def thumb(x, y, w, h, image, gray=False):
        sub = fig.add_axes([x / 100, y / 52, w / 100, h / 52])
        sub.imshow(image, cmap="gray" if gray else None, vmin=0, vmax=255)
        sub.set_xticks([]); sub.set_yticks([])
        for s in sub.spines.values():
            s.set_linewidth(0.3)

    # inputs
    ax.text(9, 50.5, "Inputs", ha="center", fontsize=7, weight="bold")
    box(1, 40.5, 16, 7.5, "Structured prompt $p$\n(nine fields, VLM)", "#eef3fb")
    chans = channel_images(DATA / r["structure_map_path"])
    for k, c in enumerate(chans):
        thumb(1 + 4.1 * k, 25.5, 3.8, 9, np.asarray(Image.fromarray(c).resize((96, 128))), gray=True)
    ax.text(9, 23.3, "Condition map $s$: line art, layer,\nblank space, salience", ha="center", va="top", fontsize=5.5)
    thumb(4, 4, 10, 12, np.asarray(Image.open(DATA / ref["image_path"]).convert("RGB").resize((160, 192))))
    ax.text(9, 2.2, "Style reference $r$ (training split)", ha="center", fontsize=5.5)

    # encoders
    box(22, 41, 17, 6.5, "SDXL text encoders\n(frozen)", "#eef3fb")
    box(22, 22, 17, 14, "Structure branch $B_\\psi$\nControlNet copy of the\nSDXL encoder (init. Canny\nControlNet, 4-ch input)\ntrained: condition embedding,\nzero convs $P_l$, LoRA r16", "#fdecea", trained=True, size=5.6)
    box(22, 6, 17, 8, "IP-Adapter image\nencoder + projection\n(frozen)", "#eef6ee")
    arrow(17.5, 44.2, 21.5, 44.2)
    arrow(17.5, 30, 21.5, 30)
    arrow(14.5, 10, 21.5, 10)
    arrow(30.5, 40.5, 30.5, 36.5, "$z_t$, $t$, text", at=(31.3, 38.5), ha="left", va="center")

    # U-Net
    box(46, 8, 24, 39.5, "", "#fff6e6", trained=True)
    ax.text(58, 44.5, "SDXL U-Net + LoRA (r8, trained)", ha="center", fontsize=6.5, weight="bold")
    box(48.5, 34, 19, 7.5, "cross-attention to text\n$\\mathrm{Softmax}(QK_c^\\top/\\sqrt{d})V_c$", "#ffffff", size=5.5, inner=True)
    box(48.5, 22, 19, 9, "encoder stages + middle block\n$\\tilde h_l = h_l + \\lambda_s P_l(F_s^l)$\n($\\lambda_s=1$)", "#ffffff", size=5.5, inner=True)
    box(48.5, 10.5, 19, 8.5, "decoupled cross-attention\n$+\\lambda_r\\,\\mathrm{Softmax}(QK_r^\\top/\\sqrt{d})V_r$\n($\\lambda_r=0.55$)", "#ffffff", size=5.5, inner=True)
    arrow(39.5, 44.2, 48, 38, "text tokens", at=(42.3, 44.8))
    arrow(39.5, 29, 48, 26.5, "$F_s^l$\n(multi-\nscale)", at=(42.6, 29.0))
    arrow(39.5, 10, 48, 14, "image\ntokens", at=(42.5, 13.2))

    # output
    ax.text(88, 50.5, "Output", ha="center", fontsize=7, weight="bold")
    arrow(70.5, 27.5, 77.5, 27.5, "30 Euler\nsteps,\nCFG 7.5,\nVAE\ndecode", at=(74, 26), va="top")
    out_img = GEN / "ours_v8_s42" / f"{sid}__seed42.png"
    thumb(78, 14, 20, 27, np.asarray(Image.open(out_img).convert("RGB").resize((256, 256))))
    ax.text(88, 12, "Generated painting\n(model output)", ha="center", va="top", fontsize=5.5)
    ax.text(1, 0.3, "Solid red outline: trained on the benchmark training split; dashed grey outline: frozen public checkpoint.",
            fontsize=5.3, color="#555555")
    save_tiff(fig, out)
    (OUT / "Fig2_sources.json").write_text(json.dumps({"condition_and_output": sid, "style_reference": ref["sample_id"]}, indent=2), encoding="utf-8")


def fig1_motivation(out: Path) -> None:
    """Motivation figure built only from benchmark images and existing outputs (replaces the original Fig 1).

    Panel A: a real training painting from a museum source; panel B: unadapted SDXL outputs for test prompts;
    panel C: the inputs and output of the proposed model for one test painting. All examples are drawn at
    random (SAMPLING_SEED) from paintings not shown in Figs 2, 4, 5, 7 and 8; the labels only describe.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    m = manifest()
    fig2 = json.loads((OUT / "Fig2_sources.json").read_text(encoding="utf-8"))
    used = set(pick_random_tests(4)) | {fig2["condition_and_output"], fig2["style_reference"]}
    ctrl = pd.read_csv(CTRL / "structure_controllability.csv")
    p0 = ctrl.prompt_sample.iloc[0]
    used |= {p0} | set(ctrl[ctrl.prompt_sample == p0].structure_sample)
    f8 = per_image("ours_v8_s42")
    used |= {Path(p).stem.split("__seed")[0] for p in f8[f8.path.str.contains("seed42")].nsmallest(2, "edge_iou").path}
    rng = np.random.default_rng(SAMPLING_SEED + 1)
    test = [s for s in sorted(s for s, r in m.items() if r["split"] == "test") if s not in used]
    picks = list(rng.choice(test, 4, replace=False))
    sdxl_ids, cid = picks[:3], picks[3]
    met_train = sorted(s for s, r in m.items() if r["split"] == "train" and r["source"] == "met"
                       and r["height"] > 1.6 * r["width"])
    aid = str(rng.choice(met_train))
    c = m[cid]
    ref = m[c["extra"]["style_reference_id"]]

    fig = plt.figure(figsize=(7.5, 3.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def panel(x, w, letter, title, colour):
        ax.add_patch(FancyBboxPatch((x, 1), w, 47, boxstyle="round,pad=0.2,rounding_size=1", lw=0.6, ec="#bbbbbb", fc=colour))
        ax.text(x + 1.2, 46, letter, fontsize=8, weight="bold", va="center")
        ax.text(x + w / 2 + 1, 46, title, fontsize=6.8, weight="bold", ha="center", va="center")

    def thumb(x, y, w, h, image, gray=False):
        sub = fig.add_axes([x / 100, y / 50, w / 100, h / 50])
        sub.imshow(image, cmap="gray" if gray else None, vmin=0, vmax=255, aspect="auto" if gray else "equal")
        sub.set_xticks([]); sub.set_yticks([])
        for s in sub.spines.values():
            s.set_linewidth(0.3)
        return sub

    def label(x, y, text, size=5.6, **kw):
        ax.text(x, y, text, fontsize=size, ha="center", va="center", **kw)

    # A: real painting and the aspects of pictorial organisation that motivate the conditions
    panel(1, 29, "A", "Pictorial organisation", "#eef4f3")
    a = m[aid]
    img = Image.open(DATA / a["image_path"]).convert("RGB")
    img.thumbnail((300, 600))
    thumb(2.5, 6, 12, 37, np.asarray(img))
    for k, t in enumerate(["Layering of\nnear and far", "Blank space\n(liubai)", "Brush and ink", "Colour treatment\n(ink / ink and colour)"]):
        ax.add_patch(FancyBboxPatch((16.5, 34 - 8.5 * k), 12.5, 6, boxstyle="round,pad=0.2,rounding_size=0.6", lw=0.6, ec="#5f9e96", fc="white"))
        label(22.75, 37 - 8.5 * k, t)
    label(15.5, 3.6, "Real painting (benchmark training split; S2 Table)", size=5.2, color="#444444")

    # B: unadapted SDXL from benchmark prompts
    panel(33, 29, "B", "Unadapted SDXL, prompt only", "#f6eff4")
    label(47.5, 40.5, "Inputs: text prompt only; no condition\nspecifies where masses, water and mist go", size=5.4)
    for k, sid in enumerate(sdxl_ids):
        thumb(34.2 + 9.2 * k, 17, 8.6, 17.2, np.asarray(Image.open(GEN / "sdxl_base" / f"{sid}__seed42.png").convert("RGB").resize((256, 256))))
    label(47.5, 13.5, "Model outputs for three randomly\ndrawn test prompts (seed 42)", size=5.2, color="#444444")

    # C: this work
    panel(65, 34, "C", "This work", "#eef4ea")
    chans = channel_images(DATA / c["structure_map_path"])
    for k, (ch, name) in enumerate(zip(chans, ["line art", "layer", "blank", "salience"])):
        thumb(66.3 + 4.3 * k, 30, 4, 8.5, np.asarray(Image.fromarray(ch).resize((96, 160))), gray=True)
        label(68.3 + 4.3 * k, 28.6, name, size=4.6)
    label(74.8, 41.5, "Prompt + four-channel condition map", size=5.4)
    thumb(69.5, 9, 10, 14, np.asarray(Image.open(DATA / ref["image_path"]).convert("RGB").resize((160, 200))))
    label(74.5, 6.8, "Style reference\n(training split)", size=5.0)
    ax.add_patch(FancyArrowPatch((84.2, 24), (87.2, 24), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="#333333"))
    thumb(87.6, 15, 10.8, 18, np.asarray(Image.open(GEN / "ours_v8_s42" / f"{cid}__seed42.png").convert("RGB").resize((256, 256))))
    label(93, 39, "SDXL + LoRA\n+ residual structure\nbranch + IP-Adapter", size=5.0)
    label(93, 11.5, "Model output\n(seed 42)", size=5.0)
    save_tiff(fig, out)
    (OUT / "Fig1_sources.json").write_text(json.dumps({"panel_A_real_painting": aid, "panel_B_sdxl_prompts": sdxl_ids,
                                                       "panel_C_condition_and_output": cid, "panel_C_style_reference": ref["sample_id"]},
                                                      indent=2), encoding="utf-8")


def s_training_curves(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6))
    for ax, prefix in zip(axes, ["v8_cn", "lora_only"]):
        for seed in [42, 52, 62]:
            log = TRAIN / f"{prefix}_s{seed}" / "train_log.csv"
            if log.exists():
                d = pd.read_csv(log)
                ax.plot(d.step, d.loss.rolling(20, min_periods=1).mean(), lw=0.8, label=f"seed {seed}")
        ax.set_title({"v8_cn": "Ours (LoRA + residual structure branch)", "lora_only": "LoRA-only"}[prefix], fontsize=8)
        ax.set_xlabel("optimiser step"); ax.set_ylabel("training loss (20-step mean)")
        ax.legend(fontsize=6)
    fig.tight_layout()
    save_tiff(fig, out)


def main(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "1" in args.which:
        fig1_motivation(OUT / "Fig1.tif")
    if "2" in args.which:
        fig2_framework(OUT / "Fig2.tif")
    if "3" in args.which:
        fig3_fidelity(OUT / "Fig3.tif")
    if "4" in args.which:
        fig4_qualitative(OUT / "Fig4.tif", [("LoRA-only", "lora_only_s42"), ("ControlNet", "controlnet"), ("IP-Adapter-only", "ip_adapter_only"),
                                             ("LoRA+ControlNet", "lora_controlnet_s42"), ("LoRA+IP-Adapter", "lora_ip_adapter_s42"),
                                             ("Token adapter", "ours_s42"), ("Ours", "ours_v8_s42")])
    if "5" in args.which:
        fig5_structure(OUT / "Fig5.tif")
    if "6" in args.which:
        fig6_style(OUT / "Fig6.tif")
    if "7" in args.which:
        fig4_qualitative(OUT / "Fig7.tif", [("w/o structure branch\n(LoRA+IP-Adapter)", "lora_ip_adapter_s42"), ("w/o style ref.", "ours_v8_no_style_s42"),
                                             ("Line art only", "ours_v8_lineart_s42"), ("Full model", "ours_v8_s42")], n=3)
    if "8" in args.which:
        fig8_failures(OUT / "Fig8.tif")
    if "S" in args.which:
        s_training_curves(OUT / "S4_Fig_training_curves.tif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", default="345678S")
    main(parser.parse_args())
