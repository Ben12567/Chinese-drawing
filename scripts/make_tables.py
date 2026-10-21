"""Generate the LaTeX tables and number macros of the revised manuscript from result files.

Every number in the Results section is produced here from outputs/revision so that nothing is
transcribed by hand. Writes revision/generated/tables.tex and revision/generated/numbers.tex.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT.parent / "revision" / "generated"
EVAL = ROOT / "outputs/revision/eval"
STATS = ROOT / "outputs/revision/stats"
DATA = ROOT / "data/processed/clp_v7"

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
CHANNELS = [
    ("Line art only", "ch_lineart_s42"), ("Layer map only", "ch_depth_s42"), ("Blank mask only", "ch_blank_s42"),
    ("Salience only", "ch_saliency_s42"), ("Line art + layer", "ch_lineart_depth_s42"), ("w/o salience", "ch_no_saliency_s42"),
    ("All four (Ours, seed 42)", "ours_s42"),
]


def load(run: str) -> dict:
    return json.loads((EVAL / run / "summary.json").read_text(encoding="utf-8"))


def fmt(values: list[float], digits: int) -> str:
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} $\\pm$ {np.std(values, ddof=1):.{digits}f}"


def fmt_ci(summaries: list[dict], key: str, digits: int, scale: float = 1.0) -> str:
    vals = [s["plain"][key] * scale for s in summaries]
    lo, hi = (v * scale for v in summaries[0]["plain"][f"{key}_ci"])
    main = fmt(vals, digits)
    if len(vals) == 1:
        return f"{main} [{lo:.{digits}f}, {hi:.{digits}f}]"
    return main


def bold_best(cells: list[str], values: list[float], lower_better: bool) -> list[str]:
    best = int(np.argmin(values) if lower_better else np.argmax(values))
    return [f"\\textbf{{{c}}}" if i == best else c for i, c in enumerate(cells)]


def main_tables() -> str:
    real = load("real_test")
    rows_fid, rows_struct = [], []
    for name, runs in MAIN:
        s = [load(r) for r in runs]
        rows_fid.append((name, s))
    lines = []
    lines.append("% Table: fidelity")
    lines.append("\\begin{tabular}{lccc}\n\\hline\n\\textbf{Method} & \\textbf{KID}$\\times10^{3}\\downarrow$ & \\textbf{CMMD}$\\downarrow$ & \\textbf{FID}$\\downarrow$ \\\\\n\\hline")
    kid_vals = [np.mean([x["plain"]["kid"] for x in s]) for _, s in rows_fid]
    cmmd_vals = [np.mean([x["plain"]["cmmd"] for x in s]) for _, s in rows_fid]
    fid_vals = [np.mean([x["plain"]["fid"] for x in s]) for _, s in rows_fid]
    kid_c = bold_best([fmt_ci(s, "kid", 1, 1000) for _, s in rows_fid], kid_vals, True)
    cmmd_c = bold_best([fmt_ci(s, "cmmd", 3) for _, s in rows_fid], cmmd_vals, True)
    fid_c = bold_best([fmt_ci(s, "fid", 1) for _, s in rows_fid], fid_vals, True)
    for (name, _), k, c, f in zip(rows_fid, kid_c, cmmd_c, fid_c):
        lines.append(f"{name} & {k} & {c} & {f} \\\\")
    lines.append("\\hline")
    lines.append(f"Real vs.\\ real (val vs.\\ test) & {real['plain']['kid'] * 1000:.1f} & {real['plain']['cmmd']:.3f} & {real['plain']['fid']:.1f} \\\\")
    lines.append("\\hline\n\\end{tabular}")

    lines.append("% Table: per-sample metrics")
    lines.append("\\begin{tabular}{lcccccccc}\n\\hline\n\\textbf{Method} & \\textbf{CLIP}$\\uparrow$ & \\textbf{Pick}$\\uparrow$ & \\textbf{HPSv2}$\\uparrow$ & \\textbf{LPIPS div.}$\\uparrow$ & \\textbf{Style acc.}$\\uparrow$ & \\textbf{Style F1}$\\uparrow$ & \\textbf{Edge IoU}$\\uparrow$ & \\textbf{Blank IoU}$\\uparrow$ \\\\\n\\hline")
    cols = [("clipscore", 3), ("pickscore", 2), ("hpsv2", 3), ("lpips_diversity", 3), ("style_correct", 3), (None, 3), ("edge_iou", 3), ("blank_iou", 3)]
    table = []
    for name, s in rows_fid:
        cells, vals = [], []
        for key, d in cols:
            v = [x["style"]["macro_f1"] for x in s] if key is None else [x[key] for x in s]
            cells.append(fmt(v, d))
            vals.append(np.mean(v))
        table.append((name, cells, vals))
    for j in range(len(cols)):
        best = int(np.nanargmax([t[2][j] for t in table]))
        table[best][1][j] = f"\\textbf{{{table[best][1][j]}}}"
    for name, cells, _ in table:
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("Real test paintings & " + " & ".join(
        f"{real[k]:.{d}f}" if k else f"{real['style']['macro_f1']:.3f}" for k, d in cols) + " \\\\")
    lines.append("\\hline\n\\end{tabular}")
    return "\n".join(lines)


def tests_table() -> str:
    df = pd.read_csv(STATS / "paired_tests.csv")
    names = {"clipscore": "CLIPScore", "pickscore": "PickScore", "hpsv2": "HPSv2", "style_correct": "Style acc.",
             "edge_iou": "Edge IoU", "blank_iou": "Blank IoU", "lpips_diversity": "LPIPS div."}
    lines = ["\\begin{tabular}{llcccc}\n\\hline\n\\textbf{Baseline} & \\textbf{Metric} & \\textbf{Diff. (Ours $-$ baseline) [95\\% CI]} & $d_z$ & $p_t$ (Holm) & $p_W$ (Holm) \\\\\n\\hline"]
    for comp, group in df.groupby("comparison", sort=False):
        first = True
        for _, r in group.iterrows():
            base = comp.split(" vs ")[1] if first else ""
            first = False
            pt = f"{r.p_t_holm:.2g}" if r.p_t_holm >= 1e-3 else f"$<$0.001"
            pw = f"{r.p_wilcoxon_holm:.2g}" if r.p_wilcoxon_holm >= 1e-3 else f"$<$0.001"
            lines.append(f"{base} & {names[r.metric]} & {r.mean_diff:+.4f} [{r.diff_ci_lo:+.4f}, {r.diff_ci_hi:+.4f}] & {r.cohens_dz:+.2f} & {pt} & {pw} \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def channel_table() -> str:
    lines = ["\\begin{tabular}{lcccccc}\n\\hline\n\\textbf{Structure channels} & \\textbf{KID}$\\times10^3\\downarrow$ & \\textbf{CMMD}$\\downarrow$ & \\textbf{FID}$\\downarrow$ & \\textbf{Style acc.}$\\uparrow$ & \\textbf{Edge IoU}$\\uparrow$ & \\textbf{Blank IoU}$\\uparrow$ \\\\\n\\hline"]
    for name, run in CHANNELS:
        s = load(run)
        lines.append(f"{name} & {s['plain']['kid']*1000:.1f} [{s['plain']['kid_ci'][0]*1000:.1f}, {s['plain']['kid_ci'][1]*1000:.1f}] & {s['plain']['cmmd']:.3f} & {s['plain']['fid']:.1f} & {s['style_correct']:.3f} & {s['edge_iou']:.4f} & {s['blank_iou']:.4f} \\\\")
    lines.append("\\hline\n\\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    REV.mkdir(parents=True, exist_ok=True)
    parts = {"main": main_tables, "tests": tests_table, "channels": channel_table}
    out = []
    for key, fn in parts.items():
        try:
            out.append(f"%%% {key}\n" + fn())
        except FileNotFoundError as exc:
            out.append(f"%%% {key}: missing {exc.filename}")
    (REV / "tables.tex").write_text("\n\n".join(out), encoding="utf-8")
    print((REV / "tables.tex").read_text(encoding="utf-8")[:3000])


if __name__ == "__main__":
    main()
