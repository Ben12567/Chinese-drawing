"""Generate the LaTeX tables and number macros of the revised manuscript from result files.

Every number in the Results section is produced here from outputs/revision so that nothing is
transcribed by hand. Writes revision/generated/tables.tex and revision/generated/numbers.tex.
Run scripts/stats_v8.py first (outputs/revision/stats_v8).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT.parent / "revision" / "generated"
EVAL = ROOT / "outputs/revision/eval"
STATS = ROOT / "outputs/revision/stats_v8"
SELECT = ROOT / "outputs/revision/v8_selection"
CTRL = {"Ours": ROOT / "outputs/revision/controllability_v8_ours",
        "LoRA+ControlNet": ROOT / "outputs/revision/controllability_lora_controlnet"}

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
ABLATION = [
    ("All four channels (Ours)", "ours_v8_s42"), ("Line art only", "ours_v8_lineart_s42"),
    ("w/o layer map", "ours_v8_no_layer_s42"), ("w/o blank-space mask", "ours_v8_no_blank_s42"),
    ("w/o salience mask", "ours_v8_no_salience_s42"), ("w/o style reference", "ours_v8_no_style_s42"),
]
STRUCT = [("edge_iou", "Line art"), ("blank_iou_re", "Blank space"), ("layer_acc", "Layer"), ("salience_iou", "Salience")]
METRIC_NAMES = {"clipscore": "CLIPScore", "pickscore": "PickScore", "hpsv2": "HPSv2", "lpips_diversity": "LPIPS div.",
                "edge_iou_specific": "Line-art spec.", "blank_iou_re_specific": "Blank-space spec.",
                "layer_acc_specific": "Layer spec.", "salience_iou_specific": "Salience spec.",
                "style_v8_correct": "Colour-treatment acc."}


def load(run: str) -> dict:
    d = EVAL / run
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    for extra in ["structure_v8", "style_v8"]:
        p = d / f"{extra}_summary.json"
        if p.exists():
            s[extra] = json.loads(p.read_text(encoding="utf-8"))
    return s


def fmt(values: list[float], digits: int, scale: float = 1.0) -> str:
    values = [v * scale for v in values]
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} $\\pm$ {np.std(values, ddof=1):.{digits}f}"


def fmt_ci(summaries: list[dict], key: str, digits: int, scale: float = 1.0) -> str:
    vals = [s["plain"][key] for s in summaries]
    main = fmt(vals, digits, scale)
    if len(vals) == 1:
        lo, hi = (v * scale for v in summaries[0]["plain"][f"{key}_ci"])
        return f"{main} [{lo:.{digits}f}, {hi:.{digits}f}]"
    return main


def bold_best(cells: list[str], values: list[float], lower_better: bool) -> list[str]:
    best = int(np.nanargmin(values) if lower_better else np.nanargmax(values))
    return [f"\\textbf{{{c}}}" if i == best else c for i, c in enumerate(cells)]


def p_cell(p: float) -> str:
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def fidelity_table(rows) -> str:
    real = load("real_test")
    lines = ["\\begin{tabular}{lccc}\n\\hline\n\\textbf{Method} & \\textbf{KID}$\\times10^{3}\\downarrow$ & \\textbf{CMMD}$\\downarrow$ & \\textbf{FID}$\\downarrow$ \\\\\n\\hline"]
    cols = []
    for key, digits, scale in [("kid", 1, 1000), ("cmmd", 3, 1), ("fid", 1, 1)]:
        # Resampling with replacement duplicates images, which biases bootstrap FID upwards (the interval can
        # exclude the point estimate), so FID is given without an interval.
        cells = [fmt_ci(s, key, digits, scale) if key != "fid" else fmt([x["plain"]["fid"] for x in s], digits) for _, s in rows]
        cols.append(bold_best(cells, [np.mean([x["plain"][key] for x in s]) for _, s in rows], True))
    for i, (name, _) in enumerate(rows):
        lines.append(f"{name} & {cols[0][i]} & {cols[1][i]} & {cols[2][i]} \\\\")
    lines.append("\\hline")
    lines.append(f"Real vs.\\ real (val vs.\\ test) & {real['plain']['kid'] * 1000:.1f} & {real['plain']['cmmd']:.3f} & {real['plain']['fid']:.1f} \\\\")
    lines.append("\\hline\n\\end{tabular}")
    return "\n".join(lines)


def structure_table(rows) -> str:
    real = load("real_test")["structure_v8"]
    head = " & ".join(f"\\textbf{{{label}}}" for _, label in STRUCT)
    lines = ["\\begin{tabular}{l" + "c" * len(STRUCT) + "}\n\\hline\n\\textbf{Method} & " + head + " \\\\\n\\hline"]
    cols = []
    for key, _ in STRUCT:
        cells = [f"{fmt([x['structure_v8'][key + '_specific'] for x in s], 3)} ({np.mean([x['structure_v8'][key] for x in s]):.3f})" for _, s in rows]
        cols.append(bold_best(cells, [np.mean([x["structure_v8"][key + "_specific"] for x in s]) for _, s in rows], False))
    for i, (name, _) in enumerate(rows):
        lines.append(f"{name} & " + " & ".join(c[i] for c in cols) + " \\\\")
    lines.append("\\hline")
    lines.append("Real test paintings & " + " & ".join(f"{real[k + '_specific']:.3f} ({real[k]:.3f})" for k, _ in STRUCT) + " \\\\")
    lines.append("\\hline\n\\end{tabular}")
    return "\n".join(lines)


def quality_table(rows) -> str:
    real = load("real_test")
    cols_def = [("clipscore", 3), ("pickscore", 2), ("hpsv2", 3), ("lpips_diversity", 3), ("style_ba", 3), ("style_f1", 3)]

    def get(x, key):
        if key == "style_ba":
            return x["style_v8"]["balanced_accuracy"]
        if key == "style_f1":
            return x["style_v8"]["macro_f1"]
        return x[key]
    lines = ["\\begin{tabular}{lcccccc}\n\\hline\n\\textbf{Method} & \\textbf{CLIP}$\\uparrow$ & \\textbf{Pick}$\\uparrow$ & \\textbf{HPSv2}$\\uparrow$ & \\textbf{LPIPS div.}$\\uparrow$ & \\textbf{Colour bal.\\ acc.}$\\uparrow$ & \\textbf{Colour F1}$\\uparrow$ \\\\\n\\hline"]
    cols = []
    for key, d in cols_def:
        vals = [[get(x, key) for x in s] for _, s in rows]
        cols.append(bold_best([fmt(v, d) for v in vals], [np.mean(v) for v in vals], False))
    for i, (name, _) in enumerate(rows):
        lines.append(f"{name} & " + " & ".join(c[i] for c in cols) + " \\\\")
    lines.append("\\hline")
    lines.append("Real test paintings & " + " & ".join(f"{get(real, k):.{d}f}" if k != "lpips_diversity" else "--" for k, d in cols_def) + " \\\\")
    lines.append("\\hline\n\\end{tabular}")
    return "\n".join(lines)


def tests_table(csv: Path, keep: list[str] | None = None) -> str:
    df = pd.read_csv(csv)
    lines = ["\\begin{tabular}{llccc}\n\\hline\n\\textbf{Comparison} & \\textbf{Metric} & \\textbf{Diff.\\ [95\\% CI]} & $d_z$ & $p_W$ (Holm) \\\\\n\\hline"]
    for comp, group in df.groupby("comparison", sort=False):
        base = comp.split(" vs ")[1]
        if keep is not None and base not in keep:
            continue
        first = True
        for _, r in group.iterrows():
            label = f"vs.\\ {base}" if first else ""
            first = False
            lines.append(f"{label} & {METRIC_NAMES[r.metric]} & {r.mean_diff:+.3f} [{r.diff_ci_lo:+.3f}, {r.diff_ci_hi:+.3f}] & {r.cohens_dz:+.2f} & {p_cell(r.p_wilcoxon_holm)} \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def ablation_table() -> str:
    lines = ["\\begin{tabular}{lccccccc}\n\\hline\n\\textbf{Variant (seed 42)} & \\textbf{KID}$\\times10^3\\downarrow$ & \\textbf{CMMD}$\\downarrow$ & \\textbf{FID}$\\downarrow$ & \\textbf{Line art} & \\textbf{Blank} & \\textbf{Layer} & \\textbf{Salience} \\\\\n\\hline"]
    for name, run in ABLATION:
        s = load(run)
        t = s["structure_v8"]
        lines.append(f"{name} & {s['plain']['kid']*1000:.1f} [{s['plain']['kid_ci'][0]*1000:.1f}, {s['plain']['kid_ci'][1]*1000:.1f}] & {s['plain']['cmmd']:.3f} & {s['plain']['fid']:.1f} & "
                     + " & ".join(f"{t[k + '_specific']:.3f}" for k, _ in STRUCT) + " \\\\")
    lines.append("\\hline\n\\end{tabular}")
    return "\n".join(lines)


def selection_table() -> str:
    names = {"v8_cn": "ControlNet init., $\\lambda_s=1.0$", "v8_cn_ls": "ControlNet init.\\ + $\\mathcal{L}_{\\mathrm{struct}}$",
             "v8_t2i": "T2I-Adapter init.", "v8_t2i_ls": "T2I-Adapter init.\\ + $\\mathcal{L}_{\\mathrm{struct}}$",
             "v8_cn_infer08": "ControlNet init., $\\lambda_s=0.8$", "v8_cn_infer06": "ControlNet init., $\\lambda_s=0.6$"}
    lines = ["\\begin{tabular}{llcccccc}\n\\hline\n\\textbf{Stage} & \\textbf{Candidate} & \\textbf{KID}$\\times10^3$ & \\textbf{CMMD} & \\textbf{Line art} & \\textbf{Blank} & \\textbf{Layer} & \\textbf{Mean rank} \\\\\n\\hline"]
    for stage in ["stage1", "stage2"]:
        df = pd.read_csv(SELECT / f"{stage}.csv", index_col=0)
        for i, (cand, r) in enumerate(df.iterrows()):
            label = ("1" if stage == "stage1" else "2") if i == 0 else ""
            lines.append(f"{label} & {names[cand]} & {r.kid*1000:.1f} & {r.cmmd:.3f} & {r.edge_iou_specific:.3f} & {r.blank_iou_re_specific:.3f} & {r.layer_acc_specific:.3f} & {r.mean_rank:.1f} \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def controllability_table() -> str:
    lines = ["\\begin{tabular}{lcccc}\n\\hline\n\\textbf{Method} & \\textbf{Edge IoU, given map} & \\textbf{Edge IoU, other maps} & \\textbf{Blank IoU, given map} & \\textbf{Blank IoU, other maps} \\\\\n\\hline"]
    for name, d in CTRL.items():
        p = d / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text(encoding="utf-8"))["structure"]
        lines.append(f"{name} & {s['edge_iou_used']:.3f} & {s['edge_iou_unused']:.3f} & {s['blank_iou_used']:.3f} & {s['blank_iou_unused']:.3f} \\\\")
    lines.append("\\hline\n\\end{tabular}")
    return "\n".join(lines)


def diagnosis_table() -> str:
    """LoRA-only diagnosis (R1 major issue 4): prompts x initial noise, seed-42 images."""
    runs = [("Original template", "shared", "diag_lora42_v6template_shared"),
            ("Original template", "per sample", "diag_lora42_v6template_per_sample"),
            ("Structured (VLM)", "shared", "diag_lora42_structured_shared"),
            ("Structured (VLM)", "per sample", "lora_only_s42")]
    lines = ["\\begin{tabular}{llcccc}\n\\hline\n\\textbf{Prompts} & \\textbf{Initial noise} & \\textbf{Inter-sample LPIPS} & \\textbf{Near-duplicate pairs} & \\textbf{KID}$\\times10^3$ & \\textbf{CMMD} \\\\\n\\hline"]
    for prompts, noise, run in runs:
        s = load(run)
        i = json.loads((EVAL / run / "intersample.json").read_text(encoding="utf-8"))
        lines.append(f"{prompts} & {noise} & {i['inter_sample_lpips']:.3f} & {100 * i['near_duplicate_fraction']:.1f}\\% & {s['plain']['kid'] * 1000:.1f} & {s['plain']['cmmd']:.3f} \\\\")
    lines.append("\\hline\n\\end{tabular}")
    return "\n".join(lines)


def numbers(rows) -> str:
    """LaTeX macros for the numbers quoted in the text."""
    by = dict(rows)
    tests = pd.read_csv(STATS / "paired_tests.csv")
    out = []

    def macro(name: str, value: str) -> None:
        out.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    for tag, method in [("Ours", "Ours"), ("LoraCN", "LoRA+ControlNet"), ("CN", "ControlNet"), ("Token", "Token adapter"),
                        ("NoStyle", "Ours w/o style ref."), ("Lineart", "Ours, line art only"), ("LoraIP", "LoRA+IP-Adapter"),
                        ("LoraOnly", "LoRA-only"), ("SDXL", "SDXL")]:
        s = by[method]
        macro(f"kid{tag}", f"{np.mean([x['plain']['kid'] for x in s]) * 1000:.1f}")
        macro(f"cmmd{tag}", f"{np.mean([x['plain']['cmmd'] for x in s]):.3f}")
        macro(f"fid{tag}", f"{np.mean([x['plain']['fid'] for x in s]):.1f}")
        for key, label in [("edge_iou", "Edge"), ("blank_iou_re", "Blank"), ("layer_acc", "Layer"), ("salience_iou", "Sal")]:
            macro(f"spec{label}{tag}", f"{np.mean([x['structure_v8'][key + '_specific'] for x in s]):.3f}")
        macro(f"colourBA{tag}", f"{np.mean([x['style_v8']['balanced_accuracy'] for x in s]):.3f}")
    rename = {"edge_iou_specific": "Edge", "blank_iou_re_specific": "Blank", "layer_acc_specific": "Layer",
              "salience_iou_specific": "Sal", "clipscore": "Clip", "pickscore": "Pick", "hpsv2": "Hps",
              "lpips_diversity": "Div", "style_v8_correct": "Colour"}
    for base, tag in [("LoRA+ControlNet", "LoraCN"), ("ControlNet", "CN"), ("Ours w/o style ref.", "NoStyle")]:
        for _, r in tests[tests.comparison == f"Ours vs {base}"].iterrows():
            macro(f"p{rename[r.metric]}{tag}", p_cell(r.p_wilcoxon_holm))
            macro(f"d{rename[r.metric]}{tag}", f"{r.mean_diff:+.3f}")
    real = load("real_test")
    macro("kidReal", f"{real['plain']['kid'] * 1000:.1f}")
    macro("colourBAReal", f"{real['style_v8']['balanced_accuracy']:.3f}")
    ctrl = CTRL["Ours"] / "summary.json"
    if ctrl.exists():
        c = json.loads(ctrl.read_text(encoding="utf-8"))
        macro("ctrlEdgeUsed", f"{c['structure']['edge_iou_used']:.3f}")
        macro("ctrlEdgeUnused", f"{c['structure']['edge_iou_unused']:.3f}")
        macro("ctrlEdgeFrac", f"{100 * c['structure']['edge_used_gt_unused_fraction']:.1f}")
        macro("ctrlSatCorr", f"{c['style']['saturation_corr_output_reference']:.2f}")
    return "\n".join(out)


def main() -> None:
    REV.mkdir(parents=True, exist_ok=True)
    rows = [(name, [load(r) for r in runs]) for name, runs in MAIN]
    parts = {
        "fidelity": lambda: fidelity_table(rows),
        "structure": lambda: structure_table(rows),
        "quality": lambda: quality_table(rows),
        "tests_main": lambda: tests_table(STATS / "paired_tests.csv", ["LoRA+ControlNet", "ControlNet", "Token adapter", "Ours w/o style ref."]),
        "tests_all": lambda: tests_table(STATS / "paired_tests.csv"),
        "tests_ablation": lambda: tests_table(STATS / "ablation_tests.csv"),
        "ablation": ablation_table,
        "selection": selection_table,
        "controllability": controllability_table,
        "diagnosis": diagnosis_table,
    }
    out = []
    for key, fn in parts.items():
        try:
            body = fn()
            (REV / f"tab_{key}.tex").write_text(body + "\n", encoding="utf-8")  # \input by the manuscript
            out.append(f"%%% {key}\n" + body)
        except FileNotFoundError as exc:
            out.append(f"%%% {key}: missing {exc.filename}")
    (REV / "tables.tex").write_text("\n\n".join(out) + "\n", encoding="utf-8")
    (REV / "numbers.tex").write_text(numbers(rows) + "\n", encoding="utf-8")
    print((REV / "tables.tex").read_text(encoding="utf-8")[:2500])


if __name__ == "__main__":
    main()
