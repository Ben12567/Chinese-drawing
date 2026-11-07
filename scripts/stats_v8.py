"""Aggregate the v8 evaluation runs and run paired significance tests.

Same design as stats_v7.py, with the v8 metrics added: condition-specific structure adherence
(structure_metrics_v8.py: matched minus mismatched-map score per sample) and the binary colour-
treatment style metric (style_v8.py). Two test families, each Holm-corrected on its own:

  main       the reference method (all training seeds pooled) vs every other method of the spec
  ablation   the reference method's seed-42 run vs each single-seed ablation (same training seed)

Unit of analysis: the test sample (n = 124 in benchmark v7); a sample's value is its mean over inference seeds and,
for multi-seed methods, over training seeds. Paired t-test and Wilcoxon signed-rank; effect sizes:
mean difference with 95% bootstrap CI and Cohen's d_z. Set-level metrics (FID, KID, CMMD, style
balanced accuracy) are reported as mean and SD over training seeds with the per-run bootstrap CI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from stats_v7 import holm

QUALITY = ["clipscore", "pickscore", "hpsv2", "lpips_diversity"]
STRUCTURE = ["edge_iou_specific", "blank_iou_re_specific", "layer_acc_specific", "salience_iou_specific"]
STRUCTURE_RAW = ["edge_iou", "blank_iou_re", "layer_acc", "salience_iou"]
PER_SAMPLE = QUALITY + STRUCTURE + ["style_v8_correct"]
SET_LEVEL = ["fid", "kid", "cmmd"]


def per_sample(d: Path) -> pd.DataFrame:
    base = pd.read_csv(d / "per_sample_metrics.csv", index_col=0)[QUALITY]
    struct = pd.read_csv(d / "structure_v8_per_sample.csv").groupby("sample_id")[STRUCTURE + STRUCTURE_RAW].mean()
    style = pd.read_csv(d / "style_v8_per_image.csv").groupby("sample_id")["correct"].mean().rename("style_v8_correct")
    return base.join(struct, how="inner").join(style, how="inner")


def load_method(dirs: list[str]) -> tuple[list[dict], pd.DataFrame]:
    runs = []
    for d in map(Path, dirs):
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        s["structure_v8"] = json.loads((d / "structure_v8_summary.json").read_text(encoding="utf-8"))
        s["style_v8"] = json.loads((d / "style_v8_summary.json").read_text(encoding="utf-8"))
        runs.append(s)
    frames = [per_sample(Path(d)) for d in dirs]
    return runs, sum(frames) / len(frames)


def summary_rows(methods: dict, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, seed_rows = [], []
    for name, (runs, pooled) in methods.items():
        row = {"method": name, "n_train_runs": len(runs)}
        for key in SET_LEVEL:
            vals = [s["plain"][key] for s in runs]
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            row[f"{key}_ci_lo"], row[f"{key}_ci_hi"] = runs[0]["plain"][f"{key}_ci"]
            for variant in ["seal_inpainted", "centre_crop"]:
                if variant in runs[0]:
                    row[f"{key}_{variant}_mean"] = float(np.mean([s[variant][key] for s in runs]))
        for key in ["balanced_accuracy", "macro_f1"]:
            vals = [s["style_v8"][key] for s in runs]
            row[f"style_v8_{key}_mean"] = float(np.mean(vals))
            row[f"style_v8_{key}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        for key in STRUCTURE + STRUCTURE_RAW:
            vals = [s["structure_v8"][key] for s in runs]
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        for key in QUALITY:
            row[f"{key}_mean"] = float(pooled[key].mean())
        rows.append(row)
        for d, s in zip(spec[name], runs):
            seed_rows.append({"method": name, "run": Path(d).name, **{k: s["plain"][k] for k in SET_LEVEL},
                              "style_v8_balanced_accuracy": s["style_v8"]["balanced_accuracy"],
                              **{k: s["structure_v8"][k] for k in STRUCTURE}})
    return pd.DataFrame(rows), pd.DataFrame(seed_rows)


def paired_tests(ref_name: str, ref: pd.DataFrame, others: dict[str, pd.DataFrame], rng, n_boot: int) -> pd.DataFrame:
    tests = []
    for name, other in others.items():
        shared = ref.index.intersection(other.index)
        for metric in PER_SAMPLE:
            a = ref.loc[shared, metric].astype(float)
            b = other.loc[shared, metric].astype(float)
            mask = a.notna() & b.notna()
            a, b = a[mask].to_numpy(), b[mask].to_numpy()
            diff = a - b
            if len(diff) < 3 or np.allclose(diff, 0):
                t_p = w_p = 1.0
            else:
                t_p = float(ttest_rel(a, b).pvalue)
                w_p = float(wilcoxon(a, b).pvalue)
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
            sd = diff.std(ddof=1)
            tests.append({
                "comparison": f"{ref_name} vs {name}", "metric": metric, "n": int(len(diff)),
                "mean_reference": float(a.mean()), "mean_baseline": float(b.mean()),
                "mean_diff": float(diff.mean()), "diff_ci_lo": float(np.percentile(boots, 2.5)),
                "diff_ci_hi": float(np.percentile(boots, 97.5)), "cohens_dz": float(diff.mean() / sd) if sd > 0 else np.nan,
                "p_t": t_p, "p_wilcoxon": w_p,
            })
    df = pd.DataFrame(tests)
    df["p_t_holm"] = holm(df["p_t"].tolist())
    df["p_wilcoxon_holm"] = holm(df["p_wilcoxon"].tolist())
    return df


def main(args: argparse.Namespace) -> None:
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    ablations = json.loads(Path(args.ablations).read_text(encoding="utf-8")) if args.ablations else {}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    methods = {name: load_method(dirs) for name, dirs in spec.items()}
    table, per_run = summary_rows(methods, spec)
    table.to_csv(out / "main_table.csv", index=False)
    per_run.to_csv(out / "per_run_table.csv", index=False)
    ref = methods[args.reference][1]
    main_tests = paired_tests(args.reference, ref, {n: m[1] for n, m in methods.items() if n != args.reference}, rng, args.n_boot)
    main_tests.to_csv(out / "paired_tests.csv", index=False)

    if ablations:
        abl = {name: load_method(dirs) for name, dirs in ablations.items()}
        abl_table, _ = summary_rows(abl, ablations)
        abl_table.to_csv(out / "ablation_table.csv", index=False)
        ref_name = next(iter(ablations))  # the first entry is the seed-matched full model
        abl_tests = paired_tests(ref_name, abl[ref_name][1], {n: m[1] for n, m in abl.items() if n != ref_name}, rng, args.n_boot)
        abl_tests.to_csv(out / "ablation_tests.csv", index=False)
        print(abl_table[["method", "kid_mean", "cmmd_mean", "fid_mean"] + [f"{k}_mean" for k in STRUCTURE]].to_string())

    cols = ["method", "kid_mean", "cmmd_mean", "fid_mean", "style_v8_balanced_accuracy_mean"] + [f"{k}_mean" for k in STRUCTURE]
    print(table[cols].to_string())
    print(main_tests[["comparison", "metric", "mean_diff", "diff_ci_lo", "diff_ci_hi", "p_wilcoxon_holm"]].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--ablations", default=None, help="JSON {name: [eval dir]}; first entry = seed-matched full model")
    parser.add_argument("--reference", default="Ours")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    main(parser.parse_args())
