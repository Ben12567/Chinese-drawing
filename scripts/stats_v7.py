"""Aggregate v7 evaluation runs and run paired significance tests.

Input: a JSON file mapping method name -> list of evaluation directories (one per training
seed; single-run methods have one). Each directory contains summary.json and
per_sample_metrics.csv written by evaluate_v7.py.

Unit of analysis for paired tests: the test sample (n = 83). For each method, a sample's value
is its mean over inference seeds and, for multi-seed methods, over training seeds. Tests:
paired t-test and Wilcoxon signed-rank; Holm correction over all tests reported; effect sizes:
mean difference with 95% bootstrap CI (10,000 resamples of samples) and Cohen's d_z.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

PER_SAMPLE = ["clipscore", "pickscore", "hpsv2", "style_correct", "edge_iou", "blank_iou", "lpips_diversity"]


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvals[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def load_method(dirs: list[str]) -> tuple[list[dict], pd.DataFrame]:
    summaries = [json.loads((Path(d) / "summary.json").read_text(encoding="utf-8")) for d in dirs]
    frames = [pd.read_csv(Path(d) / "per_sample_metrics.csv", index_col=0) for d in dirs]
    pooled = sum(frames) / len(frames)
    return summaries, pooled


def main(spec_path: str, reference: str, out_dir: str, n_boot: int) -> None:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = {name: load_method(dirs) for name, dirs in spec.items()}

    rows = []
    for name, (summaries, _) in methods.items():
        row = {"method": name, "n_train_runs": len(summaries)}
        for key in ["fid", "kid", "cmmd"]:
            vals = [s["plain"][key] for s in summaries]
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            row[f"{key}_ci_lo"], row[f"{key}_ci_hi"] = summaries[0]["plain"][f"{key}_ci"]
            for variant in ["seal_inpainted", "centre_crop"]:
                if variant in summaries[0]:
                    row[f"{key}_{variant}_mean"] = float(np.mean([s[variant][key] for s in summaries]))
        for key in PER_SAMPLE:
            vals = [s[key] for s in summaries]
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        row["style_macro_f1_mean"] = float(np.mean([s["style"]["macro_f1"] for s in summaries]))
        row["style_macro_f1_sd"] = float(np.std([s["style"]["macro_f1"] for s in summaries], ddof=1)) if len(summaries) > 1 else np.nan
        row["n_distinct_predicted_styles"] = int(np.mean([s["style"]["n_distinct_predicted"] for s in summaries]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "main_table.csv", index=False)

    # Per-training-seed values (training-seed variance).
    seed_rows = []
    for name, dirs in spec.items():
        for d, s in zip(dirs, methods[name][0]):
            seed_rows.append({"method": name, "run": Path(d).name, "fid": s["plain"]["fid"], "kid": s["plain"]["kid"],
                              "cmmd": s["plain"]["cmmd"], **{k: s[k] for k in PER_SAMPLE}, "style_macro_f1": s["style"]["macro_f1"]})
    pd.DataFrame(seed_rows).to_csv(out / "per_run_table.csv", index=False)

    # Paired tests: reference method vs every other method.
    rng = np.random.default_rng(0)
    _, ref = methods[reference]
    tests = []
    for name, (_, other) in methods.items():
        if name == reference:
            continue
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
                w_p = float(wilcoxon(a, b).pvalue) if np.any(diff != 0) else 1.0
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
            sd = diff.std(ddof=1)
            tests.append({
                "comparison": f"{reference} vs {name}", "metric": metric, "n": int(len(diff)),
                "mean_reference": float(a.mean()), "mean_baseline": float(b.mean()),
                "mean_diff": float(diff.mean()), "diff_ci_lo": float(np.percentile(boots, 2.5)),
                "diff_ci_hi": float(np.percentile(boots, 97.5)), "cohens_dz": float(diff.mean() / sd) if sd > 0 else np.nan,
                "p_t": t_p, "p_wilcoxon": w_p,
            })
    tests_df = pd.DataFrame(tests)
    tests_df["p_t_holm"] = holm(tests_df["p_t"].tolist())
    tests_df["p_wilcoxon_holm"] = holm(tests_df["p_wilcoxon"].tolist())
    tests_df.to_csv(out / "paired_tests.csv", index=False)
    print(pd.DataFrame(rows)[["method", "fid_mean", "kid_mean", "cmmd_mean", "style_correct_mean", "edge_iou_mean", "blank_iou_mean"]].to_string())
    print(tests_df[["comparison", "metric", "mean_diff", "diff_ci_lo", "diff_ci_hi", "p_t_holm", "p_wilcoxon_holm"]].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--reference", default="Ours")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    main(args.spec, args.reference, args.out_dir, args.n_boot)
