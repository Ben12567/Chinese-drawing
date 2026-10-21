from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except Exception:
        return None
    if math.isnan(result):
        return None
    return result


def _read_mean_std(path: Path) -> dict[str, dict[str, float | int | None]]:
    rows = _read_csv_rows(path)
    result: dict[str, dict[str, float | int | None]] = {}
    for row in rows:
        method = row["Method"]
        result[method] = {"NumRuns": int(row["NumRuns"])}
        for key, value in row.items():
            if key in {"Method", "NumRuns"}:
                continue
            result[method][key] = _to_float(value)
    return result


def _read_raw(path: Path) -> dict[str, dict[str, float | str | None]]:
    rows = _read_csv_rows(path)
    result: dict[str, dict[str, float | str | None]] = {}
    for row in rows:
        method = row["Method"]
        result[method] = {}
        for key, value in row.items():
            if key == "Method":
                continue
            parsed = _to_float(value)
            result[method][key] = parsed if parsed is not None else value
    return result


def _read_significance(path: Path) -> dict[str, dict[str, float | int | None]]:
    rows = _read_csv_rows(path)
    result: dict[str, dict[str, float | int | None]] = {}
    for row in rows:
        metric = row["metric"]
        result[metric] = {
            "n": int(row["n"]),
            "left_mean": _to_float(row["left_mean"]),
            "right_mean": _to_float(row["right_mean"]),
            "paired_t_p": _to_float(row["paired_t_p"]),
            "wilcoxon_p": _to_float(row["wilcoxon_p"]),
        }
    return result


def _pct_improvement(lower_better_baseline: float, lower_better_ours: float) -> float:
    return ((lower_better_baseline - lower_better_ours) / lower_better_baseline) * 100.0


def _pct_gain(higher_better_ours: float, higher_better_baseline: float) -> float:
    return ((higher_better_ours - higher_better_baseline) / higher_better_baseline) * 100.0


def _build_story(results_root: Path, benchmark_summary: Path | None) -> dict:
    mean_std = _read_mean_std(results_root / "main_results_mean_std.csv")
    raw = _read_raw(results_root / "main_results_raw.csv")
    significance = {
        "lora": _read_significance(results_root / "significance_ours_vs_lora.csv"),
        "controlnet": _read_significance(results_root / "significance_ours_vs_controlnet.csv"),
        "ip_adapter_only": _read_significance(results_root / "significance_ours_vs_ip_adapter_only.csv"),
    }

    ours = mean_std["paper_ours_817"]
    lora = mean_std["paper_lora_only_817"]
    controlnet = mean_std["paper_controlnet_817"]
    ip_adapter = mean_std["paper_ip_adapter_only_817"]

    story = {
        "suite_root": str(results_root),
        "recommended_story": {
            "core_claim": "The proposed method is strongest on in-domain fidelity, style consistency, and diversity for Chinese landscape painting, rather than universal semantic alignment.",
            "safe_conclusion": "The method should be presented as a domain-specialized controllable generation framework with clear gains over SDXL+LoRA and stronger overall in-domain quality than single-branch ControlNet or IP-Adapter baselines.",
            "avoid_overclaim": [
                "Do not claim universal semantic superiority over all controllable generation baselines.",
                "Do not claim stronger edge adherence than ControlNet.",
                "Do not claim human-level or artist-level artistic quality from these metrics alone.",
            ],
        },
        "main_findings": {
            "ours_vs_lora": {
                "fid_reduction_pct": _pct_improvement(float(lora["fid_mean"]), float(ours["fid_mean"])),
                "kid_reduction_pct": _pct_improvement(float(lora["kid_mean"]), float(ours["kid_mean"])),
                "lpips_gain_pct": _pct_gain(float(ours["lpips_diversity_mean"]), float(lora["lpips_diversity_mean"])),
                "style_accuracy_gain_pct": _pct_gain(float(ours["style_accuracy_mean"]), float(lora["style_accuracy_mean"])),
            },
            "ours_vs_controlnet": {
                "fid_reduction_pct": _pct_improvement(float(controlnet["fid_mean"]), float(ours["fid_mean"])),
                "kid_reduction_pct": _pct_improvement(float(controlnet["kid_mean"]), float(ours["kid_mean"])),
            },
            "ours_vs_ip_adapter_only": {
                "fid_reduction_pct": _pct_improvement(float(ip_adapter["fid_mean"]), float(ours["fid_mean"])),
                "kid_reduction_pct": _pct_improvement(float(ip_adapter["kid_mean"]), float(ours["kid_mean"])),
            },
        },
        "results_mean_std": mean_std,
        "results_raw": raw,
        "significance": significance,
    }

    if benchmark_summary and benchmark_summary.exists():
        story["benchmark_summary"] = json.loads(benchmark_summary.read_text(encoding="utf-8"))
    return story


def _render_markdown(story: dict) -> str:
    ours = story["results_mean_std"]["paper_ours_817"]
    lora = story["results_mean_std"]["paper_lora_only_817"]
    controlnet = story["results_mean_std"]["paper_controlnet_817"]
    ip_adapter = story["results_mean_std"]["paper_ip_adapter_only_817"]
    finding = story["main_findings"]["ours_vs_lora"]

    benchmark_lines = []
    benchmark = story.get("benchmark_summary", {})
    ours_b = benchmark.get("paper_ours_817", {})
    lora_b = benchmark.get("paper_lora_only_817", {})
    if ours_b and lora_b:
        benchmark_lines = [
            "",
            "## Benchmark Caveat",
            f"- TIFA subset: ours {ours_b['tifa_subset']['tifa_average']:.4f}, lora-only {lora_b['tifa_subset']['tifa_average']:.4f}",
            f"- T2I-CompBench non-spatial: ours {ours_b['t2i_compbench_non_spatial']['score_avg']:.4f}, lora-only {lora_b['t2i_compbench_non_spatial']['score_avg']:.4f}",
            "- These public compositional benchmarks do not support a claim of universal semantic superiority.",
        ]

    return "\n".join(
        [
            "# Final Experimental Story",
            "",
            "## Recommended Claim",
            story["recommended_story"]["core_claim"],
            "",
            "## Main Table Narrative",
            (
                f"- Ours vs LoRA-only: FID {ours['fid_mean']:.2f}+/-{ours['fid_std']:.2f} vs "
                f"{lora['fid_mean']:.2f}+/-{lora['fid_std']:.2f}; "
                f"KID {ours['kid_mean']:.4f}+/-{ours['kid_std']:.4f} vs "
                f"{lora['kid_mean']:.4f}+/-{lora['kid_std']:.4f}"
            ),
            (
                f"- Ours improves FID by {finding['fid_reduction_pct']:.1f}%, "
                f"KID by {finding['kid_reduction_pct']:.1f}%, "
                f"LPIPS diversity by {finding['lpips_gain_pct']:.1f}%, "
                f"and style accuracy by {finding['style_accuracy_gain_pct']:.1f}% over LoRA-only."
            ),
            (
                f"- Ours also outperforms ControlNet on FID/KID "
                f"({controlnet['fid_mean']:.2f}/{controlnet['kid_mean']:.4f}) "
                f"and IP-Adapter only on FID/KID "
                f"({ip_adapter['fid_mean']:.2f}/{ip_adapter['kid_mean']:.4f})."
            ),
            "",
            "## Logical Interpretation",
            "- The method is strongest on in-domain fidelity and style organization.",
            "- ControlNet retains an advantage on hard edge adherence; this should be stated explicitly.",
            "- IP-Adapter only is strong on preference-style metrics, but weaker on overall distribution matching than the full method.",
            "",
            "## Safe Writing Boundary",
            "- Claim superiority in Chinese landscape painting fidelity, style consistency, and diversity.",
            "- Avoid claiming universal semantic superiority or stronger edge control than ControlNet.",
            "- Avoid claiming human-level artistic judgment from objective metrics alone.",
            *benchmark_lines,
        ]
    )


def main(results_root: str, output_markdown: str, output_json: str, benchmark_summary: str | None) -> None:
    results_root_path = Path(results_root)
    benchmark_path = Path(benchmark_summary) if benchmark_summary else None
    story = _build_story(results_root_path, benchmark_path)

    output_json_path = Path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8")

    output_markdown_path = Path(output_markdown)
    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_markdown_path.write_text(_render_markdown(story), encoding="utf-8")

    print(f"Wrote final story JSON to {output_json_path}")
    print(f"Wrote final story Markdown to {output_markdown_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--benchmark-summary")
    args = parser.parse_args()
    main(
        results_root=args.results_root,
        output_markdown=args.output_markdown,
        output_json=args.output_json,
        benchmark_summary=args.benchmark_summary,
    )
