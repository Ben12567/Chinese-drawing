from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import yaml


def _run(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command), flush=True)
    env = os.environ.copy()
    src_path = str(cwd / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    env.setdefault("HF_HOME", str(cwd / ".hf"))
    subprocess.run(command, cwd=str(cwd), check=True, env=env)


def _copy_tree_if_missing(src: Path, dst: Path) -> None:
    if dst.exists() or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _stage_existing(cwd: Path, suite_root: Path, entries: list[dict]) -> None:
    for entry in entries:
        src = cwd / entry["from"]
        dst = suite_root / entry["to"]
        _copy_tree_if_missing(src, dst)


def _run_significance(cwd: Path, suite_root: Path, block: dict) -> None:
    left_csv = suite_root / block["left"]
    metrics = block["metrics"]
    for name, rel_path in sorted(block["comparisons"].items()):
        right_csv = suite_root / rel_path
        if not (left_csv.exists() and right_csv.exists()):
            continue
        _run(
            [
                "python",
                "scripts/analyze_significance.py",
                "--left-csv",
                str(left_csv),
                "--right-csv",
                str(right_csv),
                "--metrics",
                metrics,
                "--output-csv",
                str(suite_root / f"significance_ours_vs_{name}.csv"),
            ],
            cwd,
        )


def main(config_path: str, skip_existing: bool) -> None:
    cwd = Path.cwd()
    config = _load_yaml(Path(config_path))
    suite_root = cwd / config["suite_root"]
    suite_root.mkdir(parents=True, exist_ok=True)

    stage_existing = config.get("stage_existing", [])
    if stage_existing:
        _stage_existing(cwd, suite_root, stage_existing)

    _run(
        [
            "python",
            "scripts/run_multi_seed_suite.py",
            "--configs",
            *config["configs"],
            "--train-seeds",
            ",".join(str(seed) for seed in config["train_seeds"]),
            "--inference-seeds",
            ",".join(str(seed) for seed in config["inference_seeds"]),
            "--split",
            config.get("split", "test"),
            "--suite-root",
            str(suite_root),
        ]
        + (["--skip-existing"] if skip_existing else []),
        cwd,
    )

    _run(
        [
            "python",
            "scripts/collect_experiment_table.py",
            "--results-root",
            str(suite_root),
            "--output-csv",
            str(suite_root / "main_results_raw.csv"),
        ],
        cwd,
    )
    _run(
        [
            "python",
            "scripts/aggregate_seed_metrics.py",
            "--results-root",
            str(suite_root),
            "--output-csv",
            str(suite_root / "main_results_mean_std.csv"),
            "--strip-seed-suffix",
        ],
        cwd,
    )

    significance = config.get("significance")
    if significance:
        _run_significance(cwd, suite_root, significance)

    _run(
        [
            "python",
            "scripts/package_paper_results.py",
            "--results-root",
            str(suite_root),
            "--output-markdown",
            str(suite_root / "final_results_story.md"),
            "--output-json",
            str(suite_root / "final_results_story.json"),
            "--benchmark-summary",
            "outputs/benchmarks/paper_suite_v6/benchmark_summary.json",
        ],
        cwd,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/paper_suite_strong.yaml",
        help="YAML definition for the final paper experiment suite.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    main(config_path=args.config, skip_existing=args.skip_existing)
