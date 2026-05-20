from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


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


def _stage_existing_seed42(cwd: Path, suite_root: Path) -> None:
    pairs = [
        (
            cwd / "outputs" / "paper_suite_v6" / "paper_ours_817" / "evaluation_multiseed",
            suite_root / "paper_ours_817_seed42" / "evaluation_multiseed",
        ),
        (
            cwd / "outputs" / "paper_suite_v6" / "paper_lora_only_817" / "evaluation_multiseed",
            suite_root / "paper_lora_only_817_seed42" / "evaluation_multiseed",
        ),
    ]
    for src, dst in pairs:
        _copy_tree_if_missing(src, dst)


def _significance(
    cwd: Path,
    left_csv: Path,
    right_csv: Path,
    output_csv: Path,
) -> None:
    _run(
        [
            "python",
            "scripts/analyze_significance.py",
            "--left-csv",
            str(left_csv),
            "--right-csv",
            str(right_csv),
            "--metrics",
            "clipscore,pickscore,hpsv2,style_correct,edge_consistency,blank_space_iou,blank_space_ssim",
            "--output-csv",
            str(output_csv),
        ],
        cwd,
    )


def main(suite_root: str, skip_existing: bool) -> None:
    cwd = Path.cwd()
    suite_root_path = Path(suite_root)
    _stage_existing_seed42(cwd, suite_root_path)

    configs = [
        "configs/experiments/paper_controlnet_817.yaml",
        "configs/experiments/paper_ip_adapter_only_817.yaml",
        "configs/experiments/paper_ours_817.yaml",
        "configs/experiments/paper_lora_only_817.yaml",
    ]
    _run(
        [
            "python",
            "scripts/run_multi_seed_suite.py",
            "--configs",
            *configs,
            "--train-seeds",
            "42,52,62",
            "--inference-seeds",
            "42,52,62",
            "--split",
            "test",
            "--suite-root",
            str(suite_root_path),
        ]
        + (["--skip-existing"] if skip_existing else []),
        cwd,
    )

    _run(
        [
            "python",
            "scripts/collect_experiment_table.py",
            "--results-root",
            str(suite_root_path),
            "--output-csv",
            str(suite_root_path / "main_results_raw.csv"),
        ],
        cwd,
    )
    _run(
        [
            "python",
            "scripts/aggregate_seed_metrics.py",
            "--results-root",
            str(suite_root_path),
            "--output-csv",
            str(suite_root_path / "main_results_mean_std.csv"),
            "--strip-seed-suffix",
        ],
        cwd,
    )

    ours_dir = suite_root_path / "paper_ours_817_seed42" / "evaluation_multiseed" / "per_sample_metrics.csv"
    comparisons = {
        "lora": suite_root_path / "paper_lora_only_817_seed42" / "evaluation_multiseed" / "per_sample_metrics.csv",
        "controlnet": suite_root_path / "paper_controlnet_817_seed42" / "evaluation_multiseed" / "per_sample_metrics.csv",
        "ip_adapter_only": suite_root_path / "paper_ip_adapter_only_817_seed42" / "evaluation_multiseed" / "per_sample_metrics.csv",
    }
    for name, csv_path in comparisons.items():
        if csv_path.exists() and ours_dir.exists():
            _significance(
                cwd,
                left_csv=ours_dir,
                right_csv=csv_path,
                output_csv=suite_root_path / f"significance_ours_vs_{name}.csv",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", default="outputs/paper_suite_strong")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    main(suite_root=args.suite_root, skip_existing=args.skip_existing)
