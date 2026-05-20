from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import psutil


def _run(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command), flush=True)
    env = os.environ.copy()
    src_path = str(cwd / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    env.setdefault("HF_HOME", str(cwd / ".hf"))
    subprocess.run(command, cwd=str(cwd), check=True, env=env)


def _latest_checkpoint(output_root: Path) -> Path:
    checkpoints = sorted(output_root.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {output_root}")
    return checkpoints[-1]


def _wait_for_pid(pid: int) -> None:
    while psutil.pid_exists(pid):
        time.sleep(30)


def _evaluate(cwd: Path, config_path: str, manifest_path: str, output_root: Path) -> None:
    checkpoint = _latest_checkpoint(output_root)
    pred_dir = output_root / "predictions_multiseed"
    eval_dir = output_root / "evaluation_multiseed"
    _run(
        [
            "python",
            "scripts/generate_eval_set.py",
            "--config",
            config_path,
            "--manifest-path",
            manifest_path,
            "--split",
            "test",
            "--output-dir",
            str(pred_dir),
            "--prompt-mode",
            "structured",
            "--seeds",
            "42,52,62",
            "--checkpoint-dir",
            str(checkpoint),
            "--skip-existing",
        ],
        cwd,
    )
    _run(
        [
            "python",
            "scripts/evaluate_experiments.py",
            "--manifest-path",
            manifest_path,
            "--prediction-dir",
            str(pred_dir),
            "--split",
            "test",
            "--prompt-mode",
            "structured",
            "--output-dir",
            str(eval_dir),
        ],
        cwd,
    )


def main(wait_pid: int, ours_config: str, lora_config: str, manifest_path: str) -> None:
    cwd = Path.cwd()
    ours_root = Path("outputs/paper_suite_v6/paper_ours_817")
    lora_root = Path("outputs/paper_suite_v6/paper_lora_only_817")

    _wait_for_pid(wait_pid)
    _evaluate(cwd, ours_config, manifest_path, ours_root)
    _run(["python", "-m", "clpgen.training.train_sdxl_lora", "--config", lora_config], cwd)
    _evaluate(cwd, lora_config, manifest_path, lora_root)
    _run(
        [
            "python",
            "scripts/collect_experiment_table.py",
            "--results-root",
            "outputs/paper_suite_v6",
            "--output-csv",
            "outputs/paper_suite_v6/main_results.csv",
        ],
        cwd,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--ours-config", default="configs/experiments/paper_ours_817.yaml")
    parser.add_argument("--lora-config", default="configs/experiments/paper_lora_only_817.yaml")
    parser.add_argument("--manifest-path", default="data/processed/clp4k_v6_union_en/manifest.jsonl")
    args = parser.parse_args()
    main(
        wait_pid=args.wait_pid,
        ours_config=args.ours_config,
        lora_config=args.lora_config,
        manifest_path=args.manifest_path,
    )
