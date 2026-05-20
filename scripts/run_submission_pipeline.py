from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _run(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command))
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


def _evaluate_config(cwd: Path, config_path: str, output_root: Path, split: str, prompt_mode: str, seeds: str) -> None:
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
            str(Path("data/processed/clp4k_v6_union_en/manifest.jsonl")),
            "--split",
            split,
            "--output-dir",
            str(pred_dir),
            "--prompt-mode",
            prompt_mode,
            "--seeds",
            seeds,
            "--checkpoint-dir",
            str(checkpoint),
            "--skip-existing",
        ],
        cwd=cwd,
    )
    _run(
        [
            "python",
            "scripts/evaluate_experiments.py",
            "--manifest-path",
            str(Path("data/processed/clp4k_v6_union_en/manifest.jsonl")),
            "--prediction-dir",
            str(pred_dir),
            "--split",
            split,
            "--prompt-mode",
            prompt_mode,
            "--output-dir",
            str(eval_dir),
        ],
        cwd=cwd,
    )


def main(ours_config: str, lora_config: str, split: str, prompt_mode: str, seeds: str, results_root: str) -> None:
    cwd = Path.cwd()
    ours_output_root = Path("outputs/paper_suite_v6/paper_ours_817")
    lora_output_root = Path("outputs/paper_suite_v6/paper_lora_only_817")

    _run(["python", "-m", "clpgen.training.train_sdxl_lora", "--config", ours_config], cwd=cwd)
    _evaluate_config(cwd, ours_config, ours_output_root, split, prompt_mode, seeds)

    _run(["python", "-m", "clpgen.training.train_sdxl_lora", "--config", lora_config], cwd=cwd)
    _evaluate_config(cwd, lora_config, lora_output_root, split, prompt_mode, seeds)

    _run(
        [
            "python",
            "scripts/collect_experiment_table.py",
            "--results-root",
            results_root,
            "--output-csv",
            str(Path(results_root) / "main_results.csv"),
        ],
        cwd=cwd,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-config", default="configs/experiments/paper_ours_817.yaml")
    parser.add_argument("--lora-config", default="configs/experiments/paper_lora_only_817.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--prompt-mode", default="structured")
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--results-root", default="outputs/paper_suite_v6")
    args = parser.parse_args()
    main(
        ours_config=args.ours_config,
        lora_config=args.lora_config,
        split=args.split,
        prompt_mode=args.prompt_mode,
        seeds=args.seeds,
        results_root=args.results_root,
    )
