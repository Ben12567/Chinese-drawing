from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _latest_checkpoint(output_root: Path) -> Path | None:
    checkpoints = sorted(output_root.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    return checkpoints[-1] if checkpoints else None


def _run(command: list[str], cwd: Path, retries: int = 1, retry_delay_s: int = 15) -> None:
    print("Running:", " ".join(command))
    env = os.environ.copy()
    src_path = str(cwd / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    env.setdefault("HF_HOME", str(cwd / ".hf"))
    retryable_codes = {1, 3221225477, -1073741819}
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(command, cwd=str(cwd), check=True, env=env)
            return
        except subprocess.CalledProcessError as exc:
            if attempt >= retries or exc.returncode not in retryable_codes:
                raise
            print(
                f"Command failed with exit code {exc.returncode}; retrying "
                f"({attempt}/{retries - 1}) after {retry_delay_s}s...",
                flush=True,
            )
            time.sleep(retry_delay_s)


def _seeded_config(base_config: dict, seed: int, output_root: Path) -> dict:
    config = yaml.safe_load(yaml.safe_dump(base_config))
    config.setdefault("project", {})
    config["project"]["seed"] = seed
    config["project"]["output_root"] = output_root.as_posix()
    return config


def main(
    config_paths: list[str],
    train_seeds: list[int],
    inference_seeds: list[int],
    split: str,
    limit: int | None,
    suite_root: str,
    skip_existing: bool,
) -> None:
    cwd = Path.cwd()
    env_pythonpath = cwd / "src"
    print(f"Expected PYTHONPATH root: {env_pythonpath}")
    suite_root_path = Path(suite_root)
    generated_config_root = suite_root_path / "generated_configs"

    for config_path_text in config_paths:
        config_path = Path(config_path_text)
        base_config = _load_yaml(config_path)
        base_name = config_path.stem
        manifest_path = base_config["dataset"]["manifest_path"]
        prompt_mode = base_config["dataset"]["prompt_mode_eval"]
        trainable = base_config["training"].get("max_train_steps", 0) > 1

        if not trainable:
            output_root = suite_root_path / f"{base_name}_seed{train_seeds[0]}"
            config = _seeded_config(base_config, train_seeds[0], output_root)
            config_file = generated_config_root / f"{base_name}_seed{train_seeds[0]}.yaml"
            _write_yaml(config_file, config)

            pred_dir = output_root / "predictions_multiseed"
            eval_dir = output_root / "evaluation_multiseed"
            if not (skip_existing and (eval_dir / "summary_metrics.json").exists()):
                _run(
                    [
                        "python",
                        "scripts/generate_eval_set.py",
                        "--config",
                        str(config_file),
                        "--manifest-path",
                        manifest_path,
                        "--split",
                        split,
                        "--output-dir",
                        str(pred_dir),
                        "--prompt-mode",
                        prompt_mode,
                        "--seeds",
                        ",".join(str(seed) for seed in inference_seeds),
                    ]
                    + (["--limit", str(limit)] if limit is not None else [])
                    + ["--skip-existing"],
                    cwd=cwd,
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
                        split,
                        "--prompt-mode",
                        prompt_mode,
                        "--output-dir",
                        str(eval_dir),
                    ],
                    cwd=cwd,
                    retries=3,
                )
            continue

        for seed in train_seeds:
            output_root = suite_root_path / f"{base_name}_seed{seed}"
            config = _seeded_config(base_config, seed, output_root)
            config_file = generated_config_root / f"{base_name}_seed{seed}.yaml"
            _write_yaml(config_file, config)

            pred_dir = output_root / "predictions_multiseed"
            eval_dir = output_root / "evaluation_multiseed"
            if skip_existing and (eval_dir / "summary_metrics.json").exists():
                continue

            checkpoint = _latest_checkpoint(output_root)
            if not (skip_existing and checkpoint is not None):
                _run(
                    [
                        "python",
                        "-m",
                        "clpgen.training.train_sdxl_lora",
                        "--config",
                        str(config_file),
                    ],
                    cwd=cwd,
                )
                checkpoint = _latest_checkpoint(output_root)
            if checkpoint is None:
                raise FileNotFoundError(f"No checkpoint found for {config_file}")

            _run(
                [
                    "python",
                    "scripts/generate_eval_set.py",
                    "--config",
                    str(config_file),
                    "--manifest-path",
                    manifest_path,
                    "--split",
                    split,
                    "--output-dir",
                    str(pred_dir),
                    "--prompt-mode",
                    prompt_mode,
                    "--seeds",
                    ",".join(str(item) for item in inference_seeds),
                    "--checkpoint-dir",
                    str(checkpoint),
                ]
                + (["--limit", str(limit)] if limit is not None else [])
                + ["--skip-existing"],
                cwd=cwd,
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
                    split,
                    "--prompt-mode",
                    prompt_mode,
                    "--output-dir",
                    str(eval_dir),
                ],
                cwd=cwd,
                retries=3,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--train-seeds", default="42,52,62")
    parser.add_argument("--inference-seeds", default="42,52,62")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--suite-root", default="outputs/paper_suite")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    main(
        config_paths=args.configs,
        train_seeds=[int(token.strip()) for token in args.train_seeds.split(",") if token.strip()],
        inference_seeds=[int(token.strip()) for token in args.inference_seeds.split(",") if token.strip()],
        split=args.split,
        limit=args.limit,
        suite_root=args.suite_root,
        skip_existing=args.skip_existing,
    )
