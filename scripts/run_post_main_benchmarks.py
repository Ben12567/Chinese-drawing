from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def _run(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command), flush=True)
    env = os.environ.copy()
    src_path = str(cwd / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    env.setdefault("HF_HOME", str(cwd / ".hf"))
    subprocess.run(command, cwd=str(cwd), check=True, env=env)


def _wait_for_file(path: Path, poll_seconds: int) -> None:
    while not path.exists():
        time.sleep(poll_seconds)


def _latest_checkpoint(output_root: Path) -> Path:
    checkpoints = sorted(output_root.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {output_root}")
    return checkpoints[-1]


def _subset_tifa_inputs(
    text_inputs_path: Path,
    question_answers_path: Path,
    output_root: Path,
    limit: int,
) -> tuple[Path, Path]:
    text_inputs = json.loads(text_inputs_path.read_text(encoding="utf-8"))
    selected = text_inputs[:limit]
    selected_ids = {str(row["id"]) for row in selected}
    question_answers = json.loads(question_answers_path.read_text(encoding="utf-8"))
    selected_questions = [row for row in question_answers if str(row["id"]) in selected_ids]

    output_root.mkdir(parents=True, exist_ok=True)
    text_subset_path = output_root / "tifa_text_inputs_subset.json"
    qa_subset_path = output_root / "tifa_question_answers_subset.json"
    text_subset_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    qa_subset_path.write_text(json.dumps(selected_questions, ensure_ascii=False, indent=2), encoding="utf-8")
    return text_subset_path, qa_subset_path


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _generate_tifa(
    cwd: Path,
    method_name: str,
    config_path: str,
    checkpoint_dir: Path,
    benchmark_root: Path,
    tifa_limit: int,
) -> dict:
    tifa_root = benchmark_root / method_name / "tifa_subset"
    subset_root = tifa_root / "subset_spec"
    text_subset_path, qa_subset_path = _subset_tifa_inputs(
        text_inputs_path=cwd / ".tmp" / "tifa" / "tifa_v1.0" / "tifa_v1.0_text_inputs.json",
        question_answers_path=cwd / ".tmp" / "tifa" / "tifa_v1.0" / "tifa_v1.0_question_answers.json",
        output_root=subset_root,
        limit=tifa_limit,
    )
    generation_root = tifa_root / "generated"
    result_path = tifa_root / "tifa_compatible_result.json"
    _run(
        [
            "python",
            "scripts/generate_official_benchmark_images.py",
            "--config",
            config_path,
            "--benchmark",
            "tifa",
            "--prompt-file",
            str(text_subset_path),
            "--output-dir",
            str(generation_root),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--seed",
            "42",
        ],
        cwd,
    )
    _run(
        [
            "python",
            "scripts/run_tifa_compatible.py",
            "--question-answer-path",
            str(qa_subset_path),
            "--id2img-path",
            str(generation_root / "id2img.json"),
            "--output-path",
            str(result_path),
        ],
        cwd,
    )
    summary = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "status": "completed",
        "benchmark": "tifa_subset",
        "limit": tifa_limit,
        "tifa_average": summary.get("tifa_average"),
        "tifa_stdev": summary.get("tifa_stdev"),
        "result_path": str(result_path),
    }


def _generate_t2i_compbench_nonspatial(
    cwd: Path,
    method_name: str,
    config_path: str,
    checkpoint_dir: Path,
    benchmark_root: Path,
    limit: int,
) -> dict:
    bench_root = benchmark_root / method_name / "t2i_compbench_non_spatial"
    prompt_file = cwd / ".tmp" / "t2i_compbench" / "examples" / "dataset" / "non_spatial_val.txt"
    generation_root = bench_root
    result_path = bench_root / "annotation_clip" / "score_avg.txt"
    _run(
        [
            "python",
            "scripts/generate_official_benchmark_images.py",
            "--config",
            config_path,
            "--benchmark",
            "t2i_compbench",
            "--prompt-file",
            str(prompt_file),
            "--output-dir",
            str(generation_root / "samples"),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--seed",
            "42",
            "--limit",
            str(limit),
        ],
        cwd,
    )
    _run(
        [
            "python",
            "scripts/run_t2i_compbench_official.py",
            "--repo-path",
            ".tmp/t2i_compbench",
            "--task",
            "non_spatial",
            "--outpath",
            str(generation_root),
        ],
        cwd,
    )
    score_line = result_path.read_text(encoding="utf-8").strip() if result_path.exists() else ""
    score_value = None
    if ":" in score_line:
        try:
            score_value = float(score_line.split(":", 1)[1])
        except ValueError:
            score_value = None
    return {
        "status": "completed",
        "benchmark": "t2i_compbench_non_spatial",
        "limit": limit,
        "score_avg": score_value,
        "result_path": str(result_path),
    }


def _probe_geneval() -> dict:
    try:
        import mmdet  # type: ignore  # noqa: F401
        import mmcv  # type: ignore  # noqa: F401
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": "mmdet/mmcv unavailable in current environment",
            "detail": str(exc),
        }
    return {"status": "ready"}


def _run_method_suite(
    cwd: Path,
    method_name: str,
    config_path: str,
    output_root: Path,
    benchmark_root: Path,
    tifa_limit: int,
    t2i_limit: int,
) -> dict:
    checkpoint_dir = _latest_checkpoint(output_root)
    suite_summary: dict[str, object] = {
        "method": method_name,
        "checkpoint_dir": str(checkpoint_dir),
    }
    try:
        suite_summary["tifa_subset"] = _generate_tifa(
            cwd=cwd,
            method_name=method_name,
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            benchmark_root=benchmark_root,
            tifa_limit=tifa_limit,
        )
    except Exception as exc:
        suite_summary["tifa_subset"] = {"status": "failed", "detail": str(exc)}
    try:
        suite_summary["t2i_compbench_non_spatial"] = _generate_t2i_compbench_nonspatial(
            cwd=cwd,
            method_name=method_name,
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            benchmark_root=benchmark_root,
            limit=t2i_limit,
        )
    except Exception as exc:
        suite_summary["t2i_compbench_non_spatial"] = {"status": "failed", "detail": str(exc)}
    suite_summary["geneval_official"] = _probe_geneval()
    return suite_summary


def main(
    results_csv: str,
    benchmark_root: str,
    ours_config: str,
    lora_config: str,
    tifa_limit: int,
    t2i_limit: int,
    poll_seconds: int,
) -> None:
    cwd = Path.cwd()
    results_path = Path(results_csv)
    benchmark_root_path = Path(benchmark_root)
    _wait_for_file(results_path, poll_seconds=poll_seconds)

    suite_results = {
        "paper_ours_817": _run_method_suite(
            cwd=cwd,
            method_name="paper_ours_817",
            config_path=ours_config,
            output_root=Path("outputs/paper_suite_v6/paper_ours_817"),
            benchmark_root=benchmark_root_path,
            tifa_limit=tifa_limit,
            t2i_limit=t2i_limit,
        ),
        "paper_lora_only_817": _run_method_suite(
            cwd=cwd,
            method_name="paper_lora_only_817",
            config_path=lora_config,
            output_root=Path("outputs/paper_suite_v6/paper_lora_only_817"),
            benchmark_root=benchmark_root_path,
            tifa_limit=tifa_limit,
            t2i_limit=t2i_limit,
        ),
    }
    _write_status(benchmark_root_path / "benchmark_summary.json", suite_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", default="outputs/paper_suite_v6/main_results.csv")
    parser.add_argument("--benchmark-root", default="outputs/benchmarks/paper_suite_v6")
    parser.add_argument("--ours-config", default="configs/experiments/paper_ours_817.yaml")
    parser.add_argument("--lora-config", default="configs/experiments/paper_lora_only_817.yaml")
    parser.add_argument("--tifa-limit", type=int, default=256)
    parser.add_argument("--t2i-limit", type=int, default=300)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    main(
        results_csv=args.results_csv,
        benchmark_root=args.benchmark_root,
        ours_config=args.ours_config,
        lora_config=args.lora_config,
        tifa_limit=args.tifa_limit,
        t2i_limit=args.t2i_limit,
        poll_seconds=args.poll_seconds,
    )
