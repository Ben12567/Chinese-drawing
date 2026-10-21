"""Run the complete v7 revision protocol (training, generation, evaluation), resumably.

Stages are ordered so that the seed-42 main comparison finishes first. Every step is skipped
when its output already exists, so the script can be stopped and restarted at any time.

  python scripts/run_revision_suite.py --stage all
  python scripts/run_revision_suite.py --stage main42      # only the seed-42 main comparison
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

PY = sys.executable
MANIFEST = "data/processed/clp_v7/manifest.jsonl"
CLASSIFIER = "outputs/revision/style_classifier.pkl"
GEN = Path("outputs/revision/gen")
EVAL = Path("outputs/revision/eval")
TRAIN = Path("outputs/revision/train")
CACHE = "outputs/revision/cache"
SEEDS = [42, 52, 62]
INFER = "42,52,62"
LOG = Path("logs/revision_suite.log")


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    log("RUN " + " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    log(f"DONE in {(time.time() - t0) / 60:.1f} min")


def train(config: str, seed: int) -> Path:
    name = f"{config}_s{seed}"
    out = TRAIN / name
    cfg = yaml.safe_load(Path(f"configs/revision/{config}.yaml").read_text(encoding="utf-8"))
    steps = cfg["training"]["max_train_steps"]
    final = out / f"checkpoint-{steps}"
    if final.exists():
        return final
    cfg["project"]["seed"] = seed
    cfg["project"]["output_root"] = out.as_posix()
    gen_cfg = Path(f"outputs/revision/configs/{name}.yaml")
    gen_cfg.parent.mkdir(parents=True, exist_ok=True)
    gen_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    run([PY, "-m", "clpgen.training.train_sdxl_lora", "--config", str(gen_cfg)])
    return final


def generate(name: str, config: str, ckpt: Path | None, seeds: str = INFER, seed_mode: str = "per_sample",
             prompt_mode: str = "structured", limit: int | None = None) -> Path:
    out = GEN / name
    marker = out / ".complete"
    if marker.exists():
        return out
    cmd = [PY, "scripts/generate_v7.py", "--config", f"configs/revision/{config}.yaml", "--manifest-path", MANIFEST,
           "--output-dir", str(out), "--seeds", seeds, "--seed-mode", seed_mode, "--prompt-mode", prompt_mode, "--skip-existing"]
    if ckpt is not None:
        cmd += ["--checkpoint-dir", str(ckpt)]
    if limit:
        cmd += ["--limit", str(limit)]
    run(cmd)
    marker.write_text("ok")
    return out


def evaluate(name: str, sensitivity: bool = False, n_boot: int = 1000) -> Path:
    out = EVAL / name
    if (out / "summary.json").exists():
        return out
    cmd = [PY, "scripts/evaluate_v7.py", "--manifest-path", MANIFEST, "--prediction-dir", str(GEN / name),
           "--output-dir", str(out), "--cache-dir", CACHE, "--style-classifier", CLASSIFIER, "--n-boot", str(n_boot)]
    if sensitivity:
        cmd.append("--sensitivity")
    run(cmd)
    return out


def intersample(name: str) -> None:
    out = EVAL / name / "intersample.json"
    if not out.exists():
        run([PY, "scripts/diag_intersample.py", "--pred-dir", str(GEN / name), "--seed", "42", "--out", str(out)])


def stage_setup() -> None:
    if not Path(CLASSIFIER).exists():
        run([PY, "scripts/train_style_classifier.py", "--manifest-path", MANIFEST, "--out", CLASSIFIER,
             "--report", "outputs/revision/style_classifier_report.json"])
    if not (EVAL / "real_test" / "summary.json").exists():
        run([PY, "scripts/evaluate_v7.py", "--manifest-path", MANIFEST, "--real-as-predictions", "--output-dir",
             str(EVAL / "real_test"), "--cache-dir", CACHE, "--style-classifier", CLASSIFIER, "--sensitivity"])


def stage_main(seed: int) -> None:
    ours = train("ours", seed)
    lora = train("lora_only", seed)
    for name, cfg, ckpt in [
        (f"ours_s{seed}", "ours", ours),
        (f"lora_only_s{seed}", "lora_only", lora),
        (f"lora_ip_adapter_s{seed}", "lora_ip_adapter", lora),
        (f"lora_controlnet_s{seed}", "lora_controlnet", lora),
        (f"ours_no_style_s{seed}", "ours_no_style", ours),
    ]:
        generate(name, cfg, ckpt)
        evaluate(name, sensitivity=True)
        intersample(name)
    if seed == 42:
        for name in ["controlnet", "ip_adapter_only", "t2i_adapter", "sdxl_base"]:
            generate(name, name, None)
            evaluate(name, sensitivity=True)
            intersample(name)


def stage_ablation() -> None:
    for cfg in ["ch_lineart", "ch_depth", "ch_blank", "ch_saliency", "ch_lineart_depth", "ch_no_saliency"]:
        ckpt = train(cfg, 42)
        generate(f"{cfg}_s42", cfg, ckpt)
        evaluate(f"{cfg}_s42")


def stage_diagnostics() -> None:
    lora42 = TRAIN / "lora_only_s42" / "checkpoint-500"
    # Prompt/seed conditions of the v6 evaluation reproduced with the v7 LoRA-only model.
    for prompt_mode, seed_mode in [("v6template", "shared"), ("v6template", "per_sample"), ("structured", "shared")]:
        name = f"diag_lora42_{prompt_mode}_{seed_mode}"
        generate(name, "lora_only", lora42, seeds="42", seed_mode=seed_mode, prompt_mode=prompt_mode)
        evaluate(name, n_boot=200)
        intersample(name)
    # Train/evaluation prompt mismatch: trained on short prompts, evaluated on structured and short prompts.
    short = train("lora_only_short_train", 42)
    for prompt_mode in ["structured", "short"]:
        name = f"diag_lora_shorttrain_eval_{prompt_mode}"
        generate(name, "lora_only", short, seeds="42", prompt_mode=prompt_mode)
        evaluate(name, n_boot=200)
        intersample(name)
    # Training length and learning rate.
    long_final = train("lora_only_1000", 42)
    for step in [250, 500, 750, 1000]:
        name = f"diag_lora1000_step{step}"
        generate(name, "lora_only", long_final.parent / f"checkpoint-{step}", seeds="42")
        evaluate(name, n_boot=200)
        intersample(name)
    lr = train("lora_only_lr1e4", 42)
    generate("diag_lora_lr1e4", "lora_only", lr, seeds="42")
    evaluate("diag_lora_lr1e4", n_boot=200)
    intersample("diag_lora_lr1e4")


def main(stage: str) -> None:
    stage_setup()
    if stage in {"main42", "all"}:
        stage_main(42)
    if stage in {"main", "all"}:
        for seed in SEEDS:
            stage_main(seed)
    if stage in {"ablation", "all"}:
        stage_ablation()
    if stage in {"diagnostics", "all"}:
        stage_diagnostics()
    log(f"stage {stage} finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["setup", "main42", "main", "ablation", "diagnostics", "all"])
    main(parser.parse_args().stage)
