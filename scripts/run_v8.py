"""v8 protocol driver (see configs/revision/v8_preregistration.md), resumable.

  pilot   train the four candidates (seed 42), generate/evaluate on the validation split,
          select backbone x L_struct, then lambda_s, with scripts/select_v8.py
  test    freeze the selected configuration as configs/revision/ours_v8*.yaml and run the test
          phase (main comparison, ablations, controllability)

GPU steps wait while another GPU job of the user (alibi) is running or the GPU is occupied.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import yaml

from run_revision_suite import CACHE, CLASSIFIER, EVAL, GEN, MANIFEST, PY, SEEDS, TRAIN, log, run, train

GEN_VAL = Path("outputs/revision/gen_val")
EVAL_VAL = Path("outputs/revision/eval_val")
SELECT = Path("outputs/revision/v8_selection")
CFG = Path("configs/revision")
STAGE1 = ["v8_t2i", "v8_t2i_ls", "v8_cn", "v8_cn_ls"]


def gpu_busy() -> str:
    procs = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Select-Object -ExpandProperty CommandLine"],
                           capture_output=True, text=True).stdout
    if "alibi.cli" in procs:
        return "alibi running"
    used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    if used and int(used[0]) > 4000:
        return f"GPU memory in use: {used[0]} MiB"
    return ""


def wait_gpu() -> None:
    reason = gpu_busy()
    if reason:
        log(f"waiting for GPU ({reason})")
    while reason:
        time.sleep(300)
        reason = gpu_busy()


def gpu_run(cmd: list[str]) -> None:
    wait_gpu()
    run(cmd)


def train_gpu(config: str, seed: int) -> Path:
    cfg = yaml.safe_load((CFG / f"{config}.yaml").read_text(encoding="utf-8"))
    final = TRAIN / f"{config}_s{seed}" / f"checkpoint-{cfg['training']['max_train_steps']}"
    if not final.exists():
        wait_gpu()
    return train(config, seed)


def generate_split(name: str, config: str, ckpt: Path | None, split: str, seeds: str, gen_root: Path) -> Path:
    out = gen_root / name
    marker = out / ".complete"
    if marker.exists():
        return out
    cmd = [PY, "scripts/generate_v7.py", "--config", str(CFG / f"{config}.yaml"), "--manifest-path", MANIFEST,
           "--output-dir", str(out), "--seeds", seeds, "--split", split, "--skip-existing"]
    if ckpt is not None:
        cmd += ["--checkpoint-dir", str(ckpt)]
    gpu_run(cmd)
    marker.write_text("ok")
    return out


def evaluate_split(name: str, split: str, gen_root: Path, eval_root: Path, n_boot: int, sensitivity: bool = False) -> None:
    out = eval_root / name
    if not (out / "summary.json").exists():
        cmd = [PY, "scripts/evaluate_v7.py", "--manifest-path", MANIFEST, "--prediction-dir", str(gen_root / name),
               "--output-dir", str(out), "--cache-dir", CACHE, "--style-classifier", CLASSIFIER, "--n-boot", str(n_boot),
               "--split", split]
        if sensitivity:
            cmd.append("--sensitivity")
        gpu_run(cmd)
    if not (out / "structure_v8_summary.json").exists():
        run([PY, "scripts/structure_metrics_v8.py", "--manifest-path", MANIFEST, "--prediction-dir", str(gen_root / name),
             "--output-dir", str(out), "--split", split])
    if not (out / "style_v8_summary.json").exists():
        run([PY, "scripts/style_v8.py", "score", "--runs", name, "--gen-root", str(gen_root), "--eval-root", str(eval_root),
             "--split", split])


def val_run(name: str, config: str, ckpt: Path | None) -> None:
    generate_split(name, config, ckpt, "val", "42", GEN_VAL)
    evaluate_split(name, "val", GEN_VAL, EVAL_VAL, n_boot=200)


def select(candidates: list[str], tag: str) -> str:
    out = SELECT / f"{tag}.json"
    if not out.exists():
        run([PY, "scripts/select_v8.py", "--eval-root", str(EVAL_VAL), "--candidates", *candidates, "--out", str(out)])
    return json.loads(out.read_text(encoding="utf-8"))["selected"]


def stage_pilot() -> str:
    if not (EVAL_VAL / "real_val" / "structure_v8_summary.json").exists():
        run([PY, "scripts/structure_metrics_v8.py", "--manifest-path", MANIFEST, "--real", "--split", "val",
             "--output-dir", str(EVAL_VAL / "real_val")])
    for config in STAGE1:
        val_run(config, config, train_gpu(config, 42))
    winner = select(STAGE1, "stage1")
    log(f"stage 1 selected {winner}")
    ckpt = TRAIN / f"{winner}_s42" / "checkpoint-500"
    stage2 = [winner]
    for scale in ["06", "08"]:
        name = f"{winner}_infer{scale}"
        val_run(name, name, ckpt)
        stage2.append(name)
    final = select(stage2, "stage2")
    log(f"stage 2 selected {final}")
    # Reference rows on the validation split (not used for selection).
    val_run("lora_controlnet_s42", "lora_controlnet", TRAIN / "lora_only_s42" / "checkpoint-500")
    val_run("lora_ip_adapter_s42", "lora_ip_adapter", TRAIN / "lora_only_s42" / "checkpoint-500")
    val_run("ours_v7_s42", "ours", TRAIN / "ours_s42" / "checkpoint-500")
    return final


def freeze(final: str) -> dict[str, str]:
    """Write the frozen test-phase configs; returns {name: training config name}."""
    winner = json.loads((SELECT / "stage1.json").read_text(encoding="utf-8"))["selected"]
    base = yaml.safe_load((CFG / f"{final}.yaml").read_text(encoding="utf-8"))
    variants = {
        "ours_v8": {},
        "ours_v8_no_style": {"style": False},
        "ours_v8_lineart": {"channels": ["lineart"]},
        "ours_v8_no_layer": {"channels": ["lineart", "blank_space_mask", "salient_composition_mask"]},
        "ours_v8_no_blank": {"channels": ["lineart", "quantized_depth", "salient_composition_mask"]},
        "ours_v8_no_salience": {"channels": ["lineart", "quantized_depth", "blank_space_mask"]},
    }
    with_lstruct = base["training"]["struct_loss_weight"] > 0
    if with_lstruct:
        variants["ours_v8_no_lstruct"] = {"beta": 0.0}
    training_of = {}
    for name, change in variants.items():
        cfg = copy.deepcopy(base)
        cfg["project"]["name"] = name
        cfg["project"]["output_root"] = f"outputs/revision/train/{name}"
        if "style" in change:
            cfg["model"]["style_reference"]["enabled"] = change["style"]
        if "channels" in change:
            cfg["dataset"]["structure_channels_active"] = change["channels"]
        if "beta" in change:
            cfg["training"]["struct_loss_weight"] = change["beta"]
        path = CFG / f"{name}.yaml"
        if not path.exists():
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        # ours_v8 / ours_v8_no_style share the weights of the selected stage-1 candidate.
        training_of[name] = winner if name in ("ours_v8", "ours_v8_no_style") else name
    if with_lstruct:
        # identical training configuration to the stage-1 candidate without L_struct
        training_of["ours_v8_no_lstruct"] = winner.removesuffix("_ls")
    (SELECT / "frozen.json").write_text(json.dumps({"stage1": winner, "final": final, "training_of": training_of}, indent=2),
                                        encoding="utf-8")
    return training_of


def test_run(name: str, config: str, ckpt: Path, seeds: str = "42,52,62", sensitivity: bool = True) -> None:
    generate_split(name, config, ckpt, "test", seeds, GEN)
    evaluate_split(name, "test", GEN, EVAL, n_boot=1000, sensitivity=sensitivity)


def stage_test(final: str) -> None:
    training_of = freeze(final)
    ckpt = lambda name, seed: train_gpu(training_of[name], seed)  # noqa: E731
    for seed in SEEDS:
        test_run(f"ours_v8_s{seed}", "ours_v8", ckpt("ours_v8", seed))
        test_run(f"ours_v8_no_style_s{seed}", "ours_v8_no_style", ckpt("ours_v8_no_style", seed))
        test_run(f"ours_v8_lineart_s{seed}", "ours_v8_lineart", ckpt("ours_v8_lineart", seed))
    for name in ["ours_v8_no_layer", "ours_v8_no_blank", "ours_v8_no_salience", "ours_v8_no_lstruct"]:
        if name in training_of:
            test_run(f"{name}_s42", name, ckpt(name, 42), sensitivity=False)
    for name, config, weights in [
        ("controllability_v8_ours", "ours_v8", ckpt("ours_v8", 42)),
        ("controllability_lora_controlnet", "lora_controlnet", TRAIN / "lora_only_s42" / "checkpoint-500"),
    ]:
        out = Path("outputs/revision") / name
        if not (out / "summary.json").exists():
            gpu_run([PY, "scripts/controllability_v7.py", "--config", str(CFG / f"{config}.yaml"), "--checkpoint-dir",
                     str(weights), "--output-dir", str(out)])
    log("v8 test phase finished")


def stage_smoke() -> None:
    """Two optimiser steps and one validation image for each backbone with L_struct on, so that
    a code error surfaces before the long runs. Outputs go to outputs/revision/smoke."""
    root = Path("outputs/revision/smoke")
    if (root / ".complete").exists():
        return
    for config in ["v8_t2i_ls", "v8_cn_ls"]:
        cfg = yaml.safe_load((CFG / f"{config}.yaml").read_text(encoding="utf-8"))
        cfg["training"].update({"max_train_steps": 2, "grad_accum_steps": 1, "checkpoint_every": 0})
        cfg["project"]["output_root"] = (root / config).as_posix()
        path = root / f"{config}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        gpu_run([PY, "-m", "clpgen.training.train_sdxl_lora", "--config", str(path)])
        gpu_run([PY, "scripts/generate_v7.py", "--config", str(path), "--manifest-path", MANIFEST, "--output-dir",
                 str(root / f"{config}_gen"), "--seeds", "42", "--split", "val", "--limit", "1",
                 "--checkpoint-dir", str(root / config / "checkpoint-2")])
    (root / ".complete").write_text("ok")
    log("smoke test passed")


def main(stage: str) -> None:
    stage_smoke()
    if stage == "smoke":
        return
    final = stage_pilot()
    if stage == "all":
        stage_test(final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["smoke", "pilot", "all"])
    main(parser.parse_args().stage)
