"""Build the Zenodo deposit (dataset, results, weights, code) promised in the Data availability statement.

Writes <repo>/../zenodo_upload/{README.md, clp_v7_dataset.zip, results.zip, weights.zip, code_v2-revision.zip}.
Generated evaluation images (~21 GB) are not included; they can be regenerated from the weights and seeds.
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "zenodo_upload"
REV = ROOT / "outputs/revision"

RUN_LABELS = {
    "v8_cn_s42": "Ours (residual structure branch, ControlNet init.; also 'Ours w/o style ref.' at inference), seed 42",
    "v8_cn_s52": "Ours, seed 52", "v8_cn_s62": "Ours, seed 62",
    "ours_v8_lineart_s42": "Ours, line art only, seed 42", "ours_v8_lineart_s52": "Ours, line art only, seed 52",
    "ours_v8_lineart_s62": "Ours, line art only, seed 62",
    "ours_v8_no_layer_s42": "Ablation w/o layer map", "ours_v8_no_blank_s42": "Ablation w/o blank-space mask",
    "ours_v8_no_salience_s42": "Ablation w/o salience mask",
    "v8_cn_ls_s42": "Selection candidate: ControlNet init. + L_struct", "v8_t2i_s42": "Selection candidate: T2I-Adapter init.",
    "v8_t2i_ls_s42": "Selection candidate: T2I-Adapter init. + L_struct",
    "lora_only_s42": "LoRA-only (also the LoRA of LoRA+ControlNet / LoRA+IP-Adapter), seed 42",
    "lora_only_s52": "LoRA-only, seed 52", "lora_only_s62": "LoRA-only, seed 62",
    "ours_s42": "Token adapter (original implementation), seed 42", "ours_s52": "Token adapter, seed 52",
    "ours_s62": "Token adapter, seed 62",
    "ch_lineart_s42": "Token adapter channel ablation: line art only", "ch_depth_s42": "Token adapter: layer map only",
    "ch_blank_s42": "Token adapter: blank-space mask only", "ch_saliency_s42": "Token adapter: salience mask only",
    "ch_lineart_depth_s42": "Token adapter: line art + layer map", "ch_no_saliency_s42": "Token adapter: w/o salience mask",
    "lora_only_1000_s42": "LoRA-only diagnostic: 1000 steps", "lora_only_lr1e4_s42": "LoRA-only diagnostic: learning rate 1e-4",
    "lora_only_short_train_s42": "LoRA-only diagnostic: trained on short prompts",
}


def add_tree(z: zipfile.ZipFile, base: Path, arc_prefix: str, suffixes: tuple[str, ...] | None = None,
             skip: tuple[str, ...] = ("cache",)) -> int:
    n = 0
    for p in sorted(base.rglob("*")):
        if not p.is_file() or any(s in p.parts for s in skip):
            continue
        if suffixes and p.suffix.lower() not in suffixes:
            continue
        stored = p.suffix.lower() in (".png", ".jpg", ".jpeg", ".safetensors", ".pt", ".pkl")
        z.write(p, f"{arc_prefix}/{p.relative_to(base).as_posix()}",
                compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
        n += 1
    return n


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT / "clp_v7_dataset.zip", "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        n = add_tree(z, ROOT / "data/processed/clp_v7", "clp_v7")
        z.write(ROOT / "data/metadata/cleaned_report_v7.jsonl", "cleaning/cleaned_report_v7.jsonl")
        for f in ["cleaning_summary.json", "rejection_breakdown.csv", "artist_overlap.json", "visual/visual_audit_removals.py"]:
            z.write(REV / "audit" / f, f"cleaning/{f}")
        print("dataset files:", n + 5)
    with zipfile.ZipFile(OUT / "results.zip", "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        n = 0
        for d in ["eval", "eval_val", "stats_v8", "v8_selection", "controllability", "controllability_v8_ours",
                  "controllability_lora_controlnet"]:
            n += add_tree(z, REV / d, d, suffixes=(".csv", ".json"))
        for f in ["stats_spec_v8.json", "stats_ablation_v8.json", "style_classifier_v8.pkl", "style_classifier_v8_report.json"]:
            z.write(REV / f, f)
            n += 1
        print("result files:", n)
    with zipfile.ZipFile(OUT / "weights.zip", "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for run in RUN_LABELS:
            d = REV / "train" / run
            last = max(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
            add_tree(z, last, f"{run}/{last.name}")
            if (d / "train_log.csv").exists():
                z.write(d / "train_log.csv", f"{run}/train_log.csv")
        print("weight runs:", len(RUN_LABELS))
    subprocess.run(["git", "archive", "--format=zip", "--prefix=Chinese-drawing/", "-o", str(OUT / "code_v2-revision.zip"), "HEAD"],
                   cwd=ROOT, check=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    lines = ["# Data and models for PONE-D-26-39475 (revised)", "",
             "Hierarchical structure and style-coordinated diffusion for controllable Chinese landscape painting generation.", "",
             "## Files", "",
             "- `clp_v7_dataset.zip`: 1,228 cropped paintings (public domain / CC0), four-channel condition maps, "
             "`manifest.jsonl` (museum object identifiers, URLs, credit lines, licences, VLM prompt fields, style labels), "
             "train/val/test splits, the cleaning report with the decision for every candidate image, the visual-audit "
             "removal list and the artist-overlap count.",
             "- `results.zip`: per-image and per-sample metrics of every run (test and validation), statistics tables "
             "(`stats_v8/`), validation selection (`v8_selection/`), controllability results, the frozen colour-treatment classifier.",
             "- `weights.zip`: final checkpoint of every training run (LoRA, structure branch or token adapter) and its training log.",
             f"- `code_v2-revision.zip`: the code at commit {head} (GitHub: https://github.com/Ben12567/Chinese-drawing, tag v2-revision).", "",
             "Generated evaluation images (~21 GB) are not included; they are reproduced exactly from the weights, the prompts "
             "in the manifest and the per-sample seeds (see CHANGELOG_REVISION.md in the code).", "",
             "The weights are adapters for stabilityai/stable-diffusion-xl-base-1.0 (and, for the structure branch, "
             "diffusers/controlnet-canny-sdxl-1.0); their use is subject to the licences of these base models.", "",
             "## Run names in weights.zip and results.zip", "", "| run | in the manuscript |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in RUN_LABELS.items()]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for p in sorted(OUT.iterdir()):
        print(f"{p.name}: {p.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
