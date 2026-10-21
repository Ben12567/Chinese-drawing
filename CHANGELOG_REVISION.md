# Changes for the revised submission (PONE-D-26-39475, revision 1)

Baseline: commit `03c8107` (3 June 2026), the version cited in the original submission.
All experiments reported in the revised manuscript were produced with this revised code.

## Corrections to the original pipeline

| Area | Original behaviour (03c8107) | Revised behaviour |
|---|---|---|
| Evaluation sample | `evaluate_experiments.py` used only the first image per test sample (seed 42) for every metric except LPIPS, i.e. 83 generated vs 83 real images for FID/KID | `evaluate_v7.py` uses all 3 inference seeds (249 images per run); per-image metrics are averaged per test sample |
| Initial noise | identical seeds for every test sample | noise seed derived from (sample id, seed); identical across methods for the same pair (`generate_v7.py --seed-mode per_sample`) |
| Prompts | keyword rules + fixed defaults; short template for training, structured template for evaluation | per-image fields from Qwen2.5-VL-3B-Instruct (`extract_prompt_fields.py`); same nine-field prompt for training and evaluation |
| Style reference | same style label, same split (test samples received test paintings) | same style label, always from the training split (`build_dataset_v7.py`) |
| Style labels | medium keywords, named as art-historical styles ("blue_green_landscape" …) | same rules, renamed to what they encode (`ink_and_colour`, `light_colour`, `ink_monochrome`, `mi_style`, `dong_yuan_style`) |
| ControlNet input | dark lines on white passed to the Canny ControlNet | inverted to white lines on black (`invert_lineart: true`) |
| Configs | `adapter_conditioning_scale` (λs), `warmup_steps`, `style_reference.encoder_name` set but never read by the code | removed from the revision configs (`configs/revision/`) |
| Dataset cleaning | quality filter + pHash de-duplication only | + object-type metadata filter, mounting crop, CLIP ViT-H/14 subject verification, visual audit, seal statistics (`verify_and_crop.py`, `src/clpgen/data/mounting.py`) |
| MET collector | `/v1/search` (retired by the MET on 2026-10-01) | `/v1.1/search` with offset pagination; 404s are not retried |
| Training log | none | per-step loss in `train_log.csv` |
| Package metadata | author "Codex" | human authors |

## New components

- Baselines: T2I-Adapter (line art), LoRA+ControlNet, LoRA+IP-Adapter, unadapted SDXL (`configs/revision/`).
- Metrics: CMMD; bootstrap CIs for FID/KID/CMMD (exact low-rank FID, validated against clean-fid); real-vs-real reference; seal-inpainted and centre-crop sensitivity; frozen style classifier with real-test accuracy, majority baseline, macro-F1 and confusion matrices (`train_style_classifier.py`).
- Statistics: per-painting paired t / Wilcoxon, Holm correction, bootstrap CIs of differences, Cohen's d_z (`stats_v7.py`).
- Ablations under the final protocol, including salience-only and w/o-salience; LoRA-only diagnostics (`run_revision_suite.py --stage ablation|diagnostics`).
- Randomly sampled qualitative figures with per-image metrics; failure cases (`scripts/figures/`).
- Controllability experiments with quantitative summaries (`controllability_v7.py`).

## Reproducing the revised results

```
python scripts/collect_met_v7.py --out data/metadata/candidates_v7_met.jsonl
python scripts/collect_landscape_dataset.py --sources cma --output-manifest data/metadata/candidates_v7_cma.jsonl
python scripts/collect_landscape_dataset.py --sources artic --output-manifest data/metadata/candidates_v7_artic.jsonl
python scripts/collect_landscape_dataset.py --sources wikimedia --sleep-seconds 1.5 --output-manifest data/metadata/candidates_v7_wikimedia.jsonl
python scripts/download_candidates.py ...
python scripts/verify_and_crop.py ...
python scripts/extract_prompt_fields.py ...
python scripts/build_dataset_v7.py ...
python scripts/make_revision_configs.py
python scripts/run_revision_suite.py --stage all
python scripts/stats_v7.py --spec outputs/revision/stats_spec.json --out-dir outputs/revision/stats
python scripts/controllability_v7.py
python scripts/figures/make_revision_figures.py
```
Because museum collections and their APIs change over time, re-collection will not reproduce the
exact image set; the released manifest (Zenodo) lists the images used.

## Notes

The documents under `docs/` that were drafted with an AI coding agent during the original study
(paper blueprints and experiment narratives) are retained for transparency but are not part of the
method; see the "Use of AI tools" section of the manuscript.
