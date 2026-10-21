# SCI Experiment Protocol

## Objective

This document defines a paper-grade experimental protocol for controllable Chinese landscape painting generation under typical SCI reviewer expectations:

- explicit innovations mapped to experiments
- strong baselines and ablations
- fixed train/val/test protocol
- repeated runs and variance reporting
- significance testing where applicable
- objective metrics plus expert evaluation

## Recommended Innovation Claims

### Innovation 1: Hierarchical structure prior for Chinese landscape composition

Validate with:

- `SDXL text-only`
- `SDXL + LoRA`
- `SDXL + LoRA + Hierarchical Adapter`
- `lineart only`
- `lineart + depth`

Reviewer-facing evidence:

- FID / KID
- edge consistency
- blank-space IoU / SSIM
- paired significance tests versus `SDXL + LoRA`

### Innovation 2: Multi-branch controllable generation with style-reference guidance

Validate with:

- `no_style_reference`
- full model with `IP-Adapter`
- same prompt + same structure + different style references

Reviewer-facing evidence:

- style accuracy
- visual style-transfer panels
- expert score on brushwork / artistic quality

### Innovation 3: Structured multimodal prompt design for Chinese-painting semantics

Validate with:

- `short prompt`
- `structured prompt`
- `dense prompt`

Reviewer-facing evidence:

- CLIPScore
- ImageReward
- prompt generalization cases
- prompt-length truncation analysis

## Minimal Main Table

The main quantitative table should contain at least:

1. `SDXL text-only`
2. `SDXL + LoRA`
3. `Ours`

Recommended extra rows:

1. `Ours + short prompt`
2. `Ours + dense prompt`
3. `Ours - style reference`
4. `Ours - structure adapter`

## Reproducibility Protocol

### Data

- Fixed dataset split written to disk
- Group by painter or source before splitting
- Report kept / rejected statistics
- Report final source distribution

### Training

- Use the same base model, image resolution, inference steps, and guidance scale for all methods
- Repeat every trainable method with at least `3` random seeds
- Report `mean+/-std`

### Inference

- Use the same test set for all methods
- Use identical prompt templates for matched comparisons
- Keep structure map and style reference identical unless a control variable is being tested

### Statistics

- For scalar set-level metrics such as `FID`/`KID`, report `mean+/-std` across seeds
- For per-sample metrics such as `CLIPScore`, `edge_consistency`, and `blank_space_ssim`, run:
  - paired t-test
  - Wilcoxon signed-rank test

## Expert Study

### Participants

- `12-15` raters with Chinese painting or digital art background
- blind evaluation
- randomized image order

### Dimensions

- prompt consistency
- compositional structure
- brushwork charm
- artistic quality
- creativity

### Reliability

- Cronbach's alpha or ICC

## Current Repository Status

Already implemented:

- dataset pipeline
- train/eval split materialization
- unified generation pipeline
- quantitative evaluation scripts
- main results export
- significance test script
- seed aggregation script

Still required for final submission-grade paper:

- `3` end-to-end seeds per trainable method
- completed `ImageReward` compatibility check
- completed LPIPS diversity from multi-seed outputs
- expert evaluation and reliability statistics
- larger dataset scale than the current `149` images

## Recommended Final Experimental Narrative

1. Show that diffusion fine-tuning alone improves domain fidelity.
2. Show that hierarchical structure control is responsible for the major fidelity jump over LoRA-only.
3. Show that style reference improves style controllability rather than raw fidelity only.
4. Show that prompt formulation changes semantic alignment and should be optimized separately.
5. Acknowledge that objective structure metrics may not always track artistic quality perfectly.
