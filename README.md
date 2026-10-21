# CLPGen

`CLPGen` is a research scaffold for the paper topic:

**融合层次结构先验、LoRA 微调与多模态提示的中国山水画高保真可控生成方法研究**

**融合层次结构先验、LoRA 微调与多模态提示的中国山水画高保真可控生成方法研究**

The repository is designed to make the project reproducible from dataset curation to training, inference, evaluation, and paper asset preparation.

## Scope

- Structured prompt generation for Chinese landscape paintings
- Hierarchical structure-map extraction with four channels:
  - `lineart`
  - `quantized_depth`
  - `blank_space_mask`
  - `salient_composition_mask`
- Dataset manifest creation and split management
- Research-ready scaffolding for `SDXL + LoRA + Hierarchical Adapter + Style Reference`
- Evaluation templates for quantitative and expert studies
- Paper tables, figure checklist, and questionnaire templates

## Repository Layout

```text
configs/                 experiment and paper configs
data/                    local metadata and sample manifests
docs/                    paper-facing notes and blueprints
manifests/               schema notes
scripts/                 command-line entrypoints
src/clpgen/              Python package
templates/               tables and questionnaires
```

## Quick Start

### 1. Install base dependencies

```powershell
python -m pip install -e .
```

### 2. Install generation dependencies

```powershell
python -m pip install -e .[gen,metrics]
```

If Hugging Face cache permissions are restrictive on Windows, point the cache to the workspace before running model downloads:

```powershell
$env:HF_HOME = (Resolve-Path ".").Path + "\\.hf"
```

### 3. Prepare a dataset manifest

Put source images under a directory such as `data/raw/landscape_images` and run:

```powershell
python scripts/prepare_dataset.py `
  --images-dir data/raw/landscape_images `
  --output-dir data/processed/clp4k `
  --split-name clp4k_v1
```

This generates:

- normalized image copies
- four-channel structure maps
- a `manifest.jsonl`
- split files for `train`, `val`, and `test`

### 4. Inspect prompt generation

```powershell
python -c "from clpgen.data.prompts import build_prompt_bundle; print(build_prompt_bundle(subject='高山', season='秋', weather='云雾', foreground=['松树','溪亭'], midground=['山径'], background=['远山'], ink_tone='淡墨', palette='浅绛', blankness='大留白'))"
```

### 5. Run an experiment scaffold

```powershell
python scripts/run_inference.py --config configs/experiments/main_sdxl_landscape.yaml --dry-run
```

## Real SDXL Wiring

The repository now includes real `diffusers + peft + SDXL + IP-Adapter` integration:

- training:
  - base model loading with `StableDiffusionXLPipeline`
  - optional external SDXL VAE
  - PEFT LoRA injection into the UNet and both SDXL text encoders
  - hierarchy-aware structure tokens appended to SDXL cross-attention inputs
  - checkpoint export in both:
    - diffusers LoRA format
    - direct PEFT adapter format
- inference:
  - SDXL pipeline loading
  - PEFT bundle reload from training checkpoints
  - structure-token prompt augmentation
  - optional IP-Adapter image guidance through `load_ip_adapter()`

## Tiny Smoke Test

To validate the full training/inference code path without downloading the full SDXL base model:

```powershell
$env:PYTHONPATH = "src"
$env:HF_HOME = (Resolve-Path ".").Path + "\\.hf"
python -m clpgen.training.train_sdxl_lora --config configs/experiments/tiny_sdxl_smoke.yaml
```

This uses the tiny SDXL test pipeline and should produce:

- `outputs/tiny_sdxl_smoke/checkpoint-1/lora`
- `outputs/tiny_sdxl_smoke/checkpoint-1/peft_unet`
- `outputs/tiny_sdxl_smoke/checkpoint-1/peft_text_encoder`
- `outputs/tiny_sdxl_smoke/checkpoint-1/peft_text_encoder_2`
- `outputs/tiny_sdxl_smoke/checkpoint-1/structure_adapter.pt`

You can then validate generation with the saved adapters by pointing `model.lora.weights_path` to the checkpoint root.

## Research Workflow

1. Curate `Chinese Landscape Painting-4K`.
2. Generate structured prompts and structure maps.
3. Fine-tune `SDXL` with `LoRA`.
4. Train the hierarchical structure adapter.
5. Load an image-prompt style branch for reference-style control.
6. Evaluate with `FID`, `KID`, `CLIPScore`, `ImageReward`, `LPIPS diversity`, structure consistency, style accuracy, and expert study.

## Final Paper Suite

The cleaned entrypoint for the final paper-ready experiment package is:

```powershell
$env:PYTHONPATH = "src"
$env:HF_HOME = (Resolve-Path ".").Path + "\\.hf"
python scripts/run_paper_suite.py --config configs/experiments/paper_suite_strong.yaml --skip-existing
```

This suite:

- runs the final baseline set
- aggregates `mean ± std`
- computes significance tables
- writes a packaged results summary for the paper

See also:

- [docs/reproduce_final_experiments.md](docs/reproduce_final_experiments.md)
- [docs/final_experiment_story.md](docs/final_experiment_story.md)

## Current Implementation Status

- Implemented:
  - data schema
  - prompt construction
  - structure-map extraction
  - dataset loading
  - lightweight hierarchical adapter module
  - real SDXL LoRA training entrypoint
  - real SDXL inference wrapper with structure-token conditioning
  - optional IP-Adapter loading path
  - experiment config and report templates
  - evaluation entrypoint

## Recommended Paper Assets

See:

- [docs/paper_blueprint.md](docs/paper_blueprint.md)
- [docs/dataset_pipeline.md](docs/dataset_pipeline.md)
- [templates/questionnaires/expert_eval_form.md](templates/questionnaires/expert_eval_form.md)
- [templates/tables/main_results.csv](templates/tables/main_results.csv)
