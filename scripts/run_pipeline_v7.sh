#!/bin/bash
# Full v7 revision pipeline. Every step is resumable; rerun the script after an interruption.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1  # all models are cached locally; avoid network stalls
export TRANSFORMERS_OFFLINE=1
PY=.venv/Scripts/python
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/pipeline_v7.log; }

log "1. VLM prompt fields"
$PY -u scripts/extract_prompt_fields.py --report-path data/metadata/cleaned_report_v7.jsonl \
    --out-path data/metadata/vlm_fields_v7.jsonl >> logs/vlm_fields.log 2>&1

log "2. Build dataset"
if [ ! -f data/processed/clp_v7/manifest.jsonl ]; then
  $PY -u scripts/build_dataset_v7.py --report data/metadata/cleaned_report_v7.jsonl \
      --fields data/metadata/vlm_fields_v7.jsonl --output-dir data/processed/clp_v7 >> logs/build_dataset.log 2>&1
fi
$PY scripts/make_revision_configs.py >> logs/pipeline_v7.log 2>&1

log "3. Main comparison, seed 42"
$PY -u scripts/run_revision_suite.py --stage main42
log "4. Main comparison, seeds 52 and 62"
$PY -u scripts/run_revision_suite.py --stage main
log "5. Channel ablation"
$PY -u scripts/run_revision_suite.py --stage ablation
log "6. LoRA-only diagnostics"
$PY -u scripts/run_revision_suite.py --stage diagnostics

log "7. Controllability"
$PY -u scripts/controllability_v7.py >> logs/controllability.log 2>&1

log "8. Statistics"
cat > outputs/revision/stats_spec.json <<'JSON'
{
  "SDXL": ["outputs/revision/eval/sdxl_base"],
  "LoRA-only": ["outputs/revision/eval/lora_only_s42", "outputs/revision/eval/lora_only_s52", "outputs/revision/eval/lora_only_s62"],
  "ControlNet": ["outputs/revision/eval/controlnet"],
  "T2I-Adapter": ["outputs/revision/eval/t2i_adapter"],
  "IP-Adapter-only": ["outputs/revision/eval/ip_adapter_only"],
  "LoRA+ControlNet": ["outputs/revision/eval/lora_controlnet_s42", "outputs/revision/eval/lora_controlnet_s52", "outputs/revision/eval/lora_controlnet_s62"],
  "LoRA+IP-Adapter": ["outputs/revision/eval/lora_ip_adapter_s42", "outputs/revision/eval/lora_ip_adapter_s52", "outputs/revision/eval/lora_ip_adapter_s62"],
  "Ours w/o style ref.": ["outputs/revision/eval/ours_no_style_s42", "outputs/revision/eval/ours_no_style_s52", "outputs/revision/eval/ours_no_style_s62"],
  "Ours": ["outputs/revision/eval/ours_s42", "outputs/revision/eval/ours_s52", "outputs/revision/eval/ours_s62"]
}
JSON
$PY -u scripts/stats_v7.py --spec outputs/revision/stats_spec.json --out-dir outputs/revision/stats >> logs/stats.log 2>&1

log "9. Figures and tables"
(cd scripts/figures && ../../$PY make_revision_figures.py) >> logs/figures.log 2>&1 || log "figures failed (see logs/figures.log)"
$PY scripts/make_tables.py >> logs/tables.log 2>&1 || log "tables failed (see logs/tables.log)"
log "pipeline finished"
