#!/bin/bash
# v8 protocol (configs/revision/v8_preregistration.md). Resumable; rerun after an interruption.
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/Scripts/python -u scripts/run_v8.py --stage all >> logs/run_v8.log 2>&1
