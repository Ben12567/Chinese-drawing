# Available Experiments

This document records which experiments are supported by the final result files.

## Final Result Location

Final public result snapshot:

`reports/final_results`

Full local experiment outputs:

`outputs/paper_suite_strong`

## Experiments Supported by Final Results

The final result package supports the following claims.

### Main Quantitative Comparison

Available methods:

- `paper_ours_817`
- `paper_lora_only_817`
- `paper_controlnet_817`
- `paper_ip_adapter_only_817`

Main files:

- `reports/final_results/main_results_mean_std.csv`
- `reports/final_results/main_results_raw.csv`

### Multi-Seed Training Results

Available:

- `Ours`: 3 training seeds, `42`, `52`, `62`
- `LoRA-only`: 3 training seeds, `42`, `52`, `62`

Single-run baselines:

- `ControlNet`: seed `42`
- `IP-Adapter only`: seed `42`

### Sample-Level Significance

Available files:

- `reports/final_results/significance_ours_vs_lora.csv`
- `reports/final_results/significance_ours_vs_controlnet.csv`
- `reports/final_results/significance_ours_vs_ip_adapter_only.csv`

These files support sample-level significance analysis on the test split.

### Public Benchmark Caveat

Available file:

- `reports/final_results/benchmark_summary_v6.json`

Available benchmark results:

- `TIFA subset`
- `T2I-CompBench non-spatial`

These benchmark results should be discussed as a caveat. They do not support a claim of universal semantic superiority.

### Reproducibility Protocol

Available files:

- `configs/experiments/paper_suite_strong.yaml`
- `scripts/run_paper_suite.py`
- `scripts/run_multi_seed_suite.py`
- `scripts/evaluate_experiments.py`
- `scripts/package_paper_results.py`
- `docs/reproduce_final_experiments.md`

## Experiments Not Supported as Final Completed Results

The following should not be described as completed final experiments in the paper.

### Ablation Study

There are older intermediate ablation files under `outputs/paper_suite`, but they are not part of the final `paper_suite_strong` protocol and are not included in `reports/final_results`.

Do not write the ablation study as a completed final result unless it is rerun under the same `817`-image final protocol.

### Expert Evaluation

Questionnaire templates and analysis scripts exist, but no completed human rating file is included in the final result package.

Do not report expert evaluation, Cronbach's alpha, or ICC as completed results.

### ImageReward

`ImageReward` columns in the final CSV files are empty or `NaN`.

Do not report ImageReward in the formal result table.

### GenEval Official

The official GenEval run was skipped because the local environment lacked `mmdet/mmcv`.

Do not report GenEval as a completed result.

## Recommended Paper Scope

The final paper should present:

- main quantitative comparison
- sample-level significance analysis
- objective structure/style metrics
- public benchmark caveat
- reproducibility protocol

The paper should not present:

- completed ablation study
- completed expert evaluation
- ImageReward comparison
- official GenEval comparison
