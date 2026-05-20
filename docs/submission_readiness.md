# Submission Readiness

## Hard Requirements for a Submission-Ready Version

- fixed train/val/test split saved to disk
- same base model and inference settings across compared methods
- at least `3` end-to-end training seeds for every trainable method
- report `mean +/- std` for set-level metrics
- paired significance tests for per-sample metrics
- expert blind study with reliability analysis
- disclose data source, size, and filtering criteria

## Current Automated Status on 2026-04-27

Completed in this repository:

- dataset preparation, cleaning, and split materialization
- SDXL + LoRA training and checkpoint reload
- hierarchical structure adapter integration
- optional IP-Adapter style-reference branch
- unified evaluation for `FID`, `KID`, `CLIPScore`, `LPIPS diversity`, structure consistency, and style accuracy
- significance test script
- multi-seed suite runner
- expert-score analysis script with `Cronbach's alpha` and `ICC(2,1)`

Still requiring real execution before direct submission:

- full `3-seed` training runs for `SDXL + LoRA` and `Ours`
- completed expert blind ratings from human raters
- larger public dataset if the target journal expects stronger scale than the current `149` images
- stable `ImageReward` environment, or explicit removal of that metric from the paper

## Author Guidance

If the paper must be submitted immediately, describe the current work as a small-scale controlled study and do not claim a large-scale benchmark. If the goal is a stronger SCI submission, finish the remaining items above and refresh the main tables before writing the final manuscript.
