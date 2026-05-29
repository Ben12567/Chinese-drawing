# Paper Figure and Table Index

This document maps the generated figures and tables to the paper narrative.

## Generated Figures

All generated figures are available as both `.png` and `.pdf` under:

`reports/final_results/figures`

### Figure A: Main Fidelity Comparison

File:

`reports/final_results/figures/fig_main_fidelity.pdf`

Use in:

`Experiments -> Main Quantitative Results`

Purpose:

Shows that the proposed method has the best in-domain distribution fidelity under `FID` and `KID`.

Recommended caption:

`Distribution fidelity comparison on the Chinese landscape painting benchmark. Lower FID and KID indicate better alignment with the target painting distribution.`

### Figure B: Direction-Aware Metric Profile

File:

`reports/final_results/figures/fig_metric_profile.pdf`

Use in:

`Experiments -> Overall Metric Profile`

Purpose:

Shows that the full method is not merely optimized for one score. It is strongest on domain fidelity, style consistency, and blank-space related control, while IP-Adapter and ControlNet have their own strengths.

Recommended caption:

`Direction-aware normalized metric profile. Each column is normalized so that higher values indicate better performance.`

### Figure C: Ours vs LoRA Relative Gains

File:

`reports/final_results/figures/fig_ours_vs_lora_gains.pdf`

Use in:

`Experiments -> Effectiveness over LoRA Domain Adaptation`

Purpose:

Summarizes the most defensible main claim: the proposed method improves FID, KID, diversity, and style accuracy over `SDXL+LoRA`.

Recommended caption:

`Relative improvements of the proposed method over the LoRA-only baseline.`

### Figure D: Significance Analysis

File:

`reports/final_results/figures/fig_significance_ours_vs_lora.pdf`

Use in:

`Experiments -> Statistical Analysis`

Purpose:

Shows which sample-level metrics have statistical support.

Recommended caption:

`Sample-level Wilcoxon significance test between the proposed method and SDXL+LoRA. The dashed line indicates p=0.05.`

### Figure E: Fidelity-Style Trade-off

File:

`reports/final_results/figures/fig_fidelity_style_tradeoff.pdf`

Use in:

`Discussion -> Method Trade-off`

Purpose:

Shows that the proposed method obtains the strongest balance between low FID and high style accuracy.

Recommended caption:

`Fidelity-style trade-off across methods. The x-axis is inverted so points further right indicate lower FID. Marker size encodes LPIPS diversity.`

### Figure 4: Main Visual Comparison

Files:

`reports/final_results/figures/figure4_main_visual_comparison.png`

`reports/final_results/figures/figure4_main_visual_comparison.pdf`

Use in:

`Experiments -> Qualitative Comparison`

Purpose:

Compares `LoRA-only`, `ControlNet`, `IP-Adapter only`, and the proposed method under matched test-set prompts and structure conditions. The figure emphasizes the intended paper claim: the proposed method gives the most balanced structure-style result, while single-branch baselines tend to show weak layout, rigid contours, or style/layout drift.

Recommended caption:

`Qualitative comparison under matched prompts and structure conditions. LoRA-only captures partial domain appearance but shows weak layout control; ControlNet follows structure more strongly but often produces rigid contours; IP-Adapter only transfers style cues but may drift in spatial composition; the proposed method better balances hierarchy, blank-space organization, brush-and-ink coherence, and style fidelity.`

## Generated Tables

Generated tables are available under:

`reports/final_results/tables`

### Table 1: Clean Main Results

Files:

`reports/final_results/tables/table_main_results_clean.csv`

`reports/final_results/tables/table_main_results_clean.tex`

Use in:

`Experiments -> Main Quantitative Results`

Important:

This table intentionally excludes `ImageReward` because the final result files contain `NaN` for that metric.

### Table 2: Significance Against LoRA

Files:

`reports/final_results/tables/table_significance_ours_vs_lora.csv`

`reports/final_results/tables/table_significance_ours_vs_lora.tex`

Use in:

`Experiments -> Statistical Analysis`

Important:

This is sample-level significance on the test split, not training-seed-level significance.

## Recommended Figure Order

1. Method framework figure
2. Hierarchical structure representation figure
3. `figure4_main_visual_comparison`
4. `fig_main_fidelity`
5. `fig_metric_profile`
6. `fig_ours_vs_lora_gains`
7. `fig_significance_ours_vs_lora`
8. `fig_fidelity_style_tradeoff`
9. Failure cases figure

## Writing Boundary

The generated figures support:

- in-domain fidelity improvement
- style consistency improvement
- diversity improvement
- sample-level significance over `SDXL+LoRA`
- balanced trade-off compared with single-branch baselines

The generated figures do not support:

- completed expert evaluation
- completed final ablation study
- ImageReward comparison
- universal semantic superiority on public compositional benchmarks
