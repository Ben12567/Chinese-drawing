# Paper Figure and Table Index

This index maps the current reproducible outputs to the paper narrative. It separates main-text evidence from appendix-only evidence.

## Main Figure Set

1. Figure 1: Motivation and task formulation.
   - File: `paper/editable_figure1_motivation_overview.pptx`
   - Use: Introduction.

2. Figure 2: Method framework.
   - Status: keep the current framework diagram or redraw as editable PPT.
   - Use: Method.

3. Figure 3: Dataset and structural annotation overview.
   - Files: `reports/final_results/figures/figure3_dataset_and_structure_overview.png`, `reports/final_results/figures/figure3_dataset_and_structure_overview.pdf`
   - Use: Dataset and experimental protocol.

4. Figure 4: Main qualitative comparison.
   - Files: `reports/final_results/figures/figure4_main_visual_comparison.png`, `reports/final_results/figures/figure4_main_visual_comparison.pdf`
   - Editable PPT: `paper/editable_figures_4_5.pptx`
   - Use: Qualitative results.

5. Figure 5: Structure controllability visualization.
   - Files: `reports/final_results/figures/figure5_structure_controllability.png`, `reports/final_results/figures/figure5_structure_controllability.pdf`
   - Editable PPT: `paper/editable_figures_4_5.pptx`
   - Use: Controllability analysis.

6. Figure 6: Quantitative results dashboard.
   - Files: `reports/final_results/figures/figure6_quantitative_results_dashboard.png`, `reports/final_results/figures/figure6_quantitative_results_dashboard.pdf`
   - Use: Main quantitative results.

7. Figure 7: Fidelity-style trade-off and significance.
   - Files: `reports/final_results/figures/figure7_tradeoff_and_significance.png`, `reports/final_results/figures/figure7_tradeoff_and_significance.pdf`
   - Use: Statistical analysis and discussion.

8. Figure 8: Representative limitations and failure cases.
   - Files: `reports/final_results/figures/figure8_failure_cases.png`, `reports/final_results/figures/figure8_failure_cases.pdf`
   - Use: Discussion or limitations.

## Appendix Figure

- Figure 9: Pilot ablation and general benchmark boundary.
  - Files: `reports/final_results/figures/figure9_ablation_and_generalization.png`, `reports/final_results/figures/figure9_ablation_and_generalization.pdf`
  - Use only as appendix/supporting evidence because the ablation part comes from an earlier protocol.

## Main Table Set

All generated LaTeX tables are under `paper/tables`.

1. `table1_dataset_overview.tex`: dataset size, split, source, painter, and resolution statistics.
2. `table2_dataset_style_distribution.tex`: long-tailed style-label distribution.
3. `table3_experimental_protocol.tex`: training/evaluation protocol and sample counts.
4. `table4_method_configuration.tex`: compared method configuration.
5. `table5_main_quantitative_results.tex`: FID, KID, CLIP, PickScore, HPSv2, and LPIPS.
6. `table6_style_structure_results.tex`: style accuracy, edge consistency, Blank IoU, and Blank SSIM.
7. `table7_significance_ours_vs_lora.tex`: paired significance analysis against LoRA-only.
8. `table8_pilot_ablation_and_general_benchmark.tex`: pilot ablation plus general benchmark boundary.

## Recommended Writing Boundary

The current evidence supports claims about target-domain fidelity, style consistency, blank-space controllability, structure-style balance, and significant improvements over LoRA-only on selected sample-level metrics.

Do not claim completed expert evaluation, valid ImageReward results, or a final full-protocol ablation study. The existing ablation should be described as pilot or moved to appendix unless rerun under the final 817-image protocol.
