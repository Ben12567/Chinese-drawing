# Extended Experiment Package

This package expands the paper result presentation without fabricating unrun experiments.

## Recommended Final Figure Set

1. Figure 1: Motivation overview, editable PowerPoint.
2. Figure 2: Method framework, still needs final editable diagram.
3. Figure 3: Dataset and structure overview, generated from real dataset statistics.
4. Figure 4: Main qualitative comparison, editable PowerPoint.
5. Figure 5: Structure controllability visualization, editable PowerPoint.
6. Figure 6: Quantitative results dashboard.
7. Figure 7: Fidelity-style trade-off and significance.
8. Figure 8: Representative failure cases and limitations.

Optional appendix figure:

- Figure 9: Pilot ablation and general benchmark boundary.

## Recommended Final Table Set

1. Dataset overview.
2. Style-label distribution.
3. Experimental protocol.
4. Compared method configuration.
5. Main quantitative results.
6. Style and structure metrics.
7. Paired significance analysis.
8. Pilot ablation and general benchmark boundary.

## Ablation Policy

The existing component ablation is usable only as a pilot/appendix result because it was produced under an earlier protocol. For a strong method paper, rerun the following under the final 817-image protocol:

- Full model.
- w/o hierarchical structure adapter.
- lineart only.
- lineart + quantized depth.
- w/o blank-space mask.
- w/o saliency mask.
- w/o style reference.
- short prompt vs structured prompt.
- LoRA U-Net only vs U-Net + text encoder.

Report all final ablations with the same test split, resolution, inference seeds, and metrics as the main table.
