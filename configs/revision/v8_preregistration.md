# v8 structure branch — pre-registered development and test protocol

Written on 2026-10-31, before any v8 model was trained or any v8 output was generated.
Test-split results of v8 are produced only after the configuration below has been frozen.

## Motivation (from the v7 results)

The v7 token adapter (193 pooled structure tokens appended to the cross-attention context) did not
make outputs follow their structure condition: in the swap test, edge IoU against the received map
was 0.055 vs 0.053 against maps not received, and all six channel ablations were within noise.
The original manuscript described residual injection, h_l + lambda_s * P_l(F_s^l), and an optional
auxiliary loss L_struct; neither was implemented in v7. v8 implements both.

## What is fixed in advance

* Data, splits, prompts, style references, seeds, base model, LoRA settings, training length
  (500 optimiser steps, batch 1 x 4), resolution, sampler and IP-Adapter (lambda_r = 0.55) are
  identical to v7.
* Branch: input layer extended from 3 to 4 channels (line art inherits the pretrained filters,
  other channels zero-initialised); training scale 1.0; LR 2e-5 for the branch, 1e-4 for its input
  layer; LoRA LR 5e-5.
* L_struct weight beta = 0.05 when used (one-sided L1 on line-art and blank-space regions of the
  TAESD-decoded one-step estimate, weighted by alpha_bar_t).

## Validation pilot (split = val, 122 paintings, seed 42, one image per painting)

Stage 1 — four trained candidates, inference lambda_s = 1.0:
  v8_t2i, v8_t2i_ls, v8_cn, v8_cn_ls.
Stage 2 — the stage-1 winner at lambda_s = 0.6 and 0.8 (no retraining).

Selection rule (applied identically in both stages): rank candidates on each of
  KID (lower), CMMD (lower), edge_iou_specific, blank_iou_re_specific, layer_acc_specific (higher)
(structure metrics from scripts/structure_metrics_v8.py: matched minus mismatched condition) and
choose the lowest mean rank; ties are broken by KID. The rule is applied mechanically by
scripts/select_v8.py; its output is saved next to the pilot results.

## Test phase (split = test, frozen configuration "ours_v8")

Main comparison, seeds 42/52/62: ours_v8 (with IP-Adapter), ours_v8 without style reference, and
ours_v8 trained and run with the line-art channel only (same procedure; isolates the contribution
of the three painting-specific channels from that of fine-tuning the branch). Existing v7 rows
(SDXL, LoRA-only, ControlNet, T2I-Adapter, IP-Adapter-only, LoRA+ControlNet, LoRA+IP-Adapter, v7
token adapter) are reported unchanged.
Channel ablation, seed 42: without layer map, without blank-space mask, without salience mask;
and with/without L_struct if the selected configuration uses it.
Controllability (structure swap and style swap, same sampling as v7) for ours_v8 and
LoRA+ControlNet.

All results are reported whatever their direction, including the pilot results and the v7 token
adapter results.
