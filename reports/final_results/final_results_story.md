# Final Experimental Story

## Recommended Claim
The proposed method is strongest on in-domain fidelity, style consistency, and diversity for Chinese landscape painting, rather than universal semantic alignment.

## Main Table Narrative
- Ours vs LoRA-only: FID 169.16+/-5.68 vs 273.40+/-26.12; KID 0.0614+/-0.0050 vs 0.2287+/-0.0611
- Ours improves FID by 38.1%, KID by 73.1%, LPIPS diversity by 25.4%, and style accuracy by 71.4% over LoRA-only.
- Ours also outperforms ControlNet on FID/KID (265.89/0.1841) and IP-Adapter only on FID/KID (213.32/0.1095).

## Logical Interpretation
- The method is strongest on in-domain fidelity and style organization.
- ControlNet retains an advantage on hard edge adherence; this should be stated explicitly.
- IP-Adapter only is strong on preference-style metrics, but weaker on overall distribution matching than the full method.

## Safe Writing Boundary
- Claim superiority in Chinese landscape painting fidelity, style consistency, and diversity.
- Avoid claiming universal semantic superiority or stronger edge control than ControlNet.
- Avoid claiming human-level artistic judgment from objective metrics alone.

## Benchmark Caveat
- TIFA subset: ours 0.7566, lora-only 0.7783
- T2I-CompBench non-spatial: ours 0.3040, lora-only 0.3122
- These public compositional benchmarks do not support a claim of universal semantic superiority.