# Paper Blueprint

## Recommended Title

融合层次结构先验、LoRA微调与多模态提示的中国山水画高保真可控生成方法研究

## Core Contributions

1. A Chinese-landscape-specific four-channel hierarchical structure representation.
2. A controllable generation framework that couples structured prompts, hierarchy-aware adapter guidance, and style-reference conditioning.
3. A reproducible experimental protocol combining objective metrics and expert evaluation.

## Suggested Section Flow

### 1. Introduction

- Problem: general T2I models fail to preserve Chinese landscape composition, blankness, and brushwork hierarchy.
- Gap: existing controllable generation methods use natural-image controls and under-model Chinese painting semantics.
- Contribution summary: structured prompt design, hierarchical structure maps, multi-branch controllable generation, and rich evaluation.

### 2. Related Work

- Chinese painting generation and restoration
- Diffusion-based controllable generation
- LoRA and lightweight adaptation
- Art aesthetics and expert evaluation

### 3. Method

- Structured prompt specification
- Four-channel hierarchical structure-map extraction
- SDXL backbone and LoRA adaptation
- Hierarchical structure adapter
- Style-reference branch
- Joint inference objective

### 4. Experimental Setup

- Dataset construction
- Baselines
- Metrics
- Expert-study protocol
- Implementation details

### 5. Results

- Main quantitative comparison
- Visual comparison
- Ablation study
- Prompt generalization study
- Structure and style control cases
- Failure analysis

### 6. Conclusion

- Summarize gains in fidelity, structure control, and style control separately.

## Mandatory Figures

1. Overall framework
2. Dataset and annotation examples
3. Structure-map extraction workflow
4. Main visual comparison
5. Ablation comparison
6. Same prompt, different structure
7. Same structure, different style reference
8. Failure cases

## Mandatory Tables

1. Dataset statistics
2. Main quantitative results
3. Ablation
4. Prompt-type generalization
5. Sampling/control sensitivity
6. Expert study
