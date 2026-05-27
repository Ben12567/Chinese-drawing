# 顶刊风格论文整体设计

## 论文定位

建议不要把论文写成“通用文生图模型改进”，而是写成一篇面向文化艺术图像的任务型方法论文：

**面向中国山水画的层次结构与风格协同可控生成方法**

英文标题建议：

**Hierarchical Structure and Style-Coordinated Diffusion for High-Fidelity Chinese Landscape Painting Generation**

更稳的中文标题：

**融合层次结构先验与风格协同控制的中国山水画高保真生成方法**

这篇论文最核心的判断是：通用扩散模型和 LoRA 域适配可以学到“中国画外观”，但很难同时稳定处理山水画中的章法层次、留白关系和笔墨风格。本文方法的价值在于把这些文化图像特征拆成可建模的条件，并通过 `LoRA + hierarchical structure adapter + style reference branch` 做协同生成。

## 核心故事

中国山水画生成不是普通风格迁移问题。它至少包含三类约束：

1. 结构约束：远中近景、山石、水体、树木、亭台与云雾之间存在稳定章法关系。
2. 风格约束：笔墨组织、设色方式和画派特征共同决定图像是否像中国画。
3. 留白约束：留白不是背景空洞，而是构图和意境的一部分。

现有方法的问题可以这样写：

- `SDXL text-only` 语义表达能力强，但域内风格不稳定。
- `SDXL + LoRA` 能提升中国画外观，但结构和风格控制仍然弱。
- `ControlNet` 对边缘服从强，但容易过度依赖线条，整体画面风格和域内分布未必最优。
- `IP-Adapter only` 感知偏好和风格参考较强，但缺少山水画结构先验。

本文方法的切入点：

**不是单独加强文本、边缘或风格，而是把中国山水画拆成“文本语义、层次结构、风格参考”三类互补条件，并让它们在扩散生成过程中协同工作。**

## 创新点包装

### 创新点一：面向中国山水画的层次结构先验

不直接使用自然图像里的通用边缘或深度，而是构建四通道结构表示：

- `lineart`
- `quantized depth`
- `blank-space mask`
- `salient composition mask`

这套结构不是为了精确还原物理深度，而是为了表达中国山水画中的章法层次、留白分布和视觉重心。

论文中应强调：

**结构图是一种任务特定的艺术结构先验，而不是普通控制图的简单替换。**

### 创新点二：结构、文本与风格参考的协同扩散框架

方法由三部分组成：

- `LoRA` 负责中国山水画域适配。
- `Hierarchical Adapter` 负责注入层次结构与留白约束。
- `Style Reference Branch` 负责增强笔墨风格一致性。

这比单独的 LoRA、ControlNet 或 IP-Adapter 更符合中国画生成任务，因为中国山水画的质量不是由某一个条件决定，而是由结构、风格和语义共同决定。

### 创新点三：面向文化绘画生成的客观评测协议

不能只用 `FID/KID`，也不能只用 `CLIPScore`。本文应建立一个多维评测协议：

- 分布保真：`FID`, `KID`
- 文本一致性：`CLIPScore`
- 人类偏好代理：`PickScore`, `HPSv2`
- 多样性：`LPIPS diversity`
- 结构保持：`edge consistency`, `blank-space IoU`, `blank-space SSIM`
- 风格一致性：`style accuracy`
- 统计显著性：paired t-test, Wilcoxon test

论文写法要谨慎：

**这套协议用于客观评估生成质量、结构控制和风格一致性，不声称完全替代人工艺术评价。**

## 全文结构

### 1. Introduction

第一段：从中国山水画的文化与视觉复杂性切入。

第二段：指出通用扩散模型在中国山水画生成上的问题：结构漂移、留白失控、风格不稳定。

第三段：分析现有控制方法的局限：LoRA 只做域适配，ControlNet 偏硬结构，IP-Adapter 偏风格参考，缺少协同建模。

第四段：提出本文方法：层次结构先验与风格协同控制的扩散框架。

贡献点建议写三条：

- 提出一种面向中国山水画的层次结构表示，显式编码线稿、伪深度、留白和构图显著区域。
- 设计结构、文本与风格参考协同的扩散生成框架，实现中国山水画的高保真可控生成。
- 构建统一客观实验协议，并在多基线、多种子和显著性检验下验证方法有效性。

### 2. Related Work

建议分四节：

- Text-to-image diffusion models
- Parameter-efficient adaptation and LoRA
- Controllable generation with ControlNet, T2I-Adapter, IP-Adapter
- Chinese painting generation and cultural heritage image synthesis

不要把相关工作写成罗列，要围绕一个问题组织：

**现有方法各自解决了文本、结构或风格的一部分，但没有针对中国山水画的结构-风格协同机制。**

### 3. Method

建议结构：

#### 3.1 Problem Formulation

定义输入：

- 文本提示 `p`
- 层次结构图 `s`
- 风格参考图 `r`

输出：

- 中国山水画图像 `x`

目标：

在保持文本语义的同时，提高域内保真、结构一致和风格一致。

#### 3.2 Hierarchical Structure Representation

解释四通道结构图：

- `lineart` 表达笔线和主要轮廓
- `quantized depth` 表达远中近层次
- `blank-space mask` 表达留白布局
- `salient composition mask` 表达构图重心

这里要放一张图，展示原图和四个结构通道。

#### 3.3 LoRA-based Domain Adaptation

说明为什么不用全参数微调：

- 单卡资源可复现
- 避免过拟合
- 适合中等规模文化图像数据集

#### 3.4 Hierarchical Adapter

说明如何把结构图转成结构 token 或结构条件，并与扩散模型交互。

这一节是方法核心之一，要画清楚结构分支进入 U-Net 或 cross-attention 的路径。

#### 3.5 Style Reference Branch

说明风格参考图如何提供笔墨和设色倾向。

强调它不是单独替代 LoRA，而是和结构先验共同工作。

#### 3.6 Training and Inference

写训练目标、推理流程、控制强度和多 seed 设置。

### 4. Dataset and Protocol

这节要写得非常严谨。

内容包括：

- 数据来源
- 清洗规则
- 去重策略
- 训练/验证/测试划分
- prompt 生成模板
- 结构图生成规则
- 统一训练设置
- 统一推理设置
- 所有方法的公平比较设置

当前数据集规模是 `817`，顶刊写法建议避免夸成大规模数据集。可以称为：

**a curated benchmark of Chinese landscape paintings**

不要写成：

**large-scale dataset**

### 5. Experiments

实验顺序建议如下：

#### 5.1 Main Quantitative Comparison

主表放：

- `ControlNet`
- `IP-Adapter only`
- `SDXL + LoRA`
- `Ours`

主表重点解释：

- `Ours` 在 `FID/KID` 上显著最好。
- `Ours` 在 `LPIPS diversity` 和 `StyleAcc` 上优于 `LoRA-only`。
- `ControlNet` 的结构边缘更强，但整体分布不如 `Ours`。
- `IP-Adapter only` 的偏好指标强，但分布保真不如 `Ours`。

#### 5.2 Statistical Significance

单独成小节。

重点写：

- `Ours vs LoRA-only` 在 `PickScore`, `HPSv2`, `style accuracy`, `edge consistency`, `blank-space IoU` 上有显著性支持。
- `CLIPScore` 不显著，说明本文方法不是靠字面语义对齐取胜，而是提升域内视觉质量和风格结构。

#### 5.3 Visual Comparison

图像对比必须丰富。

每组图建议包含：

- prompt
- structure map
- style reference
- LoRA-only result
- ControlNet result
- IP-Adapter only result
- Ours result

图注要直接说明观察点：

- LoRA-only 风格粗略但结构松散
- ControlNet 边缘贴合但画面僵硬
- IP-Adapter only 风格接近但章法不稳
- Ours 在层次、留白和风格上更平衡

#### 5.4 Ablation Study

建议至少列这些：

- full model
- without structure
- without style reference
- LoRA-only
- short prompt / structured prompt / dense prompt
- lineart only / lineart + depth / full four-channel structure

如果其中一些还没最终跑齐，正文中只放已跑齐的，没跑齐的放 future work 或不放。

#### 5.5 Control Analysis

这是顶刊需要的“方法确实可控”的证据。

建议三组图：

- same prompt, different structure maps
- same structure, different style references
- same style reference, different prompts

这能证明三个条件分支各自起作用。

#### 5.6 Benchmark Caveat

这节很重要。不要回避 TIFA 和 T2I-CompBench。

可以写：

公共组合语义 benchmark 上，本文方法并不优于 LoRA-only。这说明本文方法的优势不在通用语义组合能力，而在中国山水画目标域内的结构、风格与保真协同。

这反而会让论文更可信。

### 6. Discussion

讨论三点：

- 为什么中国山水画需要任务特定结构先验
- 为什么单一控制分支不足
- 自动指标能证明什么，不能证明什么

局限性必须写：

- 数据规模仍有限
- 风格类别分布不均衡
- 自动指标不能完全代表艺术评价
- 通用语义 benchmark 不是本文强项

### 7. Conclusion

结论收束到：

- 高保真提升
- 风格一致性提升
- 留白与结构控制提升
- 方法适合作为文化绘画生成的可复现实验框架

## 图设计

### Figure 1: Overall Framework

展示完整 pipeline：

文本提示、结构图、风格参考三路输入，分别进入 LoRA-adapted SDXL、hierarchical adapter、style branch，最后生成中国山水画。

图里一定要突出：

- LoRA 是域适配
- hierarchical structure 是章法/留白先验
- style reference 是笔墨/设色控制

### Figure 2: Hierarchical Structure Representation

展示：

- 原始山水画
- lineart
- quantized depth
- blank-space mask
- salient composition mask

这张图支撑第一个创新点。

### Figure 3: Dataset and Prompt Construction

展示：

- 数据来源
- 清洗过滤
- 结构化 prompt 模板
- train/val/test 划分

这张图支撑实验规范性。

### Figure 4: Main Visual Comparison

每行一个 prompt。

列：

- structure condition
- style reference
- LoRA-only
- ControlNet
- IP-Adapter only
- Ours

### Figure 5: Control Analysis

三组小图：

- same prompt, different structure
- same structure, different style
- same style, different prompt

### Figure 6: Ablation Visualization

展示去掉结构、去掉风格、完整模型的差异。

### Figure 7: Failure Cases

顶刊更喜欢诚实展示失败样例。

可展示：

- 复杂亭台结构失败
- 细长树枝结构断裂
- 留白过强导致主体不足
- 风格参考过强导致内容漂移

## 表设计

### Table 1: Dataset Statistics

字段：

- total images
- train/val/test
- sources
- style labels
- prompt types
- structure channels

### Table 2: Main Quantitative Results

使用 `reports/final_results/main_results_mean_std.csv`。

核心指标：

- FID
- KID
- PickScore
- HPSv2
- LPIPS
- StyleAcc
- EdgeConsistency
- BlankSpaceIoU

`ImageReward` 当前是空值，建议不放进主表。

### Table 3: Statistical Significance

使用：

- `significance_ours_vs_lora.csv`
- `significance_ours_vs_controlnet.csv`
- `significance_ours_vs_ip_adapter_only.csv`

正文里重点讲 `ours vs lora-only`。

### Table 4: Ablation Study

如果实验齐全，放：

- full
- no structure
- no style reference
- LoRA-only
- lineart only
- full four-channel structure

### Table 5: Public Benchmark Results

放 TIFA 和 T2I-CompBench。

这张表不是为了证明方法全面领先，而是为了说明：

**本文方法不是通用语义 benchmark 取向，而是文化绘画域内保真与控制取向。**

### Table 6: Reproducibility Checklist

字段：

- code released
- configs released
- seeds fixed
- metrics scripts released
- result tables released
- dataset construction scripts released

这张表对应用型顶刊很有帮助。

## 当前结果如何支撑贡献

### 支撑贡献一：域内高保真

证据：

- `FID 169.16` vs `LoRA-only 273.40`
- `KID 0.0614` vs `LoRA-only 0.2287`
- 优于 `ControlNet` 和 `IP-Adapter only`

可写结论：

所提结构-风格协同框架显著提升中国山水画目标域的分布拟合能力。

### 支撑贡献二：风格一致性

证据：

- `StyleAcc 0.5783` vs `LoRA-only 0.3373`
- `ours vs lora` 的显著性成立

可写结论：

风格参考分支与 LoRA 域适配协同提高了笔墨风格稳定性。

### 支撑贡献三：多样性

证据：

- `LPIPS 0.4790` vs `LoRA-only 0.3819`

可写结论：

方法没有通过模式收缩换取低 FID，而是在保真度提升的同时保持更高生成多样性。

### 支撑贡献四：结构与留白控制

证据：

- `blank-space IoU` 优于 LoRA-only
- `edge consistency` 明显优于 LoRA-only
- 但弱于 ControlNet 的硬边缘对齐

可写结论：

本文结构先验提升了目标域结构组织能力，但并不追求像 ControlNet 一样的强边缘锁定。

## 顶刊写作边界

建议坚决避免这些表述：

- “our method outperforms all baselines on all metrics”
- “our metric replaces human evaluation”
- “our model has superior general text-to-image understanding”
- “our structure control is stronger than ControlNet”

建议使用这些表述：

- “better in-domain distribution fidelity”
- “stronger style consistency”
- “more balanced structure-style coordination”
- “task-specific controllability for Chinese landscape painting”
- “objective evidence under a unified evaluation protocol”

## 推荐最终摘要逻辑

本文研究中国山水画生成中的结构漂移、留白失控和风格不稳定问题，提出一种融合层次结构先验与风格协同控制的扩散生成框架。该方法以 SDXL 为主干，通过 LoRA 实现领域适配，利用四通道层次结构图编码线稿、伪深度、留白和构图显著区域，并通过风格参考分支增强笔墨与设色一致性。在自建中国山水画 benchmark 上，本文方法相比 `SDXL + LoRA` 将 FID 从 `273.40` 降至 `169.16`，KID 从 `0.2287` 降至 `0.0614`，并在风格准确率和生成多样性上取得稳定提升。进一步的显著性检验表明，所提方法在偏好代理、风格一致性、边缘一致性和留白控制上均具有统计支持。实验同时显示，本文方法并不追求通用语义 benchmark 的全面优势，而是面向中国山水画目标域实现更平衡的结构、风格与保真协同。
