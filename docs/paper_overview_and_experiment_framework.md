# 论文概述与整体实验框架

## 1. 论文定位

### 1.1 推荐英文标题

**Hierarchical Structure-Guided and Style-Conditioned High-Fidelity Generation for Chinese Landscape Painting**

### 1.2 推荐中文标题

**融合层次结构先验与风格条件的中国山水画高保真可控生成方法**

### 1.3 论文定位建议

本文不宜包装为“纯艺术生成”论文，而应定位为：

- 面向中国山水画的高保真可控生成方法论文
- 面向领域图像生成的结构保持与风格一致性研究
- 面向数字人文与文化视觉计算的任务型生成研究

这样的定位更符合高水平期刊的审稿逻辑，即：

- 问题定义清晰
- 方法设计闭环
- 实验协议严谨
- 结论边界明确

### 1.4 核心研究问题

本文关注如下问题：

1. 通用文生图扩散模型在中国山水画生成中，往往难以同时保持章法结构、留白关系与笔墨风格。
2. 仅依赖文本提示或单纯 LoRA 微调，通常会出现结构漂移、构图失衡和风格不稳定问题。
3. 如何通过结构先验、领域适配与风格参考协同建模，实现中国山水画的高保真可控生成，是本文的核心目标。

---

## 2. 论文整体叙事

### 2.1 建议总述

本文面向中国山水画生成中结构易漂移、风格难稳定和留白难控制的问题，提出一种融合层次结构先验、LoRA 领域适配与风格参考条件的高保真可控扩散生成框架。该方法以 SDXL 为生成主干，通过 LoRA 实现领域内高效适配，利用面向中国山水画构建的层次结构图约束远中近景、线稿与留白布局，并结合风格参考分支增强笔墨风格一致性。在自建中国山水画数据集上，本文建立统一客观评测协议，从分布保真度、文本一致性、结构一致性、风格准确率及感知多样性等多个维度进行比较。实验结果表明，所提方法在 `FID`、`KID`、`PickScore`、`HPSv2`、`LPIPS diversity` 和 `style accuracy` 等指标上均优于 `SDXL + LoRA` 基线，验证了层次结构引导与风格条件协同建模的有效性。

### 2.2 四句式主线

整篇论文建议围绕以下四句话展开：

1. 通用文生图模型难以同时满足中国山水画的章法结构、留白组织和笔墨风格控制。
2. 现有 LoRA 微调虽然能提升领域适配，但对结构保持和风格一致性的控制仍然不足。
3. 为此，本文提出一个由 `LoRA + hierarchical structure adapter + style reference branch` 组成的协同生成框架。
4. 在自建中国山水画数据集和统一客观评测协议上，本文方法在保真度、风格一致性和多样性上均优于 `SDXL + LoRA` 基线。

---

## 3. 创新点设计

建议将创新点固定为三个主创新，避免写得过散。

### 3.1 创新点一：任务特定层次结构先验

本文不是直接套用自然图像的边缘图或深度图，而是构造适用于中国山水画的层次结构表示，显式编码：

- 线稿信息
- 景深层次
- 留白区域
- 构图显著区域

该表示更贴合中国山水画“远中近景”和“虚实相生”的画面组织规律。

### 3.2 创新点二：三条件协同生成机制

本文将以下三类条件统一纳入扩散生成框架：

- 文本语义条件
- 层次结构条件
- 风格参考条件

从而解决单一文本条件下常见的结构漂移和风格不稳问题，实现内容、构图和风格的协同控制。

### 3.3 创新点三：面向中国画的统一客观评测协议

除通用图像生成指标外，本文进一步引入：

- 结构一致性指标
- 留白相关指标
- 风格准确率
- 人类偏好代理模型指标

从而形成更贴近中国山水画任务属性的统一实验协议。

---

## 4. 论文结构建议

### 4.1 Introduction

本章建议围绕以下几个层次展开：

1. 中国山水画在数字人文、文化遗产保护和智能内容生成中的价值。
2. 中国山水画生成与自然图像生成的差异：
   - 远中近景层次更强
   - 留白具有语义功能
   - 风格不仅是颜色纹理，更涉及笔墨组织
3. 通用扩散模型在该领域中的不足：
   - 结构漂移
   - 构图失衡
   - 风格不稳定
4. 本文解决方案概述
5. 本文贡献总结

### 4.2 Related Work

建议拆分为四个小节：

1. Text-to-image diffusion models
2. Controllable generation with adapters, ControlNet and T2I-Adapter
3. Style-conditioned generation and IP-Adapter
4. Chinese painting generation and evaluation

### 4.3 Method

建议按模块组织：

1. Problem formulation
2. Structured prompt construction
3. Hierarchical structure map extraction
4. LoRA-based domain adaptation
5. Hierarchical adapter for structure control
6. Style-reference branch with IP-Adapter
7. Training objective and inference composition

### 4.4 Dataset and Experimental Protocol

本章必须体现高水平期刊所要求的实验严谨性，包括：

1. 数据来源与采集原则
2. 数据清洗、去重和筛选规则
3. 数据划分规则
4. 训练与推理的统一设置
5. 基线方法的公平对齐
6. 评测指标设计

### 4.5 Results and Analysis

建议顺序固定为：

1. Main quantitative comparison
2. Visual comparison
3. Ablation study
4. Structure and style control analysis
5. Generalization and robustness analysis
6. Statistical discussion
7. Limitations

### 4.6 Conclusion

结论只回收三个方面：

- 高保真提升
- 结构控制提升
- 风格一致性提升

---

## 5. 方法概述

### 5.1 整体框架

本文方法以 `SDXL` 为基础生成主干，构建三条件协同生成框架：

- 文本条件：结构化提示词
- 结构条件：层次结构图
- 风格条件：参考风格图像

最终生成高分辨率中国山水画图像。

### 5.2 输入设计

#### 文本提示

采用结构化提示模板，包含：

- 题材对象
- 构图层次
- 远中近景元素
- 笔墨浓淡
- 留白与气韵
- 季节与天气

#### 结构图

结构图由四通道组成：

- `lineart`
- `quantized_depth`
- `blank_space_mask`
- `salient_composition_mask`

#### 风格参考图

风格参考图用于控制如下风格倾向：

- 水墨写意
- 董源式
- 米氏云山式
- 浅绛设色式
- 青绿山水式

### 5.3 模型组成

本文框架由三部分组成：

1. `LoRA` 领域适配分支
2. `Hierarchical Adapter` 结构控制分支
3. `IP-Adapter` 风格参考分支

三者共同作用于 SDXL 的生成过程，实现内容、结构与风格的协同约束。

---

## 6. 数据集与实验协议

### 6.1 数据集构建思路

建议按以下逻辑描述数据集：

1. 从公开馆藏与开放版权来源采集中国山水画图像。
2. 去除边框过重、题跋遮挡严重、分辨率过低和扫描污损明显样本。
3. 对图像进行近重复去重。
4. 为每张图像自动生成：
   - 稠密描述
   - 结构化提示词
   - 四通道层次结构图
   - 风格标签

### 6.2 数据划分原则

建议说明：

- 训练、验证、测试按固定划分进行
- 按来源和相近作品进行分层切分
- 尽量避免相似作品在不同集合间泄漏

### 6.3 当前实验版本说明

当前已完成的正式实验版本基于 `clp4k_v6_union_en` 数据版本进行，适合作为主实验与方法有效性验证基础。

### 6.4 统一训练与推理设置

建议在论文中固定以下设置：

- 基础模型：`stabilityai/stable-diffusion-xl-base-1.0`
- 训练分辨率：`768`
- 推理分辨率：`1024`
- LoRA rank：`8`
- 训练步数：`500`
- 推理步数：`30`
- guidance scale：`7.5`
- 测试集多 seed：`42, 52, 62`

### 6.5 公平性原则

主结果表中所有方法使用：

- 相同测试集
- 相同采样步数
- 相同提示协议
- 相同评测指标

从而保证比较公平。

---

## 7. 整体实验框架

### 7.1 主对比实验

推荐至少包含以下方法：

- `SDXL text-only`
- `SDXL + LoRA`
- `Ours`

如版面允许，可进一步补充：

- `SDXL + ControlNet(lineart)`
- `SDXL + T2I-Adapter`
- `SDXL + IP-Adapter`

### 7.2 评价指标

#### 分布保真度

- `FID`
- `KID`

#### 语义与偏好一致性

- `CLIPScore`
- `PickScore`
- `HPSv2`
- `ImageReward`（若环境稳定可补）

#### 多样性

- `LPIPS diversity`

#### 结构保持

- `edge consistency`
- `blank_space_iou`
- `blank_space_ssim`

#### 风格一致性

- `style accuracy`

### 7.3 主实验结论写法

实验部分建议突出以下叙事：

1. `Ours` 在 `FID` 和 `KID` 上显著优于 `SDXL + LoRA`。
2. `Ours` 在 `PickScore` 和 `HPSv2` 上进一步提升，说明生成结果更接近人类偏好代理模型。
3. `Ours` 在 `LPIPS diversity` 上更高，说明结构与风格协同控制并未显著牺牲多样性。
4. `Ours` 在 `style accuracy` 和结构指标上更优，证明任务特定结构先验与风格参考分支有效。

---

## 8. 消融实验设计

建议消融实验至少包括以下几组：

1. Full model
2. Without hierarchical structure
3. Without style reference
4. Text-only LoRA
5. Short prompt vs structured prompt vs dense prompt
6. Different structure-channel combinations

### 8.1 重点分析点

消融分析建议围绕：

- 结构分支对章法组织的影响
- 风格分支对风格准确率和偏好指标的影响
- 提示形式对文本一致性与生成稳定性的影响

---

## 9. 控制实验与可视化实验

建议设置以下控制实验场景：

### 9.1 同提示词，不同结构图

验证结构分支对构图和留白组织的控制能力。

### 9.2 同结构图，不同风格参考

验证风格参考分支对笔墨风格迁移的有效性。

### 9.3 同内容，不同提示形式

比较：

- short prompt
- structured prompt
- dense prompt

### 9.4 泛化实验

建议加入：

- 未见组合提示
- 诗意提示词
- 季节变化提示
- 稀疏场景与复杂构图场景

---

## 10. 当前真实主结果概述

当前基于正式实验流水线得到的主结果如下：

### 10.1 Ours

- `FID = 164.81`
- `KID = 0.05868`
- `CLIPScore = 0.32797`
- `PickScore = 19.67`
- `HPSv2 = 0.1631`
- `LPIPS diversity = 0.4747`
- `Style accuracy = 0.5542`

### 10.2 SDXL + LoRA

- `FID = 261.36`
- `KID = 0.19544`
- `CLIPScore = 0.32753`
- `PickScore = 19.38`
- `HPSv2 = 0.1379`
- `LPIPS diversity = 0.3706`
- `Style accuracy = 0.3373`

### 10.3 当前结果可支持的主结论

1. 相比 `SDXL + LoRA`，本文方法显著降低了 `FID` 和 `KID`，说明其在分布保真度上具有明显优势。
2. `PickScore` 与 `HPSv2` 的提升表明，本文方法更符合人类偏好代理模型的判断。
3. `LPIPS diversity` 的提升说明本文方法在提高保真度的同时保留了更好的生成多样性。
4. `Style accuracy` 的提升验证了风格条件分支和结构先验协同建模的有效性。

---

## 11. 图表设计建议

### 11.1 图

建议至少包含以下图：

1. 方法总体框架图
2. 数据集示例与结构标注图
3. 层次结构图生成流程图
4. 主对比可视化图
5. 消融可视化图
6. 同结构不同风格控制图
7. 同提示不同结构控制图
8. 失败案例与局限性图

### 11.2 表

建议至少包含以下表：

1. 数据集统计表
2. 主结果对比表
3. 消融实验表
4. 提示形式比较表
5. 控制强度或参数敏感性表
6. Benchmark 扩展结果表

---

## 12. 高水平期刊写作边界

### 12.1 推荐表述

建议使用如下关键词：

- high-fidelity generation
- controllable generation
- structure preservation
- style-consistent synthesis
- objective superiority under a unified benchmark

### 12.2 不推荐表述

应避免写成：

- 达到人类艺术家水平
- 完全替代人工评价
- 艺术价值优于人工作品

### 12.3 更稳妥的结论表述

建议写为：

- 本文方法在统一客观评测协议下取得了显著优于基线的方法结果。
- 本文方法在结构保持、风格一致性与分布保真度之间实现了更优平衡。
- 现有结果验证了层次结构先验与风格条件协同建模的有效性。

---

## 13. 可直接扩写的结论模板

本文提出一种融合层次结构先验、LoRA 领域适配与风格参考条件的中国山水画高保真可控生成框架。与仅依赖文本条件或单一 LoRA 适配的基线相比，所提方法能够更有效地约束章法结构、留白组织与风格表达，从而在保真度、风格一致性和生成多样性上取得更优结果。基于统一测试协议和多种客观指标的实验表明，本文方法在 `FID`、`KID`、`PickScore`、`HPSv2`、`LPIPS diversity` 和 `style accuracy` 等方面均优于 `SDXL + LoRA` 基线。上述结果说明，面向中国山水画的结构先验建模与风格条件控制是实现高保真可控生成的关键因素。

