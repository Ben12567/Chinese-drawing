# 中国山水画生成结果专家盲评表

## 评审说明

- 请根据图像本身进行评分，不提供方法名称。
- 所有样本随机混排，建议在同一显示设备上完成评价。
- 每个维度采用 `1-5` 分 Likert 量表：
  - `1` 分：很差
  - `2` 分：较差
  - `3` 分：一般
  - `4` 分：较好
  - `5` 分：优秀
- 若有特殊观察，请在备注中简要说明。

## 评分维度

- `PromptConsistency`：生成结果与提示词在题材、季节、景物和意境上的一致性
- `CompositionStructure`：章法布局、远中近景层次、留白处理与画面组织质量
- `BrushworkCharm`：笔墨表现、皴擦点染韵味和中国画风格特征
- `ArtisticQuality`：整体艺术完成度、审美协调性和视觉感染力
- `Creativity`：在保持中国画特征的前提下是否体现出合理的新意

## 记录格式

请将评分录入到 `templates/tables/expert_scores.csv`，字段如下：

```csv
Method,SampleID,RaterID,PromptConsistency,CompositionStructure,BrushworkCharm,ArtisticQuality,Creativity,Comment
```

## 盲评建议流程

1. 从每个方法中随机抽取相同数量的测试样本。
2. 对所有图像重新命名并随机打乱顺序。
3. 保证每位评审者看到的图像顺序不同或经过充分随机化。
4. 评分完成后使用 `scripts/analyze_expert_scores.py` 计算均值、标准差、`Cronbach's alpha` 和 `ICC(2,1)`。
