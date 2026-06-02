# Scientific Reports 排版说明

这个目录用于整理 Scientific Reports 投稿版 LaTeX。当前文件只做排版组织，不改写论文内容。

## 文件

- `scirep_main_clean.tex`: 按 Scientific Reports 风格整理后的主文件。
- `figures/figure4_qualitative_comparison.pdf`: Figure 4 主对比图。
- `figures/figure5_structure_controllability.pdf`: Figure 5 结构可控图。
- `figures/figure6_quantitative_summary.pdf`: 定量结果总览图。
- `figures/figure7_tradeoff_and_significance.pdf`: trade-off 与显著性图。
- `editable_figures_4_5.pptx`: Figure 4 和 Figure 5 的 PowerPoint 可编辑版本。
- `ppt_previews/`: PowerPoint 实际渲染后的 PNG 预览。

## 可编辑 PPT 图

`editable_figures_4_5.pptx` 包含两页：

- 第 1 页：Figure 4 主视觉对比图。
- 第 2 页：Figure 5 结构可控性图。

PPT 中的列标题、prompt、短标签、边框、虚线框和注释标签均为可编辑 PowerPoint 元素。山水画生成结果作为原始图像嵌入，没有删除或修改画作内部的题跋、印章和模型生成内容。

重新生成 PPT：

```powershell
python scripts/make_editable_ppt_figures.py
```

## 需要你粘入的内容

把当前稿件正文按下面位置粘入 `scirep_main_clean.tex`：

- Abstract 粘到 `\begin{abstract}` 内。
- Introduction 粘到 `\section*{Introduction}` 下。
- Related Work 粘到 `\section*{Related Work}` 下。
- Method 粘到 `\section*{Culture-Aware Hierarchical Diffusion Framework}` 下。
- Experiments and Results 粘到 `\section*{Experiments and Results}` 下，但表格和 Figure 4--7 建议直接用文件中已排好的 float blocks。
- Discussion 粘到 `\section*{Discussion}` 下。

## 原稿中建议清理的排版问题

- 删除模板说明文字，例如 “Please note...”, “Example text under a subsection...”, “Figures and tables can be referenced...”。
- 删除重复章节标题：原稿里同时有 `Experiment` 和 `Experiments and Results`，以及重复的 `Discussion`。
- 删除模板示例图表 `stream` 和 example table。
- Figure 4--7 已按正文引用文件名复制到 `paper/figures/`，不需要改正文里的 `\includegraphics`。
- Figure 1 和 Figure 2 当前没有对应图片文件，建议先注释，等图做好后再启用。
- Scientific Reports 正文一般使用无编号章节：`\section*{...}`、`\subsection*{...}`、`\subsubsection*{...}`。
- 表格建议使用 `booktabs` 的 `\toprule`、`\midrule`、`\bottomrule`，不要用密集竖线。
- 大图和宽表使用 `figure*` / `table*`，普通表使用 `table`。

## 编译前还缺什么

当前仓库没有 `wlscirep.cls` 和完整 TeX Live 环境。你需要从 Scientific Reports 官方 LaTeX 模板中放入：

- `wlscirep.cls`
- `sample.bib` 或你的正式 `.bib` 文件
- Figure 1 和 Figure 2 图片文件，如果正文中启用它们

推荐编译命令：

```powershell
pdflatex scirep_main_clean.tex
bibtex scirep_main_clean
pdflatex scirep_main_clean.tex
pdflatex scirep_main_clean.tex
```

如果使用 Overleaf，把 `scirep_main_clean.tex`、`figures/`、`wlscirep.cls` 和 `.bib` 一起上传即可。
