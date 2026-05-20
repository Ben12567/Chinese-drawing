# Reproduce Final Experiments

最终论文主表与显著性分析使用的是 `paper_suite_strong`。

## 入口

统一入口脚本：

`scripts/run_paper_suite.py`

统一 suite 配置：

`configs/experiments/paper_suite_strong.yaml`

## 运行方式

```powershell
$env:PYTHONPATH = "src"
$env:HF_HOME = (Resolve-Path ".").Path + "\.hf"
python scripts/run_paper_suite.py --config configs/experiments/paper_suite_strong.yaml --skip-existing
```

## 这条流水线会做什么

1. 复用已存在的 `seed42` 结果
2. 跑 `ControlNet` 和 `IP-Adapter only`
3. 跑 `Ours` 与 `LoRA-only` 的多训练 seed
4. 汇总 `main_results_raw.csv`
5. 汇总 `main_results_mean_std.csv`
6. 输出 `ours vs baseline` 的显著性结果
7. 自动生成最终结果摘要：
   - `final_results_story.md`
   - `final_results_story.json`

## 最终结果文件

- `outputs/paper_suite_strong/main_results_raw.csv`
- `outputs/paper_suite_strong/main_results_mean_std.csv`
- `outputs/paper_suite_strong/significance_ours_vs_lora.csv`
- `outputs/paper_suite_strong/significance_ours_vs_controlnet.csv`
- `outputs/paper_suite_strong/significance_ours_vs_ip_adapter_only.csv`
- `outputs/paper_suite_strong/final_results_story.md`
- `outputs/paper_suite_strong/final_results_story.json`

## 推荐用于论文的主线

- 以 `Ours vs LoRA-only` 作为主对比
- 用 `ControlNet` 和 `IP-Adapter only` 说明单分支控制的局限
- 强调域内保真度、风格一致性和多样性
- 不把结论扩展到“普适语义控制全面更强”
