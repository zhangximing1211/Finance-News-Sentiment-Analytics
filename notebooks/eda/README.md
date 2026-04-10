# EDA Notebook Area

这个目录用于放探索性分析 notebook，例如：

- 标签分布分析
- 高频事件关键词检查
- 中文与英文文本长度分布
- 误判案例回溯

## 当前 notebook

- `data_audit.ipynb` — 由 `services/trainer/scripts/prepare_data.py` 自动生成的数据审计报告，包含标签分布、文本长度直方图、清洗规则统计等可视化。

## 生成方式

```bash
make audit-data
# 会同时生成 data/processed/*.csv、data/processed/data_description.md 以及本目录下的 data_audit.ipynb
```
