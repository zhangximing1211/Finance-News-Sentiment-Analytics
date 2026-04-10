# finance-schemas

Pydantic models shared by the finance sentiment agent services and apps.

## 安装

```bash
# editable 模式安装（推荐本地开发）
python3 -m pip install -e packages/schemas/python
```

依赖：`pydantic>=2.8,<3.0`，Python >=3.11。

## 主要模型

| 模型 | 说明 |
| --- | --- |
| `AnalyzeRequest` | 单条分析请求，包含文本与可选上下文 |
| `BatchAnalyzeRequest` | 批量分析请求（`texts` 或 `items`） |
| `AnalyzeResponse` | 完整分析响应 |
| `SentimentResult` | 情绪标签、置信度、概率分布 |
| `EventResult` | 事件类型与匹配信号 |
| `Entities` | 公司名、ticker、行业 |
| `RiskAlert` | 人工复核建议与原因 |
| `ReviewQueueItem` | review queue 条目 |
| `AnalysisContext` | 新闻来源、公司、历史公告等分析上下文 |

## 用法

```python
from finance_schemas import AnalyzeRequest, AnalyzeResponse

req = AnalyzeRequest(text="Company X reported record revenue...")
```

所有模型定义在 `finance_schemas/models.py`，通过 `finance_schemas/__init__.py` 统一导出。
