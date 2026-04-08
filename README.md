# finance-sentiment-agent

面向金融新闻与公告理解的 monorepo 仓库。

## 仓库目标

用户输入一段新闻、公告或 filing，系统返回：

- 情绪标签：`positive / neutral / negative`
- 置信度
- 事件类型：财报 / 收购 / 裁员 / 合同 / 产能 / 价格变动 / 指引更新
- 实体：公司名、ticker、行业
- 一句解释
- 风险提示：是否建议人工复核

## Agent 工作流

当前主链路已经接成 agent workflow：

1. 接收文本
2. 抽取实体与事件
3. 跑 sentiment model
4. 低置信度时触发 LLM 复判
5. 生成简洁解释
6. 写入数据库
7. 根据规则决定是否告警
8. 聚合为日报/周报

## 当前结构

```text
finance-sentiment-agent/
├── apps/
│   ├── web/
│   └── api/
├── services/
│   ├── trainer/
│   ├── model-serving/
│   └── worker/
├── packages/
│   ├── schemas/
│   ├── prompts/
│   └── utils/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   └── eda/
├── infra/
│   ├── docker/
│   └── github-actions/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── eval/
├── .github/workflows/
├── README.md
└── Makefile
```

## 职责划分

- `apps/web`: Next.js dashboard，面向分析输入与结果展示
- `apps/api`: FastAPI gateway，负责 HTTP 接入、schema 校验、路由编排
- `services/model-serving`: 在线推理逻辑，封装情绪、事件、实体与人工复核建议
- `services/trainer`: 数据清洗、训练、评估与报告输出
- `services/worker`: 异步任务、review queue、告警和日报/周报聚合
- `data/interim/review_queue.sqlite3`: review queue、agent runs、alerts、report snapshots 的 SQLite 持久化库
- `packages/schemas`: Python Pydantic models 与 TypeScript types
- `packages/prompts`: LLM 解释、人工复核等 prompt 模板
- `packages/utils`: 通用文本和建模工具
- `data`: 原始数据、中间产物、训练报告
- `tests`: 单元、集成与评估测试

## 现在保留的业务核心

这次重构保留了现有 MVP 的核心分析能力，并迁移到：

- `services/model-serving/src/model_serving/analyzer.py`
- `services/model-serving/src/model_serving/service.py`

训练语料已迁移到：

- `data/raw/all-data.csv`

## 本地开发

### API

```bash
python3 -m pip install -r apps/api/requirements.txt
uvicorn apps.api.app.main:app --reload --port 8000
```

### Trainer

```bash
python3 -m pip install -r services/trainer/requirements.txt
python3 services/trainer/scripts/prepare_data.py
python3 services/trainer/scripts/train_baseline.py
python3 services/trainer/scripts/evaluate.py --split test
python3 services/trainer/scripts/train.py
```

### Tests

```bash
python3 -m unittest discover -s tests
```

### Review Queue Worker

```bash
python3 -m pip install -r services/worker/requirements.txt
python3 services/worker/jobs/process_review_queue.py --limit 20
python3 services/worker/jobs/daily_digest.py
python3 services/worker/jobs/generate_agent_report.py --report-type daily
python3 services/worker/jobs/generate_agent_report.py --report-type weekly
python3 services/worker/jobs/run_feedback_loop_maintenance.py --report-type weekly
```

## Review Queue 与 LLM 解释

- `POST /api/analyze` 在命中人工复核条件时，会把 queue item 持久化到 `data/interim/review_queue.sqlite3`
- `GET /api/review-queue` 返回当前队列项
- `GET /api/review-queue/summary` 返回状态、优先级和复核原因汇总
- `GET /api/alerts` 返回 agent workflow 触发的告警
- `GET /api/reports/daily` 和 `GET /api/reports/weekly` 返回按数据库聚合的日报/周报
- `services/worker/jobs/process_review_queue.py` 会从 SQLite 里 claim `pending` 项，调用真实 LLM 补全二次解释后把状态更新为 `ready_for_review`
- `POST /api/analyze` 已改为走 `AgentWorkflowService`，分析结果会在 `metadata.agent_workflow` 记录最终决策、LLM 复判、告警和 workflow steps
- 如果没有配置 OpenAI 凭证，`secondary_explainer` 会保留模板兜底，并把 queue item 维持为待补全状态

## 上线后反馈闭环

- `GET /api/reports/daily` 和 `GET /api/reports/weekly` 现在会返回 `monitoring` 与 `feedback_loop_assets`
- `monitoring` 包含每日推理量、低置信度比例、类别漂移、用户纠错率、每类 precision / recall / F1、新闻源分布偏移
- `feedback_loop_assets` 包含自动采样复核数、错误样本池规模、黄金测试集规模、打开的重训任务数，以及是否触发周期性再训练
- `GET /api/error-samples` 返回错误样本池
- `POST /api/golden-set` 与 `GET /api/golden-set` 用于维护黄金测试集
- `POST /api/feedback-loop/maintenance` 会执行自动采样复核，并在满足阈值时自动创建周期性重训任务
- `services/worker/jobs/run_feedback_loop_maintenance.py` 提供 CLI 入口，可挂到 cron 或调度器

## 产品页面

- `/`: 总览页，展示日报指标、最新告警、错误池入口和产品导航
- `/analyze`: 单条文本分析页
- `/batch`: 批量分析页
- `/watchlist`: 公司 watchlist 页
- `/trends`: 情绪趋势页
- `/errors`: 错误案例回看页
- `/feedback`: 人工反馈标注页

## 核心 API

- `POST /analyze`: 单条文本分析
- `POST /batch-analyze`: 批量文本分析
- `GET /results`: 查询结果库，支持 `label / event_type / entity_query / source / error_only / watchlist_only`
- `POST /watchlist`: 新增或更新 watchlist 项
- `GET /watchlist`: 查询 watchlist 列表
- `GET /alerts`: 查询告警，支持 `watchlist_only`
- `POST /feedback`: 提交人工反馈
- `GET /feedback`: 查询历史反馈
- `POST /retrain`: 创建重训任务
- `GET /error-samples`: 查询错误样本池
- `POST /golden-set`: 新增黄金测试样本
- `GET /golden-set`: 查询黄金测试集
- `POST /feedback-loop/maintenance`: 执行自动复核采样与周期性再训练检查

上述接口同时保留了 `/api/*` 兼容路由，便于现有脚本和平滑迁移。

### OpenAI 配置

服务端的二次解释层使用 OpenAI Responses API，并通过 Structured Outputs 要求模型返回稳定 JSON。运行前可配置：

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
export OPENAI_TIMEOUT_SECONDS=30
# 可选
export OPENAI_ORGANIZATION=...
export OPENAI_PROJECT=...
export LOW_CONFIDENCE_THRESHOLD_OVERRIDE=0.6
export NEUTRAL_BOUNDARY_MARGIN_OVERRIDE=0.08
export REVIEW_QUEUE_DB_PATH=data/interim/review_queue.sqlite3
```

也可以直接复制 [.env.example](/Users/zhangximing/Desktop/Finance%20News%20Sentiment%20Analytics/.env.example) 为根目录 `.env`，服务启动时会自动加载。

其中 `LOW_CONFIDENCE_THRESHOLD_OVERRIDE` 和 `NEUTRAL_BOUNDARY_MARGIN_OVERRIDE` 可用于线上或联调时临时覆盖低置信度阈值与 neutral 边界阈值，无需重训模型。

## 当前说明

- `apps/web` 已经改成 Next.js 形态脚手架，但默认未安装 node 依赖
- `apps/api` 已切换为 FastAPI 入口定义，但运行前需要安装 `pydantic`、`fastapi`、`uvicorn`
- 现有训练与推理依然基于经典机器学习与规则引擎，适合最小闭环与后续迭代
- `secondary_explainer` 已支持真实 OpenAI 调用；无 key 时会退回模板解释，不会阻断主链路
- `LLMReviewer` 只在低置信度路径触发，正常高置信度样本不会额外调用外部 LLM
- 数据审计与切分脚本在 `services/trainer/scripts/prepare_data.py`，会生成 `data/processed/*.csv`、`data_description.md` 和 `notebooks/eda/data_audit.ipynb`
- baseline 训练脚本在 `services/trainer/scripts/train_baseline.py`，评估脚本在 `services/trainer/scripts/evaluate.py`
