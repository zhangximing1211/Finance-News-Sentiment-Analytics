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
├── .env.example          # 环境变量模板
├── .github/workflows/
│   └── ci.yml            # GitHub Actions CI
├── Makefile              # 常用命令快捷入口
├── README.md
├── apps/
│   ├── api/              # FastAPI gateway
│   └── web/              # Next.js dashboard
├── services/
│   ├── model-serving/    # 在线推理（analyzer / LLM reviewer）
│   ├── trainer/          # 数据清洗、训练、评估
│   └── worker/           # 异步任务、review queue、告警、日报/周报
├── packages/
│   ├── schemas/          # Pydantic models & TypeScript types
│   ├── prompts/          # LLM prompt 模板
│   └── utils/            # 通用文本/建模工具
├── data/
│   ├── raw/              # 原始语料
│   ├── interim/          # review_queue.sqlite3 等中间产物
│   └── processed/        # 训练/评估数据与报告
├── notebooks/
│   └── eda/              # 探索性分析 notebook
├── infra/
│   ├── docker/           # Dockerfile & docker-compose
│   └── github-actions/   # CI/CD 脚本与约定
└── tests/
    ├── unit/
    ├── integration/
    └── eval/
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

### 依赖安装

```bash
# Python 服务（API + model-serving + trainer + worker）
python3 -m pip install -r apps/api/requirements.txt
python3 -m pip install -r services/model-serving/requirements.txt
python3 -m pip install -r services/trainer/requirements.txt
python3 -m pip install -r services/worker/requirements.txt

# 共享 schemas 包（editable 模式）
python3 -m pip install -e packages/schemas/python

# Web 前端
npm --prefix apps/web install
```

### API

```bash
uvicorn apps.api.app.main:app --reload --port 8000
# 或
make api
```

### Web

```bash
npm --prefix apps/web run dev
# 或
make web
```

浏览器访问 `http://localhost:3000`，默认连接 `http://localhost:8000` 的 API。

### Trainer

```bash
make audit-data        # 数据审计与切分
make train-baseline    # baseline 训练
make evaluate-baseline # baseline 评估
make train             # 正式训练
```

### Tests

```bash
make test
# 等价于 python3 -m unittest discover -s tests
```

### Review Queue Worker

```bash
make process-review-queue       # 处理 review queue
make review-queue-digest        # 生成 review 摘要
make daily-report               # 日报
make weekly-report              # 周报
make feedback-loop-maintenance  # 自动采样复核 + 周期性再训练检查
```

### Makefile 速查

| 目标 | 说明 |
| --- | --- |
| `make api` | 启动 FastAPI dev server（8000 端口） |
| `make web` | 启动 Next.js dev server（3000 端口） |
| `make test` | 运行全部测试 |
| `make audit-data` | 数据审计与切分 |
| `make train-baseline` | baseline 训练 |
| `make evaluate-baseline` | baseline 评估 |
| `make train` | 正式训练 |
| `make process-review-queue` | 处理 review queue |
| `make review-queue-digest` | review 摘要 |
| `make daily-report` | 日报 |
| `make weekly-report` | 周报 |
| `make feedback-loop-maintenance` | 自动复核采样与再训练检查 |
| `make tree` | 打印目录树 |

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

也可以直接复制 `.env.example` 为根目录 `.env`，服务启动时会自动加载：

```bash
cp .env.example .env
# 编辑 .env 填入你的 OPENAI_API_KEY
```

其中 `LOW_CONFIDENCE_THRESHOLD_OVERRIDE` 和 `NEUTRAL_BOUNDARY_MARGIN_OVERRIDE` 可用于线上或联调时临时覆盖低置信度阈值与 neutral 边界阈值，无需重训模型。

## 当前说明

- `apps/web` 已经改成 Next.js 形态脚手架，但默认未安装 node 依赖
- `apps/api` 已切换为 FastAPI 入口定义，但运行前需要安装 `pydantic`、`fastapi`、`uvicorn`
- 现有训练与推理依然基于经典机器学习与规则引擎，适合最小闭环与后续迭代
- `secondary_explainer` 已支持真实 OpenAI 调用；无 key 时会退回模板解释，不会阻断主链路
- `LLMReviewer` 只在低置信度路径触发，正常高置信度样本不会额外调用外部 LLM
- 数据审计与切分脚本在 `services/trainer/scripts/prepare_data.py`，会生成 `data/processed/*.csv`、`data_description.md` 和 `notebooks/eda/data_audit.ipynb`
- baseline 训练脚本在 `services/trainer/scripts/train_baseline.py`，评估脚本在 `services/trainer/scripts/evaluate.py`

## Docker 部署

项目提供了 Docker Compose 配置，可一键启动 API 和 Web 服务：

```bash
cd infra/docker
docker compose up --build
```

| 服务 | 端口 | Dockerfile |
| --- | --- | --- |
| api | 8000 | `infra/docker/api.Dockerfile` |
| web | 3000 | `infra/docker/web.Dockerfile` |

> `infra/docker/model-serving.Dockerfile` 目前仅供独立构建使用，尚未加入 `docker-compose.yml`。

## CI/CD

GitHub Actions 配置在 `.github/workflows/ci.yml`，在 `main`/`master` push 和 PR 时自动运行：

- Python 3.13 环境
- 安装 `model-serving` 和 `trainer` 依赖
- 执行 `python -m unittest discover -s tests`

> 当前 CI 暂未覆盖 `apps/api`、`services/worker` 的依赖安装，也暂无 Web 前端构建步骤。
