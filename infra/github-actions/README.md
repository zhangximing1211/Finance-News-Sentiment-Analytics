# GitHub Actions Notes

这个目录预留给可复用的 GitHub Actions 片段、脚本和组织级约定。

当前仓库把实际 workflow 放在：

- `.github/workflows/ci.yml` — 在 `main`/`master` push 和 PR 时自动运行 Python 测试

### 当前 CI 覆盖范围

- Python 3.13
- 安装 `services/model-serving` 和 `services/trainer` 的依赖
- `python -m unittest discover -s tests`

### 待补充

- `apps/api` 和 `services/worker` 的依赖安装
- `packages/schemas/python` 的 editable 安装
- Web 前端构建与 lint（`npm --prefix apps/web run build`）
- coverage 上传
- 模型评估报告归档
- Docker 镜像构建与发布
