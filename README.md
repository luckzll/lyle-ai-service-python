# ai-service-python

Python AI 服务，采用 FastAPI。

## 目录结构

```text
app/
  main.py
  api/       # HTTP 路由
  core/      # 配置
  rag/       # 检索增强与临时内存知识库
  llm/       # 大模型调用
  chains/    # LangChain 调用封装
  graphs/    # LangGraph 智能体流程
  embedding/ # 向量生成
  workers/   # 文档入库任务
  schemas/   # 请求响应模型
```

## 当前接口

- GET /health
- GET /ready（安全配置与向量模型均完成初始化后返回 200）
- POST /ai/chat
- POST /ai/ingest
- GET /ai/ingest/tasks/{task_id}
- POST /ai/agent/chat

`/health` 仅用于进程存活检查；部署流量应以 `/ready` 为准。Docker 镜像构建会预拉取默认 FastEmbed 模型，服务启动时还会执行一次真实向量预热，避免首个用户请求承担模型下载和加载时间。

## 执行体验

执行体验在工作流与自动化之上增加受控模板、草稿、首次巡检、执行结果与确认收件箱。对外接口统一由 Java 登录态代理，Python 路由仅接收已注入的 `user_id`：

- `GET /ai/experience/templates/{user_id}`：只读结果型模板目录。
- `POST /ai/experience/drafts/{user_id}/from-template`、`from-intent`：仅创建草稿，不执行。
- `POST /ai/experience/drafts/{user_id}/{draft_id}/enable`：物化为既有工作流与自动化。
- `GET /ai/experience/execution/{user_id}/results`：用户可读的执行摘要。
- `GET /ai/experience/confirmations/{user_id}`：复用既有待确认操作状态机。

首次部署执行体验前运行数据库迁移：

```bash
cd ai-app/ai-service-python
.venv312/bin/alembic upgrade head
```

迁移会新增草稿、引导、结果摘要与确认收件箱表，并仅为既有 MCP 连接新增可空的 `secret_ref`；不会删除或重写工作流运行和审计数据。

外部 MCP 凭据使用运行环境引用，而非 JSON 明文。例如在部署环境设置 `LYLE_GITHUB_TOKEN`，连接配置使用 `secretRef: "env:LYLE_GITHUB_TOKEN"`，并在授权 Header 或进程环境变量中使用 `{{secret_ref}}`。列表和前端响应只会显示“已配置凭据”。

## 本地测试环境

项目统一且仅支持 Python 3.12。仓库的 `.python-version`、应用启动保护和 Docker 镜像都会阻止误用 Python 3.14。`requirements.txt` 维护直接依赖，`requirements.lock` 固定完整依赖树；本地、CI 与 Docker 均应优先使用 lock 文件，避免 MCP SDK、Starlette 与 Pydantic 因 Python 条件依赖发生漂移。

```bash
cd ai-app/ai-service-python
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -r requirements.lock
.venv312/bin/python -m pytest
```

直接依赖变化后，用固定版本的 `pip-tools` 重新生成 lock：

```bash
.venv312/bin/python -m pip install pip-tools==7.5.3
.venv312/bin/pip-compile --resolver=backtracking --strip-extras --allow-unsafe --output-file=requirements.lock requirements.txt
```

如果本机已经存在 `.venv312`，可以直接运行：

```bash
cd ai-app/ai-service-python
.venv312/bin/python -m pytest
```

## 当前 RAG 状态

```text
Java 上传文件
  -> Python /ai/ingest
  -> Qdrant 知识库集合
  -> Query Rewrite / Multi-query Retrieval
  -> Reranker
  -> Sufficiency Check
  -> LangGraph 多 Agent 回答
```

## LangChain / LangGraph

当前已经加入：

- `langchain`
- `langchain-openai`
- `langgraph`

当前主聊天接口已经走 LangGraph 多 Agent：

```text
Supervisor -> Memory -> RAG -> Tool -> Writer -> Reviewer -> Memory Write
```

流式接口 `/ai/chat/stream` 使用 SSE：

```text
event: delta
event: meta
event: error
event: done
```

## 可选：启用 bge-reranker

默认 `RERANK_PROVIDER=lexical`，不需要额外模型。要启用真正的 bge-reranker：

```bash
/Users/luckzll/Desktop/Lyle/ai-app/ai-service-python/.venv312/bin/python -m pip install -r requirements-reranker.txt
```

然后修改 `.env`：

```env
RERANK_ENABLED=true
RERANK_PROVIDER=flag_embedding
RERANK_MODEL=BAAI/bge-reranker-base
RERANK_USE_FP16=false
RERANK_FALLBACK_ENABLED=true
```

如果未安装依赖或模型加载失败，会自动退回 `lexical`，除非设置：

```env
RERANK_FALLBACK_ENABLED=false
```

## 文档解析

当前 MVP 支持文本类文件：

- `.txt`
- `.md`
- `.csv`
- `.json`
- `.log`
- `.pdf`
- `.docx`

上传文件由 Java 后端写入本地 `ai-app/storage/`，Python 通过 `file://` 路径读取内容并切分到内存 RAG。旧版 `.doc` 暂不支持，建议转换为 `.docx`。
