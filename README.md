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
alembic/                # 数据库迁移
deploy/                 # Dockerfile、环境配置与容器启动脚本
docs/releases/          # 版本发布说明
requirements/optional/  # 可选功能依赖
tests/                  # 自动化测试
```

## 当前接口

- GET /health
- GET /ready（安全配置与向量模型均完成初始化后返回 200）
- POST /ai/chat
- POST /ai/ingest
- GET /ai/ingest/tasks/{task_id}
- POST /ai/agent/chat

`/health` 仅用于进程存活检查；部署流量应以 `/ready` 为准。Docker 镜像构建会预拉取默认 FastEmbed 模型，服务启动时还会执行一次真实向量预热，避免首个用户请求承担模型下载和加载时间。

V6.4 为 Chat、Intent、Memory、Summary、Judge、Embedding、Rerank、图片理解与图片生成接入统一用量上下文。生产环境先把事件写入 PostgreSQL `usage_event_outbox`，后台 worker 再批量上报 Java；无 PostgreSQL 的本地模式才同步降级上报。模型输入、图片内容和 Authorization 不进入用量事件。

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
event: agent.step
event: tool.call
event: reasoning.delta
event: delta
event: meta
event: error
event: done
```

`agent.step` 是实际执行步骤；`reasoning.delta` 仅在主回答模型返回可展示的
`reasoning_content` 或推理摘要时发送。未返回时不生成占位思考，也不会自动启用
模型的思考模式或增加调用次数。摘要和推理内容不混入 `delta` 正文、记忆或用量事件。

```json
{"turnId":"当前轮次 ID","sequence":1,"step":"writer_agent","delta":"模型返回的片段","kind":"reasoning"}
```

`kind` 为 `reasoning` 或 `summary`；`sequence` 在每轮思考流中独立递增，与
`agent.step` 的序号无关。每轮最多接收 16,000 字符，发送前做跨片段凭据脱敏，
保留短尾部缓冲；达到上限时丢弃可能未完成的尾部。前端可将这些事件放在可折叠的
思考与执行区域，正文仍使用原有 `delta`、`meta`、`done` 协议。

`tool.call` 在真实工具执行边界发送，示例：

```json
{"turnId":"当前轮次 ID","sequence":1,"callId":"调用唯一 ID","tool":"weather","title":"查询天气","status":"running","summary":"正在处理工具请求"}
```

每次调用都有独立 `callId`，同工具的重试或再次调用不会合并；序号在工具流内独立递增。
每轮最多展示 128 次调用。`status` 从 `running` 进入一个终态：`completed`、`failed`、
`awaiting_confirmation`、`needs_clarification` 或 `interrupted`，终态附带 `durationMs`。
确认预览只表示等待确认，不表示操作已完成。事件只有工具标识、应用编写的标题和状态，
不包含原始参数、结果、异常或模型自检。图片批处理作为一次工具调用，逐图结果仍走图片事件。
MCP 显示标识使用 `mcp_<id>:<注册工具名>`（仅允许字母、数字、`_.:-`，总长最多 128），
不合法的注册工具名只显示 `mcp_<id>`；执行名称、权限检查和确认流程不变。

## 可选：启用 bge-reranker

默认 `RERANK_PROVIDER=lexical`，不需要额外模型。要启用真正的 bge-reranker：

```bash
/Users/luckzll/Desktop/Lyle/ai-app/ai-service-python/.venv312/bin/python -m pip install -r requirements/optional/reranker.txt
```

然后修改 `deploy/.env`：

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
