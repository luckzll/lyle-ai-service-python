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
- POST /ai/chat
- POST /ai/ingest
- GET /ai/ingest/tasks/{task_id}
- POST /ai/agent/chat

## 本地测试环境

推荐使用 Python 3.12。当前依赖里的 `pydantic-core==2.27.2` 不适合用 Python 3.14 构建；如果系统默认 `python3` 指向 3.14，请显式使用 `.venv312`。

```bash
cd ai-app/ai-service-python
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -r requirements.txt
.venv312/bin/python -m pytest
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
