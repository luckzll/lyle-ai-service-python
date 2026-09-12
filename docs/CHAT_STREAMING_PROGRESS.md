# 聊天实时进度、计时与断流恢复

此实现跨 Python AI 服务、Java 后端、Vue 前端。检索结果与查询顺序保持原有策略。

## 进度协议

`POST /ai/chat/stream` 在准备上下文前建立 SSE 响应，阻塞工作在有并发上限的独立线程执行。空闲时每 10 秒发送注释心跳 `: keep-alive`。心跳只说明连接存活，不代表任务成功。

真实节点开始、结束、跳过或失败时发送 `agent.step`：

```json
{
  "turnId": "当前轮次",
  "sequence": 1,
  "step": "rag_agent",
  "status": "completed",
  "durationMs": 120,
  "summary": "检索 1 次，命中 3 个片段",
  "timings": {"vector_search_ms": 25}
}
```

- 节点：`supervisor_agent`、`memory_agent`、`rag_agent`、`tool_agent`、`gather_agent`、`project_resolver`、`writer_agent`、`reviewer_agent`。
- 状态：`running`、`completed`、`skipped`、`failed`。尚未收到事件的节点显示待开始，不能由空字段推断已经运行。
- `sequence` 在每轮内递增。Java/Vue 校验轮次，保留每个节点的最新事件，防止历史快照覆盖新事件。
- Java 保存 `metadata.agentProgress`；最终 `meta/agentTrace` 为独立的完整证据快照，不能覆盖进度记录。
- 摘要只用应用生成的状态文案，计时只接受限定字段与有限非负数；不加入模型提示词或供应商错误原文。

## 耗时含义

知识库记录查询改写、各查询召回、向量化、向量搜索、关键词搜索、合并、相邻内容补充、重排和资料充分性判断等耗时。

`writer_first_token_ms` 是调用回答模型到收到第一个非空文本块的等待时间；`first_delta_ms` 是本轮 Python 请求开始到转发第一段正文的时间。前者不包含之前的路由、检索等待，二者不能混用。`writer_ms` 是回答生成耗时；`prepare_context_ms` 是准备上下文耗时。`review_ms` 包含回答后处理与检查。

父阶段包含子阶段，多个 Agent 还可能并行执行，所有计时不能直接相加。它们用于定位瓶颈，不是提速基准或浏览器实际渲染耗时。

## 提前输出正文的边界

模型全文审核、知识库引用/资料不足检查、记忆治理、明确记忆写入/概览、工具、项目候选与附件相关场景保留完整回答检查。

其余回答可增量输出：沿用本地内容规则，保留短尾部与未闭合的图片 Markdown，发送前检查。增量校验明确关闭模型审核调用，不会每块调用审核模型；审核开关与重试配置按请求隔离。已输出的内容不能被全文改写覆盖；后续检查失败时保留已显示部分并报告失败。

## 恢复与生命周期

Java 后台生成与浏览器订阅独立。慢客户端队列有容量上限，断开订阅后任务继续。正文首段及后续约每秒落库，最终保存完成后才发送 `done`。

Vue 遇到断流或没有 `done` 的 EOF，先按 `turnId` 查询后台：活动轮次继续轮询，终态恢复保存结果，暂时查不到时显示待确认并退避重查。不会自动重新发起模型生成。明确失败或用户成功停止后不继续显示节点正在执行。

Python 连接真正断开时，响应生命周期主动停止流桥；正在执行的供应商调用仍占用并发槽直到返回或其已有超时生效。没有新增整轮任务硬截止时间。

## 验证入口

- Python：`test_chat_stream_progress.py`、`test_streaming_response.py`、`test_chat_progress.py`、`test_rag_timings.py`、`test_runtime_config.py` 以及原有聊天/审核/附件回归。
- Java：`ChatStreamProgressTest`、`AiServiceClientStreamTest`、`ChatControllerTest` 与聊天服务/元数据回归。
- Vue：聊天进度、恢复、SSE、历史记录与组件 SSR 测试，以及生产构建。

测试使用受控模型/存储替身。部署后的真实首字、检索和总耗时需要另行采样，不能由单元测试推算。
