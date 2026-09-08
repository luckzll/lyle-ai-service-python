# Lyle：版本化 Schema 构建与召回

> 2026-09-08：默认新模式已升级为 [B 方案：统一事实与分层召回](MEMORY_B_ARCHITECTURE.md)。下文记录前一版 Schema 链路；其中“问候跳过、小 Schema 不分场景全量、核心等待发布、全量重建”等行为在 B 模式下已替换。以 B 方案说明为准。

## 当前实现

借鉴 Elys 的“原始记忆 → 构建 Schema → 按场景读取槽位”链路，在现有 Python 服务内实现，不另起 Mongo/Loom 服务。Java 绑定当前登录用户；Vue 复用现有记忆页显示整理状态和重试入口。

```text
用户明确陈述 → 抽取 pending 候选 → 用户确认 / 纠正
  → structured_memory + fragment + profile_evidence
  → user_profile（兼容槽位和治理索引）
  → 后台队列 pending → worker 领取任务 building
  → 收集已确认来源，按 Schema 槽位归档、去重、合并
  → 校验路径、值、来源 ID、来源覆盖及完整槽位集合
  → 重新检查来源和任务修订号
  → 原子发布 Schema vN，状态 ready
```

这是独立构建产物，不是聊天时把 profile 行改名为 Schema。构建器以当前确认值为权威，只收集与它一致的 active 支持来源。无证据旧画像标记为 legacy_confirmed_profile 适配；不能把全部历史聊天升级为确认事实。来源文本不是可执行指令，模型不得推测新事实。

核心文件：schema_builder.py（构建与来源校验）、schema_store.py（持久队列/租约/CAS/版本历史）、memory_schema_worker.py（后台执行）、schema_recall.py（读取版本与召回）。

配置 memory 模型且 MEMORY_SCHEMA_BUILD_WITH_LLM=true 时用 LLM 构建；否则确定性归档当前确认值，明确返回 engine=deterministic，不声称调用过 LLM。缺槽位、引用不存在的来源、超预算或模型异常均失败，不发布半份 Schema。结构性来源校验不等于证明所有模型改写都语义正确。

## 聊天触发与召回

```text
准备回答 → Memory Agent 判定
  ├─ skip：空输入、精确短问候 / 确认语 → 不召回个人记忆
  ├─ light：普通问题 → 固定行为约束 + 当前会话上下文
  └─ full：个人信息、历史、项目、计划等线索
       → 已发布 Schema + 当前来源校验
       ├─ auto + 小 Schema：全量选择可用槽位
       ├─ single_pass：一次 LLM 选择路径
       └─ auto + 大 Schema / agent：受限多轮 Agent
             read_schema / search_schema → 工具结果 → finish
       → 加入固定行为约束 → 完整条目字符预算
       → Writer 使用 → 现有 Reviewer 检查
       → usedMemories 记录实际注入字段和原因
```

Agent 能多轮读取当前用户可用 Schema，仅执行只读动作；finish 只能选择已读取路径。未知路径、越界路径、错误协议、步数耗尽均降级为规则选择。默认最多 4 轮、累计读取 8000 字符。single_pass 根据允许路径目录选择，服务端再取值。无 intent 模型时，真实模式为 rule_fallback，并返回原因。

小 Schema 默认按可用槽位 JSON 不超过 3000 字符判定；全量指选槽位，注入仍受总预算限制。selectedPaths 是选择结果，injectedPaths 才是实际注入路径。

固定约束：互动边界、触发点、一般不喜欢的内容、沟通风格、输出风格、语言、语言习惯、社交偏好和社交风格。仍需 active、置信度至少 0.75、非敏感/非控制、来源未变，在调用允许范围及字符预算内，不绕过治理。

## 版本、失效与隐私

- 首次无版本：后台入队，长期槽位为空，返回 pending/building；本轮不等待构建模型，完成后下一轮可用。
- 有版本且重建中：只沿用 inputHash 与当前来源一致的槽位。修改、停用、遗忘、冲突、到期或低置信度字段在下一次读取时过滤。
- 构建期间新修改：修订号不一致，旧任务不得发布。租约到期可由其他 worker 领取，旧 buildId 不得覆盖新结果。
- 默认保留最近 3 个成功版本，失败不覆盖旧版本。永久删除任务同时清理该用户全部 Schema 历史派生文本；剩余合法来源可重建，版本号不倒退。
- 默认新模式不从旧 structured 行或 Qdrant/Milvus 绕路召回长期个人事实。向量写入能力保留供兼容模式使用，不自动清空数据库。
- 当前会话短期消息、摘要仍为对话上下文，不是已确认 Schema；历史文本治理过滤仍有语义匹配边界，不宣称删除原始聊天记录。
- 所有 Schema 槽位目前均为私有；未新增 public 分类或跨用户召回。allowedPaths 只缩小范围，空数组召回零槽位。
- context_digest 标识本次编译上下文；schema_recall.version/etag 标识发布产物，两者不同。

## 接口与运行

Java 接口需登录，用户 ID 来自安全上下文，不能由请求体替换：

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | /memory/schema | enabled、status、version、slotCount、engine、errorCode |
| POST | /memory/schema/rebuild | 提交后台整理/失败重试，不同步等待模型 |
| POST | /memory/schema/recall | 调试指定模式和允许路径的槽位召回 |

Python 对应 /ai/memory/{user_id}/schema...，受现有 internal token 保护。召回请求：

```json
{"question":"帮我规划当前项目","mode":"agent","allowedPaths":["work.current_role","interaction.boundaries"]}
```

返回 schema 元数据、预算内 context、usedMemories，不返回原始会话或内部治理快照。状态读取不触发构建；确认/修改入队，旧用户首次聊天召回按指纹惰性入队，也可在记忆页点“开始整理”。

部署前执行 Python Alembic 到 0029_memory_schema；新表属于 Python PostgreSQL，不是 Java MySQL Flyway。代码保留与现有存储一致的首次 DDL 确保逻辑。POSTGRES_ENABLED=false 仅使用进程内队列/历史，不跨重启或多进程共享；生产持久模式必须启用 PostgreSQL。

deploy/.env.example 已补齐 MEMORY_SCHEMA_* 参数，默认开启。MEMORY_SCHEMA_ENABLED=false 是显式回滚旧召回，不是失败时自动回退。重启 Python 后 worker 开始处理队列，无需单独 Loom 服务。

Vue 不新增页面：显示未整理、正在整理、已整理版本、失败；pending/building 每 3 秒查状态，卸载后停止；失败可重试。槽位记录继续供确认、纠正、停用、遗忘、删除和来源查看使用。

## 验证与边界

tests/test_memory_schema.py 覆盖构建、失效、并发租约、来源引用、模型降级、API 认证和真实 Memory Agent 接入。既有记忆回归显式运行兼容模式，避免混淆旧同步和新异步语义。

2026-09-07：Python 记忆/聊天/Agent/审核专项 192 项通过；前端 167 项单测和生产构建通过；Java 9 项定向测试通过。全量 Python 回归仍有此前 11 项 Windows 文件路径、符号链接权限与外部工具环境失败，1 项 PostgreSQL 条件测试跳过，不等于全量验收通过。

未调用真实付费模型，未连接生产数据库或实际迁移、提交、部署，也未进行浏览器视觉验收。PostgreSQL 锁行为与真实模型质量仍需隔离集成环境验收；并非跨原始存储与 Schema 的分布式原子事务。构建输入超过默认 64000 字符会明确失败，本次未实现超大档案分批构建。
