from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 从 ai-service-python/.env 读取环境变量；没有配置时使用下面的默认值。
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ===== 大模型基础配置 =====
    # DeepSeek API Key 从 .env 注入，不要写死在代码里。
    deepseek_api_key: str = ""
    # DeepSeek OpenAI-compatible 接口地址。
    deepseek_base_url: str = "https://api.deepseek.com"
    # 兼容旧配置：默认模型名。
    llm_model: str = "deepseek-v4-flash"
    # 兼容旧配置：默认接口地址。
    llm_base_url: str = "https://api.deepseek.com"
    # 大模型请求超时时间，单位：秒。
    llm_timeout_seconds: float = 60

    # ===== 多模型供应商配置 =====
    # OpenAI、Qwen、Ollama 是给后续多模型路由预留的供应商配置。
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ollama_base_url: str = "http://localhost:11434/v1"
    ark_api_key: str = ""
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"

    # ===== 多模型任务路由 =====
    # chat：主回答模型。
    # 推荐：开发/性价比用 deepseek-chat 或 deepseek-v4-flash；高质量回答后续可切 GPT/Claude。
    chat_model_provider: str = "deepseek"
    chat_model_name: str = "deepseek-v4-flash"
    # intent：意图识别模型，用于判断聊天、记忆、知识库等任务类型。
    # 推荐：低成本用本地 qwen2.5:3b/7b；简单稳定先用 DeepSeek。
    intent_model_provider: str = "deepseek"
    intent_model_name: str = "deepseek-v4-flash"
    # memory：结构化记忆抽取模型。
    # 推荐：中文结构化抽取后续切 qwen-plus/qwen-turbo；当前先用 DeepSeek。
    memory_extract_model_provider: str = "deepseek"
    memory_extract_model_name: str = "deepseek-v4-flash"
    # summary：长期记忆压缩模型。
    # 推荐：中文摘要后续切 qwen-plus；性价比可继续用 DeepSeek。
    summary_model_provider: str = "deepseek"
    summary_model_name: str = "deepseek-v4-flash"
    # judge：质量评估模型，后续 Reviewer Agent 使用。
    # 推荐：高质量 Judge 用 gpt-4o-mini/gpt-4.1-mini；中文低成本用 qwen-plus。
    judge_model_provider: str = "deepseek"
    judge_model_name: str = "deepseek-v4-flash"
    # vision：图片识别/图片问答模型，作为 Tool Agent 的 image_analyze 工具使用。
    vision_model_provider: str = "ark"
    vision_model_name: str = "doubao-seed-2-0-pro-260215"
    # image：文生图模型，作为 Tool Agent 的 image_generate 工具使用。
    image_model_provider: str = "ark"
    image_model_name: str = "doubao-seedream-4-5-251128"
    # ===== Reviewer Agent：最终答案质量审核 =====
    # 开启后 Writer 生成答案后会先审核，不通过时最多重写 reviewer_max_retry 次。
    reviewer_enabled: bool = False
    reviewer_max_retry: int = 1
    # ===== Evaluation：批量评测 =====
    # 开启后 evaluate 接口会额外调用 Judge 模型，和规则分加权融合。默认关闭，避免评测时产生额外模型费用。
    evaluation_judge_enabled: bool = False
    # ===== Tool Agent：外部/内部工具调用 =====
    # 开启后 Supervisor 会按问题判断是否调用工具。
    tool_agent_enabled: bool = True
    tool_router_llm_enabled: bool = True
    tool_router_context_max_chars: int = 3000
    tool_router_confidence_threshold: float = 0.65
    # 允许调用的工具白名单，逗号分隔。生产环境建议只开放明确需要的工具。
    tool_allowed_names: str = "health_check,memory_snapshot,list_tools,memory_stats,work_summary,web_search,weather,file_parse,image_analyze,image_generate,image_edit,todo,calendar"
    # 工具调用审计。开启后会写日志；PostgreSQL 可用时也会写 tool_audit_log。
    tool_audit_enabled: bool = True
    # 只读数据库统计工具。只允许查询本项目记忆表统计，不执行用户传入 SQL。
    tool_database_query_enabled: bool = True
    # 网页搜索工具。默认关闭，配置 provider 后再接真实搜索服务。
    tool_web_search_enabled: bool = False
    tool_web_search_provider: str = ""
    tool_web_search_api_key: str = ""
    # Optional provider endpoint override for an enterprise gateway or deterministic E2E fixture.
    tool_web_search_base_url: str = ""
    tool_web_search_top_k: int = 5
    # 单条网页搜索摘要长度上限，避免工具上下文过长污染回答。
    tool_web_search_snippet_max_chars: int = 240
    # 网页搜索结果缓存时间，单位：秒。开发期可以省 API 额度，生产可按需要调小。
    tool_web_search_cache_ttl_seconds: int = 600
    # 天气工具：按需查询，不默认注入每轮上下文。使用免 Key Open-Meteo。
    tool_weather_enabled: bool = True
    tool_weather_cache_ttl_seconds: int = 600
    # 文件解析工具：仅允许解析本地 file/path 或 MinIO s3:// 路径，不主动下载任意外部 URL。
    tool_file_parse_enabled: bool = True
    tool_file_parse_max_chars: int = 4000
    # file_parse 工具允许读取的本地目录白名单，逗号分隔；相对路径基于 Python 服务运行目录。
    # 用户问题中的本地路径只有落在这些目录内才会被解析，防止任意本地文件读取（LFI）。
    tool_file_parse_allowed_dirs: str = "../storage"
    # 待办和日历工具：写入 PostgreSQL 的 agent_todo / agent_calendar_event 表。
    tool_todo_enabled: bool = True
    tool_calendar_enabled: bool = True
    # 图片工具：图片理解和文生图。需要配置 ARK_API_KEY 或对应 provider key。
    tool_image_analyze_enabled: bool = True
    tool_image_generate_enabled: bool = True
    tool_image_edit_enabled: bool = True
    # 单批最多生成 8 张，并允许模型 API 请求高并发。两个值均可通过环境变量覆盖。
    image_batch_max_count: int = 8
    image_batch_concurrency: int = 8
    image_batch_retry_attempts: int = 2
    image_batch_retry_base_seconds: float = 0.6
    image_reference_max_count: int = 8
    chat_attachment_max_count: int = 8
    chat_attachment_max_file_bytes: int = 5 * 1024 * 1024
    chat_attachment_max_total_bytes: int = 20 * 1024 * 1024
    chat_attachment_text_max_chars_per_file: int = 50000
    chat_attachment_text_max_chars_total: int = 120000
    # V3 轻量连接器：本地文档只允许读取白名单目录；GitHub 先支持只读能力。
    connector_local_docs_roots: str = "../.."
    connector_local_docs_max_files: int = 200
    connector_local_docs_max_chars: int = 1200
    connector_github_token: str = ""
    connector_github_api_base_url: str = "https://api.github.com"
    # Seedream 图片生成/编辑是否添加可见水印。默认关闭。
    image_watermark: bool = False
    generated_image_storage: str = "local"
    generated_image_dir: str = "../.local/generated-images"
    generated_image_public_base_url: str = "http://127.0.0.1:8000/generated-images"
    tosutil_path: str = "tosutil"
    tos_access_key_id: str = ""
    tos_secret_access_key: str = ""
    tos_endpoint: str = ""
    tos_region: str = "cn-shanghai"
    tos_bucket: str = ""
    tos_save_path_head: str = "zl-dev/"
    tos_public_url_head: str = ""
    tos_upload_timeout_seconds: int = 60
    # 后台提醒扫描：开启后 Python 会定时扫描到期待办/日历并生成提醒。
    notification_worker_enabled: bool = True
    # 后台提醒扫描间隔，单位：秒。
    notification_worker_interval_seconds: int = 30
    # 后台记忆衰减：开启后会周期性把长期未使用画像的有效置信度写回，并过期极低置信度画像。
    memory_decay_worker_enabled: bool = True
    memory_decay_worker_interval_seconds: int = 3600
    automation_worker_enabled: bool = True
    automation_worker_interval_seconds: int = 60

    # ===== Prompt / 安全 / 日志 / 限流 =====
    # Prompt 版本号，方便后续灰度、评测和排查。
    prompt_version: str = "lyle-agent-v1"
    # Java 调 Python 时携带的内部 token，防止外部绕过 Java 直接访问 Python。
    # 必须在 .env 中显式配置强随机值（如 openssl rand -hex 32）；为空或占位值时服务会拒绝所有请求。
    internal_api_token: str = ""
    # Python 服务日志级别。
    log_level: str = "INFO"
    # Python API 简单限流，每分钟最大请求数。
    python_rate_limit_per_minute: int = 120

    # ===== V6.4 AI 用量治理 =====
    ai_usage_reporting_enabled: bool = True
    ai_usage_java_base_url: str = "http://localhost:8080/api"
    ai_usage_internal_token: str = "lyle-dev-usage-token-change-me"
    ai_usage_reporting_timeout_seconds: float = 3.0
    ai_usage_route_version: str = "v6.4-1"
    ai_usage_pricing_version: str = "2026-07-23"
    # 价格单位为 CNY / 百万 Token；费用仅作治理估算，不代表账单。
    ai_usage_input_price_cny_per_million: float = 2.0
    ai_usage_output_price_cny_per_million: float = 8.0
    ai_usage_embedding_price_cny_per_million: float = 0.0
    ai_usage_image_cost_micros_per_call: int = 0
    ai_usage_outbox_worker_enabled: bool = True
    ai_usage_outbox_worker_interval_seconds: int = 15

    # ===== PostgreSQL 连接池 =====
    # Python 记忆库 PostgreSQL 连接池大小。
    postgres_pool_min_size: int = 1
    postgres_pool_max_size: int = 10

    # ===== Redis：短期记忆 =====
    # 开启后短期对话记忆写入 Redis；关闭时使用进程内存，重启会丢失。
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"

    # ===== PostgreSQL：长期摘要 + 结构化记忆 =====
    # 开启后长期摘要、结构化记忆写入 PostgreSQL。
    postgres_enabled: bool = False
    postgres_dsn: str = "postgresql://aiapp:aiapp@localhost:5432/ai_memory"

    # ===== MinIO：对象存储 =====
    # 用于知识库文件、上传文件等对象存储。
    minio_enabled: bool = False
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False

    # ===== Qdrant：知识库向量检索 + 语义记忆 =====
    # qdrant_collection：知识库文档片段集合。
    # qdrant_memory_collection：语义记忆集合。
    qdrant_enabled: bool = False
    qdrant_endpoint: str = "http://localhost:6333"
    qdrant_collection: str = "lyle_knowledge"
    qdrant_memory_collection: str = "lyle_semantic_memory"

    # ===== Embedding：文本向量化 =====
    # 当前默认 fastembed + bge-small-zh，本地生成向量。
    # 推荐：中文本地向量继续用 BAAI/bge-small-zh-v1.5；云端可切 text-embedding-3-small。
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dimension: int = 512
    # 如果后续切到云端 Embedding 服务，可以使用这里的 API Key 和 Base URL。
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_timeout_seconds: float = 30
    # 开发环境可自动重建维度不匹配的向量集合；生产环境需要谨慎。
    recreate_vector_collections_on_dimension_mismatch: bool = True

    # ===== Reranker：RAG 和记忆召回结果重排 =====
    # 开启后先从向量库多召回一些候选，再用 reranker 重排后裁剪 top_k。
    rerank_enabled: bool = False
    # 推荐：flag_embedding + BAAI/bge-reranker-base；未安装依赖时会退化到 lexical 关键词重排。
    rerank_provider: str = "lexical"
    rerank_model: str = "BAAI/bge-reranker-base"
    # FlagEmbedding 推理选项。GPU 环境可用 fp16；CPU/Mac 开发环境建议 false。
    rerank_use_fp16: bool = False
    # flag_embedding 不可用时是否自动退回 lexical，开发环境建议 true。
    rerank_fallback_enabled: bool = True
    # RAG 和记忆初召回候选数量。
    rerank_candidate_limit: int = 20
    # ===== RAG Agent 增强 =====
    # query rewrite 会把一个问题改写成多个检索 query，再合并去重结果。
    rag_query_rewrite_enabled: bool = True
    rag_multi_query_count: int = 3
    rag_context_top_k: int = 5
    # 命中文档片段后，补充同章节/相邻片段作为父子上下文，提升跨段落回答稳定性。
    rag_parent_child_expand_enabled: bool = True
    rag_parent_child_sibling_limit: int = 2
    # sufficiency check 会判断知识库资料是否足够回答。
    rag_sufficiency_check_enabled: bool = True

    # ===== Milvus：向量记忆 =====
    # server 模式连接独立 Milvus；lite 模式写入本地文件。
    milvus_enabled: bool = False
    milvus_mode: str = "server"
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_lite_path: str = "/tmp/lyle_vector_memory.db"
    milvus_collection: str = "lyle_vector_memory"
    # 向量后端不可用时快速降级，避免每轮聊天都等待连接失败。
    vector_backend_timeout_seconds: float = 1.5
    vector_backend_failure_cooldown_seconds: float = 60

    # ===== 记忆策略 =====
    # Redis 短期记忆最终保留条数。
    memory_short_term_limit: int = 12
    # Redis 短期记忆超过该数量后触发压缩，然后裁剪回 memory_short_term_limit。
    memory_compression_trigger_limit: int = 24
    # 记忆召回 top_k 数量。
    memory_recall_limit: int = 6
    # 开启后 Recall Planner 会先用意图模型选择画像路径，失败时回退规则召回。
    recall_planner_llm_enabled: bool = True
    # 记忆接口导出给前端时，单条短期记忆内容的最大长度。
    memory_export_turn_max_chars: int = 1200
    # 向量记忆低于该相似度不注入上下文。调试页仍可看到原始语义召回候选。
    memory_vector_score_threshold: float = 0.5
    # 写入向量记忆的文本长度上限，避免助手长篇回复污染长期语义记忆。
    memory_vector_text_max_chars: int = 500
    # 向量记忆写入策略。开启后会过滤寒暄、确认词、极短无信息句，减少向量库噪音。
    memory_vector_write_policy_enabled: bool = True
    # 低于该长度且没有命中长期价值信号的问题，不写入向量记忆。
    memory_vector_write_min_chars: int = 8
    # 向量记忆写入前查重。命中高相似历史记忆时跳过新增，避免同义表达反复堆积。
    memory_vector_dedupe_enabled: bool = True
    memory_vector_dedupe_score_threshold: float = 0.92
    # 结构化记忆抽取模式：rule、llm、hybrid。
    structured_memory_extractor: str = "hybrid"
    # 结构化记忆置信度阈值，低于该值不写入。
    structured_memory_confidence_threshold: float = 0.8
    # 结构化记忆抽取时参考最近几条短期记忆。
    structured_memory_recent_turns: int = 6
    # 结构化记忆冲突处理：新记忆置信度低于旧记忆超过该差值时忽略新记忆。
    structured_memory_conflict_confidence_margin: float = 0.15


settings = Settings()
