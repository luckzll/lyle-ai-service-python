# 分块上下文与真实本地向量评测

使用原有 50 题冻结证据集，问题和原始片段保持不变；期望证据只在检索完成后用于评分。

| 方案 | Recall@5 | MRR | 引用位置有效率 | 热查询 p50 / p95（ms） | 首次查询（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| before_batch | 1.0000 | 0.9439 | 1.0000 | 4.35 / 5.29 | 8.27 |
| context | 0.9886 | 0.9621 | 1.0000 | 4.38 / 5.37 | 6.23 |
| context_fastembed_rerank | 0.9773 | 0.9545 | 1.0000 | 1781.35 / 2145.27 | 1859.75 |
| raw_fastembed_rerank | 1.0000 | 0.9602 | 1.0000 | 1230.11 / 1412.10 | 1276.67 |

before_batch：本批开始前代码与原文；context：当前代码与上下文增强；raw_fastembed_rerank：当前代码、原文与重排；context_fastembed_rerank：当前代码、增强与重排。
模型：BAAI/bge-small-zh-v1.5；可选重排：BAAI/bge-reranker-base。模型仅从本地缓存加载。
使用真实模型向量和本地余弦计算代替 Qdrant 网络调用，关键词召回、融合、相邻片段扩展调用对应版本的实际代码。
热查询耗时包含每次真实问题向量化和本地召回，重排组还包含真实模型重排；未开启跨请求缓存。
before_batch、context、raw_fastembed_rerank 的首次查询包含各自关键词索引初建；组合重排组复用 context 的索引和文档向量。模型加载与整库向量化另记，不能解释为线上冷启动或线上延迟。
固定原有片段是为了隔离上下文增强的效果；新分块器的原文位置修复另由合同测试验证。
这些组没有使用多轮补全、辅助模型改写或答案生成，不能据此判断最终回答正确率或线上提速。

本次模型初始化 116.74 ms；整库向量化：before_batch 237.00 ms、context 332.25 ms、raw_fastembed_rerank 235.86 ms。
context_fastembed_rerank：重排预算上限 40，本数据集实际每题输入 3–32 个候选；所有返回项已核对 rerank_backend=fastembed。
raw_fastembed_rerank：重排预算上限 40，本数据集实际每题输入 3–32 个候选；所有返回项已核对 rerank_backend=fastembed。
重排模型加载 1644.43 ms，单独统计。
原文重排组为同一份模型、数据和源码的追加消融，前三组测量保持原样；追加进程向量模型加载 99.86 ms、重排模型加载 1652.98 ms。
每题计时重复次数：1；表中 p50 / p95 跨 50 题汇总，小数据集不代表线上负载与尾延迟。

上下文增强组证据召回下降的题目：R035。
保留这些回退样本；MRR 上升不代表每个问题的证据覆盖都改善。

- context / R035：文档变更对缓存有什么影响，引用位置包含什么？ 缺少证据“文档标识、片段序号、起始字符与结束字符”。
- context_fastembed_rerank / R021：具体改了什么？ 缺少证据“只修复连接断开后的恢复状态”。

## 本次发布取舍

继续保持 rag_chunk_context_enabled=False、rerank_enabled=False。本次 50 题证据覆盖尚未证明优于基线，不足以支持默认开启。
上下文增强改变了候选排序；重排可能补回某些复合问题的证据，也可能把缺少上文的短追问排错。本次没有把期望答案写回问题，也没有根据回退样本调整参数。
重排使用 2 个 CPU 线程，增加的耗时来自真实 cross-encoder 推理。raw_fastembed_rerank 使用当前 store 与原文片段，单独测重排；context_fastembed_rerank 则测增强和重排组合，两个结果需要分别判断。
Recall / MRR 只统计 44 道可回答题；无答案题是否返回上下文记录在 JSON，不能仅凭检索指标认定具有拒答能力。引用位置有效只代表片段能对回原文，不代表答案正确。
启用前还需要独立的真实业务保留集、多轮问题补全与多证据覆盖检查，以及目标服务器上的负载和延迟验证。

单独重排的 Recall@5 为 1.0000，MRR 相比基线变化 +1.63 个百分点；本机 p50 为 1.23 秒。这一结果支持保留按需重排选项；组合组的退化不能直接归因给单独重排。

## 复现

基线 store.py 与 keyword_index.py 均显式从本批开始前的快照加载，文件 SHA-256 与原始 50 题数据集 SHA-256 保存在结果 JSON。模型只读本地缓存，不下载、不调用生成 API、不写业务库。
运行命令（在 Python 服务目录）：

```powershell
.venv-win/Scripts/python.exe scripts/evaluate_rag_context_v2.py --baseline-root C:/Temp/lyle-rag-context-baseline-gfiindva --cache-dir C:/Users/zhl20/.cache/lyle-rag-models --output tests/fixtures/rag_context_results_v2.json --report docs/RAG_CONTEXT_EVALUATION_V2.md --repeats 1 --include-reranker
```

## 数据版本与部署边界

增强只对重新入库的片段生成新检索文本；旧片段与旧向量保持 raw-v1，不会因为重建内存关键词索引就自动变成新向量。
Qdrant 点 ID 包含文档 revision，重试只清理相同 revision，避免跨进程的旧任务删除或覆盖新版本。旧 revision 的物理向量需后续 GC，当前只通过可信 manifest 屏蔽，仍占用存储。
Java 每次从数据库确认权限、已完成文档名单与对应 revision；Python HTTP 知识库入口必须验证完整 manifest。
发布与回滚须保持 Java/Python 的 manifest 协议同步。不得把不发送 manifest 的旧 Java 与依赖版本隔离的新 Python 混用。
无 scope 的底层 store helpers 仅用于可信内部工具、兼容测试，不能充当外部 HTTP 回退路径。

