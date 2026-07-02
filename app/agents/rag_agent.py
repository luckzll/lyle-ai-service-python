from app.tools.knowledge_tool import retrieve_knowledge_diagnostics_tool


def run_rag_agent(question: str, knowledge_base_ids: list[str], use_rag: bool) -> dict[str, object]:
    decision = decide_rag_retrieval(question, knowledge_base_ids, use_rag)
    if not decision["should_retrieve"]:
        return {
            "contexts": [],
            "queries": [],
            "sufficiency": {"sufficient": True, "reason": decision["reason"], "score": 1.0},
            "decision": decision,
            "skipped": True,
        }
    result = retrieve_knowledge_diagnostics_tool(question, knowledge_base_ids)
    return {
        "contexts": result.get("contexts") or [],
        "queries": result.get("queries") or [question],
        "sufficiency": result.get("sufficiency") or {},
        "decision": decision,
        "skipped": False,
    }


def decide_rag_retrieval(question: str, knowledge_base_ids: list[str], use_rag: bool) -> dict[str, object]:
    if not use_rag:
        return {
            "should_retrieve": False,
            "depth": "skip",
            "reason": "Supervisor 未激活知识库检索。",
        }
    if not knowledge_base_ids:
        return {
            "should_retrieve": False,
            "depth": "skip",
            "reason": "没有选择知识库，无法执行 RAG 检索。",
        }

    depth = "full" if len(question.strip()) >= 12 else "light"
    return {
        "should_retrieve": True,
        "depth": depth,
        "reason": "已选择知识库且 Supervisor 判断需要知识库检索。",
    }
