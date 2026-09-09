"""Small scene vocabulary; concrete tool choice remains model-driven."""
SCENES = {
    "data_delete": ("删除", "delete", "remove data"),
    "file_batch_edit": ("批量修改", "批量替换", "batch edit", "修改文件"),
    "email": ("邮件", "email", "mail"),
    "diagnosis": ("诊断", "故障", "报错", "失败原因", "debug", "diagnose"),
    "code_review": ("代码检查", "代码审查", "code review", "检查代码"),
    "project_analysis": ("项目", "project", "发布检查"),
    "research": ("调研", "对比", "汇总", "多来源", "research", "搜索", "search"),
    "knowledge": ("知识库", "文档检索", "knowledge"),
}


def route_scene(question: str, knowledge_base_ids: list[str] | None = None) -> str:
    text = question.lower()
    for scene, terms in SCENES.items():
        if any(term in text for term in terms):
            return scene
    return "knowledge" if knowledge_base_ids else "tool_or_chat"


def rank_mcp_tools(catalog: dict, question: str) -> dict:
    """Prefer relevant tool descriptions when the tenant has more than 20 tools."""
    query = question.lower()
    grams = {query[index:index + 2] for index in range(len(query) - 1) if not query[index:index + 2].isspace()}
    def score(item):
        tool = item[1]
        text = f"{tool['toolName']} {tool.get('description', '')}".lower()
        return sum(gram in text for gram in grams)
    return dict(sorted(catalog.items(), key=score, reverse=True)[:20])
