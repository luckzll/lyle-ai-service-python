from dataclasses import dataclass


@dataclass(frozen=True)
class ToolRiskPolicy:
    risk_level: str
    requires_confirmation: bool
    description: str


READ_ONLY = ToolRiskPolicy(
    risk_level="read",
    requires_confirmation=False,
    description="只读操作，可直接执行。",
)
EXTERNAL_READ = ToolRiskPolicy(
    risk_level="external_read",
    requires_confirmation=False,
    description="会访问外部服务读取信息，不修改用户数据。",
)
WRITE_CONFIRM = ToolRiskPolicy(
    risk_level="write",
    requires_confirmation=True,
    description="会写入或修改用户数据，必须先预览并由用户确认。",
)
GENERATIVE_ARTIFACT = ToolRiskPolicy(
    risk_level="artifact",
    requires_confirmation=False,
    description="生成内容或临时产物，当前不要求二次确认。",
)


TOOL_RISK_POLICIES: dict[str, ToolRiskPolicy] = {
    "health_check": READ_ONLY,
    "memory_snapshot": READ_ONLY,
    "memory_stats": READ_ONLY,
    "list_tools": READ_ONLY,
    "weather": EXTERNAL_READ,
    "web_search": EXTERNAL_READ,
    "file_parse": READ_ONLY,
    "image_analyze": READ_ONLY,
    "image_generate": GENERATIVE_ARTIFACT,
    "image_edit": GENERATIVE_ARTIFACT,
    "todo": WRITE_CONFIRM,
    "calendar": WRITE_CONFIRM,
}


def get_tool_risk_policy(tool_name: str) -> ToolRiskPolicy:
    return TOOL_RISK_POLICIES.get(tool_name, WRITE_CONFIRM)
