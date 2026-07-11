from dataclasses import dataclass


RISK_ORDER = {
    "read": 0,
    "read_only": 0,
    "external_read": 1,
    "compute": 1,
    "artifact": 1,
    "write_low": 2,
    "write": 2,
    "write_medium": 3,
    "write_high": 4,
    "dangerous": 5,
}


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    requires_confirmation: bool
    requires_second_confirmation: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "requiresConfirmation": self.requires_confirmation,
            "requiresSecondConfirmation": self.requires_second_confirmation,
            "reason": self.reason,
        }


def decide_permission(
    *,
    executor_type: str,
    executor_name: str,
    operation: str,
    risk_level: str,
    trigger_type: str = "manual",
    allow_automation: bool = False,
) -> PermissionDecision:
    risk_rank = RISK_ORDER.get(risk_level, 3)
    if risk_rank >= RISK_ORDER["dangerous"]:
        return PermissionDecision(False, False, False, "危险操作默认禁用。")
    if trigger_type == "automation" and not allow_automation and risk_rank >= RISK_ORDER["write_low"]:
        return PermissionDecision(False, True, False, "自动化不允许直接执行写操作。")
    if risk_rank >= RISK_ORDER["write_high"]:
        return PermissionDecision(True, True, True, "高风险操作需要二次确认。")
    if risk_rank >= RISK_ORDER["write_low"]:
        return PermissionDecision(True, True, False, "写操作需要用户确认。")
    return PermissionDecision(True, False, False, f"{executor_type}:{executor_name}:{operation} 可直接执行。")
