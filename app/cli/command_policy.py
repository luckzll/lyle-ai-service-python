from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

from app.security.permission_engine import RISK_ORDER


@dataclass(frozen=True)
class CommandPolicyDecision:
    allowed: bool
    matched: bool
    requires_confirmation: bool
    risk_level: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "matched": self.matched,
            "requiresConfirmation": self.requires_confirmation,
            "riskLevel": self.risk_level,
            "reason": self.reason,
        }


def evaluate_command_policy(command: str, policies: list[dict[str, Any]]) -> CommandPolicyDecision:
    matches = [policy for policy in policies if _matches(command, str(policy.get("commandPattern") or ""))]
    denied = next((policy for policy in matches if policy.get("policy") == "deny"), None)
    if denied:
        return _decision(False, True, denied, "命令命中工作区禁止规则。")

    confirmation = next((policy for policy in matches if policy.get("policy") == "confirm"), None)
    if confirmation:
        return _decision(True, True, confirmation, "命令命中工作区确认规则。", force_confirmation=True)

    allow_policies = [policy for policy in policies if policy.get("policy") == "allow"]
    allowed = next((policy for policy in matches if policy.get("policy") == "allow"), None)
    if allow_policies and not allowed:
        return CommandPolicyDecision(False, False, False, "dangerous", "命令不在工作区白名单中。")
    if allowed:
        return _decision(True, True, allowed, "命令命中工作区允许规则。")
    return CommandPolicyDecision(True, False, False, "read_only", "未配置匹配规则，使用系统默认安全策略。")


def stricter_risk_level(default_risk: str, policy_risk: str) -> str:
    if RISK_ORDER.get(policy_risk, 0) > RISK_ORDER.get(default_risk, 0):
        return policy_risk
    return default_risk


def _decision(
    allowed: bool,
    matched: bool,
    policy: dict[str, Any],
    reason: str,
    *,
    force_confirmation: bool = False,
) -> CommandPolicyDecision:
    return CommandPolicyDecision(
        allowed,
        matched,
        force_confirmation or bool(policy.get("requiresConfirmation", False)),
        str(policy.get("riskLevel") or "read_only"),
        reason,
    )


def _matches(command: str, pattern: str) -> bool:
    command = command.strip()
    pattern = pattern.strip()
    if not pattern:
        return False
    if any(char in pattern for char in "*?["):
        return fnmatchcase(command, pattern)
    return command == pattern or command.startswith(f"{pattern} ")
