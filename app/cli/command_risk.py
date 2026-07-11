import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandRisk:
    risk_level: str
    policy: str
    requires_confirmation: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "riskLevel": self.risk_level,
            "policy": self.policy,
            "requiresConfirmation": self.requires_confirmation,
            "reason": self.reason,
        }


READ_ONLY_ROOTS = {"pwd", "ls", "rg", "find"}
TEST_ROOTS = {"pytest", "mvn"}
DANGEROUS_TOKENS = {"sudo", "rm", "mkfs", "dd", "chmod", "chown"}
SHELL_PIPE_INSTALL = {"curl", "wget"}


def classify_command(command: str) -> CommandRisk:
    command = command.strip()
    if not command:
        return CommandRisk("dangerous", "deny", False, "空命令不能执行。")
    try:
        parts = shlex.split(command)
    except ValueError:
        return CommandRisk("dangerous", "deny", False, "命令解析失败。")
    root = parts[0] if parts else ""
    lowered = command.lower()
    if "|" in command and any(token in lowered for token in SHELL_PIPE_INSTALL) and "sh" in lowered:
        return CommandRisk("dangerous", "deny", False, "禁止 curl/wget 管道执行脚本。")
    if root in DANGEROUS_TOKENS:
        return CommandRisk("dangerous", "deny", False, f"{root} 属于默认禁用命令。")
    if root == "git":
        return _classify_git(parts)
    if root in READ_ONLY_ROOTS:
        return CommandRisk("read_only", "allow", False, "命令只读取本地工作区。")
    if root in TEST_ROOTS or (root in {"npm", "pnpm", "yarn"} and any(part in {"test", "run"} for part in parts[1:])):
        return CommandRisk("compute", "allow", False, "测试或构建类命令，不应修改业务数据。")
    if any(part in {"--fix", "format", "fmt"} for part in parts):
        return CommandRisk("write_low", "confirm", True, "格式化或自动修复可能修改文件。")
    if root in {"node", "python", "python3"}:
        return CommandRisk("compute", "allow", False, "脚本执行按计算类处理，仅允许在工作区内运行。")
    return CommandRisk("write_medium", "confirm", True, "未知命令需要用户确认后执行。")


def _classify_git(parts: list[str]) -> CommandRisk:
    sub = parts[1] if len(parts) > 1 else ""
    if sub in {"status", "diff", "log", "show", "branch"}:
        return CommandRisk("read_only", "allow", False, "git 只读查询。")
    if sub in {"add", "commit"}:
        return CommandRisk("write_medium", "confirm", True, "git 写入本地仓库状态，需要确认。")
    if sub == "push":
        return CommandRisk("write_high", "confirm", True, "git push 会影响远端仓库，需要二次确认。")
    if sub == "reset" and "--hard" in parts:
        return CommandRisk("dangerous", "deny", False, "git reset --hard 默认禁用。")
    return CommandRisk("write_medium", "confirm", True, "git 非只读操作需要确认。")
