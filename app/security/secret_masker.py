import re


SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password|cookie)\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"\b(sk-[A-Za-z0-9_-]{12,})\b"),
    re.compile(r"\b(ghp_[A-Za-z0-9_]{20,})\b"),
]


def mask_secrets(text: str | None) -> str:
    if not text:
        return ""
    masked = str(text)
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= 2:
            masked = pattern.sub(lambda match: f"{match.group(1)}=***", masked)
        else:
            masked = pattern.sub("***", masked)
    return masked
