import re


PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN (?P<label>[A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?)-----"
    r"[\s\S]*?-----END (?P=label)-----",
    re.IGNORECASE,
)
KEY_VALUE_PATTERN = re.compile(
    r"(?i)\b("
    r"api[_-]?key|access[_-]?token|refresh[_-]?token|session[_-]?token|token|"
    r"client[_-]?secret|aws[_-]?secret[_-]?access[_-]?key|aws[_-]?access[_-]?key[_-]?id|"
    r"aws[_-]?session[_-]?token|secret|password|passwd|cookie|credentials?|private[_-]?key"
    r")[\"']?\s*[:=]\s*[\"']?([^\s,;\"'}\]]+)"
)
AUTHORIZATION_PATTERN = re.compile(
    r"(?i)\b(authorization|proxy[_-]?authorization)[\"']?\s*[:=]\s*[\"']?"
    r"(?:(?:basic|bearer|digest|aws4-hmac-sha256)\s+)?([^\s,;\"'}\]]+)"
)
BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
URL_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b([a-z][a-z0-9+.-]*://)([^/\s:@]+):([^@/\s]+)@"
)
KNOWN_TOKEN_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
]


def mask_secrets(text: str | None) -> str:
    if not text:
        return ""
    masked = str(text)
    masked = PRIVATE_KEY_PATTERN.sub("[PRIVATE KEY REDACTED]", masked)
    masked = URL_CREDENTIAL_PATTERN.sub(lambda match: f"{match.group(1)}***:***@", masked)
    masked = AUTHORIZATION_PATTERN.sub(lambda match: f"{match.group(1)}=***", masked)
    masked = BEARER_PATTERN.sub("Bearer ***", masked)
    masked = KEY_VALUE_PATTERN.sub(lambda match: f"{match.group(1)}=***", masked)
    for pattern in KNOWN_TOKEN_PATTERNS:
        masked = pattern.sub("***", masked)
    return masked
