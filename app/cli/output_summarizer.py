from app.security.secret_masker import mask_secrets


def summarize_output(stdout: str, stderr: str, max_chars: int = 4000) -> str:
    merged = "\n".join(part for part in [stdout.strip(), stderr.strip()] if part)
    masked = mask_secrets(merged)
    if len(masked) <= max_chars:
        return masked
    head = masked[: max_chars - 120]
    return f"{head}\n...输出已截断，共 {len(masked)} 字符。"
