def should_retry(attempt: int, max_retries: int, *, retryable: bool = True) -> bool:
    return retryable and attempt <= max(0, max_retries)
