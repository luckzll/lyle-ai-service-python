"""Bounded, request-local display channel for provider-exposed reasoning.

Reasoning is never inferred from answers, prompts, tools or traces. The short
holdback allows credentials spanning provider chunks to be masked before SSE.
No reasoning content is logged or added to the conversation answer/history.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import re

from app.core.config import settings
from app.security.secret_masker import (
    AUTHORIZATION_PATTERN, BEARER_PATTERN, KEY_VALUE_PATTERN,
    KNOWN_TOKEN_PATTERNS, PRIVATE_KEY_PATTERN, URL_CREDENTIAL_PATTERN, mask_secrets,
)

MAX_REASONING_CHARS = 16_000
_HOLD_BACK = 96
_TOKEN_CHAR = re.compile(r"[A-Za-z0-9_:/@+=.\-~]")
_PRIVATE_START = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----", re.I)
_PENDING_CREDENTIAL = re.compile(
    r"\b(?:api[_-]?key|(?:access|refresh|session)[_-]?token|token|client[_-]?secret|"
    r"secret|password|passwd|cookie|credentials?|private[_-]?key|"
    r"(?:proxy[_-]?)?authorization|bearer)[\"']?\s*(?:[:=]\s*[\"']?\s*)?"
    r"(?:(?:basic|bearer|digest|aws4-hmac-sha256)\s+)?$", re.I,
)
# API tokens are ASCII even when immediately followed by Chinese prose; the
# shared masker's Unicode word boundaries would miss that boundary.
_KNOWN_TOKEN_PATTERNS = [re.compile(pattern.pattern, re.ASCII) for pattern in KNOWN_TOKEN_PATTERNS]
_PATTERNS = [PRIVATE_KEY_PATTERN, URL_CREDENTIAL_PATTERN, AUTHORIZATION_PATTERN,
             BEARER_PATTERN, KEY_VALUE_PATTERN, *_KNOWN_TOKEN_PATTERNS]
logger = logging.getLogger(__name__)


class ReasoningDisplay:
    def __init__(self, callback: Callable[[dict[str, object]], None]):
        self.callback = callback
        self.pending = ""
        self.kind: str | None = None
        self.received = 0
        self.emitted = 0
        self.sequence = 0
        # Opaque configured credentials need exact masking as well as patterns.
        self.secrets = tuple({
            value for key in type(settings).model_fields
            if re.search(r"api_key|secret|password|token", key, re.I)
            and isinstance((value := getattr(settings, key, None)), str)
            and 4 <= len(value) <= MAX_REASONING_CHARS
        })
        self.holdback = max([_HOLD_BACK, *(len(value) for value in self.secrets)])

    def push(self, delta: str, kind: str = "reasoning") -> None:
        if not isinstance(delta, str) or not delta or kind not in {"reasoning", "summary"}:
            return
        if self.received >= MAX_REASONING_CHARS:
            return
        if self.kind is not None and kind != self.kind:
            return
        self.kind = kind
        remaining = MAX_REASONING_CHARS - self.received
        self.pending += delta[:remaining]
        self.received += min(len(delta), remaining)
        self._flush(final=False)
        if len(delta) >= remaining:
            # A cap may bisect a credential. Discard the retained tail rather
            # than treating a truncated token/private key as complete text.
            self.pending = ""

    def _flush(self, *, final: bool) -> None:
        cut = len(self.pending) if final else max(0, len(self.pending) - self.holdback)
        if not final:
            while 0 < cut < len(self.pending) and _TOKEN_CHAR.match(self.pending[cut - 1]) and _TOKEN_CHAR.match(self.pending[cut]):
                cut -= 1
            if unfinished := _PENDING_CREDENTIAL.search(self.pending):
                cut = min(cut, unfinished.start())
            for pattern in _PATTERNS:
                for match in pattern.finditer(self.pending):
                    if match.start() < cut < match.end():
                        cut = match.start()
            for secret in self.secrets:
                start = self.pending.find(secret)
                while start >= 0:
                    if start < cut < start + len(secret):
                        cut = start
                    start = self.pending.find(secret, start + 1)
        # An unfinished PEM block must never escape, including on EOF.
        for match in _PRIVATE_START.finditer(self.pending):
            if PRIVATE_KEY_PATTERN.match(self.pending, match.start()) is None:
                cut = min(cut, match.start())
        if cut <= 0:
            return
        text, self.pending = self.pending[:cut], self.pending[cut:]
        for secret in sorted(self.secrets, key=len, reverse=True):
            text = text.replace(secret, "***")
        text = mask_secrets(text)
        for pattern in _KNOWN_TOKEN_PATTERNS:
            text = pattern.sub("***", text)
        text = text[:MAX_REASONING_CHARS - self.emitted]
        if not text:
            return
        self.sequence += 1
        self.emitted += len(text)
        try:
            self.callback({"sequence": self.sequence, "step": "writer_agent",
                           "delta": text, "kind": self.kind})
        except Exception:
            # An observer must not affect execution or leak callback data.
            logger.warning("Chat reasoning observer unavailable; task continues.")

    def finish(self) -> None:
        self._flush(final=True)
        self.pending = ""


_display: ContextVar[ReasoningDisplay | None] = ContextVar("chat_reasoning_display", default=None)


@contextmanager
def reasoning_scope(callback: Callable[[dict[str, object]], None]) -> Iterator[None]:
    token = _display.set(ReasoningDisplay(callback))
    try:
        yield
    finally:
        # Normal model completion flushes explicitly; cancellation/error must
        # not flush an incomplete provider fragment after an error event.
        _display.reset(token)


def emit_reasoning(delta: str, kind: str = "reasoning") -> None:
    if (display := _display.get()) is not None:
        display.push(delta, kind)


def finish_reasoning() -> None:
    if (display := _display.get()) is not None:
        display.finish()
