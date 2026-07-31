from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class ChatModelConfig:
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    timeout: float = 60


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def build_chat_model(
        self,
        config: ChatModelConfig,
        streaming: bool = False,
        callbacks: list[Any] | None = None,
    ) -> ChatOpenAI:
        return ChatOpenAI(
            api_key=config.api_key or "not-needed",
            base_url=config.base_url,
            model=config.model,
            timeout=config.timeout,
            streaming=streaming,
            stream_usage=streaming,
            callbacks=callbacks,
        )

    def is_configured(self, config: ChatModelConfig) -> bool:
        return bool(config.api_key)
