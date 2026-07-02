from app.llm.providers.base import ChatModelConfig, OpenAICompatibleProvider


class LocalOllamaProvider(OpenAICompatibleProvider):
    name = "local_ollama"

    def is_configured(self, config: ChatModelConfig) -> bool:
        return bool(config.base_url and config.model)
