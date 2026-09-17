"""Preserve provider-exposed reasoning without changing OpenAI request options.

langchain-openai intentionally drops nonstandard Chat Completions fields. Keep
only the public reasoning_content delta; leave usage, tools and finish handling
to the installed adapter. This does not enable a model's thinking mode.
"""
from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI


class ReasoningChatOpenAI(ChatOpenAI):
    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation is None or not isinstance(generation.message, AIMessageChunk):
            return generation
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        delta = choices[0].get("delta") if choices else None
        reasoning = delta.get("reasoning_content") if isinstance(delta, dict) else None
        if isinstance(reasoning, str) and reasoning:
            generation.message.additional_kwargs["reasoning_content"] = reasoning
        return generation
