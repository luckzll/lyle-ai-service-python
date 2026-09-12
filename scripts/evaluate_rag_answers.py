"""Opt-in generation comparison using actual retrieved synthetic contexts.

No answer keys are sent to the model. Uses the configured chat provider without
application usage callbacks, conversation writes, or retrieval/model downloads.
Run only intentionally: this consumes at most two model calls per selected case.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter


def run_comparison(payload: dict, limit: int = 9, *, disable_thinking: bool = False):
    from langchain_core.messages import HumanMessage, SystemMessage
    from app.llm import prompts
    from app.llm.model_router import TASK_CHAT, get_model_config, get_provider

    try:
        config = replace(get_model_config(TASK_CHAT), timeout=60, max_retries=0)
        provider = get_provider(config.provider)
        if not provider.is_configured(config):
            raise RuntimeError("The configured chat provider is not available")
        # No callbacks: these synthetic probes must not create customer billing rows.
        options = {"temperature": 0, "max_tokens": 768}
        if disable_thinking:
            if config.provider != "qwen":
                raise ValueError("The thinking override is only supported for Qwen evaluation")
            options["extra_body"] = {"enable_thinking": False}
        model = provider.build_chat_model(config).bind(**options)
        system_prompt, prompt_version = prompts.get_system_prompt()
        runtime_text = prompts._runtime_context_text()
    except Exception as exc:
        yield {"kind": "error", "error_type": type(exc).__name__}
        return
    original_runtime_text = prompts._runtime_context_text
    prompts._runtime_context_text = lambda: runtime_text
    try:
        yield {
            "kind": "metadata", "schema_version": 1,
            "model": config.model, "provider": config.provider,
            "fixture_sha256": payload.get("fixture_sha256"),
            "input_sha256": hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "case_count": min(len(payload["cases"]), max(1, min(9, limit))),
            "temperature": 0, "max_output_tokens": 768, "max_retries": 0,
            "thinking_disabled_for_evaluation": disable_thinking,
            "prompt_version": prompt_version,
            "limits": "Synthetic diagnostic cases, one generation per variant; not production quality or latency proof.",
        }
        for index, case in enumerate(payload["cases"][:max(1, min(9, limit))]):
            # Explicit allowlist: review_only / evidence / expected facts never
            # enter either the system prompt or the user prompt.
            history = [{"role": row["role"], "content": row["content"]}
                       for row in (case.get("history") or [])
                       if isinstance(row, dict) and row.get("role") in {"user", "assistant"}
                       and isinstance(row.get("content"), str)]
            history_text = "近期对话：\n" + json.dumps(history, ensure_ascii=False) if history else ""
            variants = ("baseline", "optimized") if index % 2 == 0 else ("optimized", "baseline")
            for variant in variants:
                contexts = [{"title": str(row.get("title", "")), "chunk": str(row.get("chunk", ""))}
                            for row in case[f"{variant}_contexts"]]
                messages = [SystemMessage(content=f"{system_prompt}\n\nPrompt-Version: {prompt_version}"),
                            HumanMessage(content=prompts.build_user_prompt(case["question"], contexts, history_text))]
                started = perf_counter()
                result = {"kind": "answer", "case_id": case["case_id"], "variant": variant}
                try:
                    response = model.invoke(messages)
                    result.update(answer=str(response.content), usage=response.usage_metadata or {},
                                  finish_reason=(response.response_metadata or {}).get("finish_reason"))
                except Exception as exc:
                    # Provider messages can contain URLs/credentials. Expose only type.
                    result["error_type"] = type(exc).__name__
                result["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
                if not result.get("error_type") and (not result.get("answer", "").strip() or result.get("finish_reason") == "length"):
                    result["error_type"] = "IncompleteGeneration"
                yield result
                if "error_type" in result:
                    return  # Do not repeat a failing provider configuration.
    finally:
        prompts._runtime_context_text = original_runtime_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=9)
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true", help="Qwen evaluation only; does not change application settings.")
    args = parser.parse_args()
    if not args.allow_model_calls:
        parser.error("Model calls are opt-in; pass --allow-model-calls to run this evaluation.")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for result in run_comparison(payload, args.limit, disable_thinking=args.disable_thinking):
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()
            print(json.dumps({key: result[key] for key in ("kind", "case_id", "variant", "elapsed_ms", "error_type") if key in result}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"kind": "error", "error_type": type(exc).__name__}))
        sys.exit(1)
