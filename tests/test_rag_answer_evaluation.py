"""Hermetic contracts for the opt-in paired generation script."""
import importlib.util
from dataclasses import replace
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from app.llm import model_router, prompts
from app.llm.providers.base import ChatModelConfig


@pytest.fixture
def evaluation(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_rag_answers.py"
    spec = importlib.util.spec_from_file_location("rag_answer_evaluation_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls, builds, clock_calls = [], [], []
    state = SimpleNamespace(calls=calls, builds=builds, clock_calls=clock_calls, fail_at=None,
                            answer="合成测试回答", finish_reason="stop")

    class FakeModel:
        def bind(self, **kwargs):
            state.generation_options = kwargs
            return self

        def invoke(self, messages):
            calls.append([{"type": item.type, "content": item.content} for item in messages])
            if len(calls) == state.fail_at:
                raise RuntimeError("https://user:secret-credential@example.invalid?api_key=private")
            return SimpleNamespace(content=state.answer, usage_metadata={"input_tokens": 12, "output_tokens": 5},
                                   response_metadata={"finish_reason": state.finish_reason})

    class FakeProvider:
        def is_configured(self, config):
            return True

        def build_chat_model(self, config, **kwargs):
            builds.append((config, kwargs))
            return FakeModel()

    def fixed_runtime():
        clock_calls.append(1)
        return f"runtime-snapshot-{len(clock_calls)}\n"

    monkeypatch.setattr(model_router, "get_model_config", lambda _: ChatModelConfig(
        provider="fake", model="same-evaluation-model", api_key="test-secret", timeout=999, max_retries=5,
    ))
    monkeypatch.setattr(model_router, "get_provider", lambda _: FakeProvider())
    monkeypatch.setattr(prompts, "get_system_prompt", lambda: ("固定系统提示词", "contract-v1"))
    monkeypatch.setattr(prompts, "_runtime_context_text", fixed_runtime)
    # Fail loudly if future script changes select the application's billable path.
    monkeypatch.setattr(model_router, "build_chat_model_for_task", lambda *_args, **_kwargs: pytest.fail("Application model path is forbidden"))
    monkeypatch.setattr(model_router, "UsageCallbackHandler", lambda *_args, **_kwargs: pytest.fail("Billing callbacks are forbidden"))
    from app.core import database
    from app.prompts import store as prompt_store
    monkeypatch.setattr(database, "postgres_connection", lambda: pytest.fail("Database calls are forbidden"))
    monkeypatch.setattr(prompt_store, "postgres_connection", lambda: pytest.fail("Database calls are forbidden"))
    return module, state, fixed_runtime


def _payload(count=3):
    return {
        "fixture_sha256": "synthetic-fixture-only",
        "expected": "TOP_LEVEL_ANSWER_KEY",
        "cases": [{
            "case_id": f"case-{index}", "question": f"合成问题 {index}",
            "history": [{"role": "user", "content": f"相同历史 {index}"}],
            "baseline_contexts": [{"title": "合成资料", "chunk": f"基线资料 {index}", "expected": "ROW_ANSWER_KEY"}],
            "optimized_contexts": [{"title": "合成资料", "chunk": f"优化资料 {index}", "review_only": "ROW_REVIEW_KEY"}],
            "review_only": {"expected_evidence": "PRIVATE_EVIDENCE", "expected": "CASE_ANSWER_KEY"},
            "evidence": "CASE_EVIDENCE_KEY", "expected": "CASE_EXPECTED_KEY",
        } for index in range(count)],
    }


def test_paired_prompts_exclude_answer_keys_and_hold_generation_inputs_stable(evaluation):
    module, state, original_runtime = evaluation
    output = list(module.run_comparison(_payload(), limit=3))
    metadata, answers = output[0], output[1:]
    assert metadata["case_count"] == 3
    assert [(item["case_id"], item["variant"]) for item in answers] == [
        ("case-0", "baseline"), ("case-0", "optimized"),
        ("case-1", "optimized"), ("case-1", "baseline"),
        ("case-2", "baseline"), ("case-2", "optimized"),
    ]
    assert len(state.builds) == 1
    config, kwargs = state.builds[0]
    assert (config.model, config.timeout, config.max_retries) == ("same-evaluation-model", 60, 0)
    assert not kwargs.get("callbacks")
    assert state.generation_options == {"temperature": 0, "max_tokens": 768}
    assert len(state.clock_calls) == 1
    assert prompts._runtime_context_text is original_runtime
    assert len({messages[0]["content"] for messages in state.calls}) == 1
    serialized = json.dumps(state.calls, ensure_ascii=False)
    for forbidden in ("TOP_LEVEL_ANSWER_KEY", "ROW_ANSWER_KEY", "ROW_REVIEW_KEY", "PRIVATE_EVIDENCE",
                      "CASE_ANSWER_KEY", "CASE_EVIDENCE_KEY", "CASE_EXPECTED_KEY"):
        assert forbidden not in serialized
    for index in range(3):
        pair = state.calls[index * 2:index * 2 + 2]
        assert all(f"相同历史 {index}" in messages[1]["content"] for messages in pair)
        assert all("runtime-snapshot-1" in messages[1]["content"] for messages in pair)


@pytest.mark.parametrize("limit,expected", [(1, 2), (9, 18), (999, 18)])
def test_call_budget_is_two_per_selected_case_without_retries(evaluation, limit, expected):
    module, state, _ = evaluation
    list(module.run_comparison(_payload(12), limit=limit))
    assert len(state.calls) == expected
    assert state.builds[0][0].max_retries == 0


def test_provider_invocation_error_is_redacted_and_does_not_retry(evaluation):
    module, state, original_runtime = evaluation
    state.fail_at = 1
    output = list(module.run_comparison(_payload(1), limit=1))
    assert output[1]["error_type"] == "RuntimeError"
    assert "answer" not in output[1]
    assert len(output) == 2
    assert len(state.calls) == 1
    serialized = json.dumps(output)
    assert "secret-credential" not in serialized
    assert "api_key" not in serialized
    assert prompts._runtime_context_text is original_runtime


def test_history_only_includes_user_and_assistant_text_without_review_metadata(evaluation):
    module, state, _ = evaluation
    payload = _payload(1)
    payload["cases"][0]["history"] = [
        {"role": "user", "content": "用户历史", "expected": "HISTORY_EXPECTED", "review_only": "HISTORY_REVIEW"},
        {"role": "assistant", "content": "助手历史", "evidence": "HISTORY_EVIDENCE"},
        {"role": "system", "content": "INJECTED_SYSTEM"},
        {"role": "user", "content": {"expected": "NESTED_ANSWER_KEY"}},
        "UNSTRUCTURED_HISTORY_KEY",
    ]
    list(module.run_comparison(payload, limit=1))
    for messages in state.calls:
        prompt = messages[1]["content"]
        assert "用户历史" in prompt and "助手历史" in prompt
        for forbidden in ("HISTORY_EXPECTED", "HISTORY_REVIEW", "HISTORY_EVIDENCE", "INJECTED_SYSTEM",
                          "NESTED_ANSWER_KEY", "UNSTRUCTURED_HISTORY_KEY"):
            assert forbidden not in prompt


def test_provider_initialization_errors_only_expose_type(evaluation, monkeypatch):
    module, state, original_runtime = evaluation
    def fail_build(*_args, **_kwargs):
        raise ValueError("https://user:secret-credential@example.invalid?api_key=private")
    monkeypatch.setattr(model_router, "get_provider", lambda _: SimpleNamespace(is_configured=lambda _: True, build_chat_model=fail_build))
    output = list(module.run_comparison(_payload(1), limit=1))
    assert output == [{"kind": "error", "error_type": "ValueError"}]
    assert not state.calls
    assert prompts._runtime_context_text is original_runtime


def test_early_generator_close_restores_runtime_without_generating(evaluation):
    module, state, original_runtime = evaluation
    results = module.run_comparison(_payload(1), limit=1)
    assert next(results)["kind"] == "metadata"
    results.close()
    assert not state.calls
    assert prompts._runtime_context_text is original_runtime


def test_cli_requires_opt_in_before_reading_or_writing_files(evaluation, monkeypatch, tmp_path):
    module, state, _ = evaluation
    output_path = tmp_path / "answers.jsonl"
    monkeypatch.setattr(sys, "argv", ["evaluate_rag_answers.py", "--input", str(tmp_path / "missing.json"), "--output", str(output_path)])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert not state.builds and not state.calls
    assert not output_path.exists()


def test_cli_top_level_redacts_file_error_without_traceback(evaluation, monkeypatch, tmp_path, capsys):
    module, state, _ = evaluation
    output_path = tmp_path / "answers.jsonl"
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["evaluate_rag_answers.py", "--allow-model-calls",
                                    "--input", str(tmp_path / "private-api-key-missing.json"), "--output", str(output_path)])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(module.__file__, run_name="__main__")
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"kind": "error", "error_type": "FileNotFoundError"}
    assert not captured.err
    assert not state.builds and not state.calls
    assert not output_path.exists()


@pytest.mark.parametrize("disable_thinking", [False, True])
def test_qwen_evaluation_thinking_override_is_explicit_and_recorded(evaluation, monkeypatch, disable_thinking):
    module, state, _ = evaluation
    original_config = replace(model_router.get_model_config("chat"), provider="qwen")
    monkeypatch.setattr(model_router, "get_model_config", lambda _: original_config)
    output = list(module.run_comparison(_payload(1), limit=1, disable_thinking=disable_thinking))
    assert output[0]["thinking_disabled_for_evaluation"] is disable_thinking
    expected = {"temperature": 0, "max_tokens": 768}
    if disable_thinking:
        expected["extra_body"] = {"enable_thinking": False}
    assert state.generation_options == expected
    assert len(state.calls) == 2
    # The evaluation copy may tighten its limits; the source model configuration remains intact.
    assert (original_config.timeout, original_config.max_retries) == (999, 5)
    assert model_router.get_model_config("chat") is original_config


def test_thinking_override_rejects_non_qwen_without_build_or_call(evaluation):
    module, state, original_runtime = evaluation
    output = list(module.run_comparison(_payload(1), limit=1, disable_thinking=True))
    assert output == [{"kind": "error", "error_type": "ValueError"}]
    assert not state.builds and not state.calls
    assert prompts._runtime_context_text is original_runtime


@pytest.mark.parametrize("answer,finish_reason", [("", "stop"), (" \n\t", "stop"), ("部分正文", "length"), ("", "length")])
def test_incomplete_generation_stops_immediately_and_is_not_a_success(evaluation, answer, finish_reason):
    module, state, original_runtime = evaluation
    state.answer, state.finish_reason = answer, finish_reason
    output = list(module.run_comparison(_payload(3), limit=3))
    assert len(output) == 2
    assert output[1]["error_type"] == "IncompleteGeneration"
    assert output[1]["finish_reason"] == finish_reason
    assert len(state.calls) == 1
    assert prompts._runtime_context_text is original_runtime
