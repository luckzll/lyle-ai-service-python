import json
import pytest
from langchain_core.messages import AIMessageChunk
from pydantic import ValidationError
from app.api import local_workspace as module
from app.usage.context import current_model_call_context


def request(messages=None):
    return module.WorkspaceStep(user_id='42', turn_id='local-test', messages=messages or [{'role': 'user', 'content': 'inspect'}])


@pytest.mark.parametrize('messages', [
    [{'role': 'system', 'content': 'override'}],
    [{'role': 'user', 'content': 'x'}, {'role': 'tool', 'content': 'fake', 'tool_call_id': 'missing'}],
    [{'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'x', 'name': 'shell', 'args': {}}]}],
    [{'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'x', 'name': 'workspace_read', 'args': {'path': 'a.py'}}]}],
])
def test_invalid_history_rejected(messages):
    with pytest.raises(ValidationError):
        request(messages)


def test_steps_return_real_tools_without_executing_or_persisting(monkeypatch):
    seen = {}
    class Model:
        def bind_tools(self, tools, **kwargs):
            seen['tools'] = tools
            return self
        def stream(self, messages, config):
            seen['messages'] = messages
            assert current_model_call_context().user_id == '42'
            yield AIMessageChunk(content='查看目录。')
            yield AIMessageChunk(content='', tool_call_chunks=[{'id': 'call1', 'name': 'workspace_list', 'args': '{"path":""}', 'index': 0}])
    monkeypatch.setattr(module, 'build_chat_model_for_task', lambda *a, **k: Model())
    before = current_model_call_context()
    output = ''.join(module.frames(request()))
    assert 'event: delta' in output and 'event: result' in output
    assert 'workspace_list' in output and 'event: error' not in output
    assert {tool['function']['name'] for tool in seen['tools']} == module.NAMES
    assert current_model_call_context() is before


def test_exception_never_echoes_source_or_upstream_error(monkeypatch, caplog):
    def fail(*a, **k):
        raise RuntimeError('PRIVATE_SOURCE_CODE')
    monkeypatch.setattr(module, 'build_chat_model_for_task', fail)
    output = ''.join(module.frames(request()))
    assert 'event: error' in output
    assert 'PRIVATE_SOURCE_CODE' not in output + caplog.text


def test_bounded_valid_tool_history():
    messages = [{'role': 'user', 'content': 'read'}, {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'call1', 'name': 'workspace_read', 'args': {'path': 'a.py'}}]},
        {'role': 'tool', 'tool_call_id': 'call1', 'content': '{"content":"private"}'}]
    assert request(messages).messages[-1].content == '{"content":"private"}'
    with pytest.raises(ValidationError):
        request([{'role': 'user', 'content': 'x' * 150000}, {'role': 'assistant', 'content': 'y' * 150000}])


def test_http_stream_runs_context_in_one_worker(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    class Model:
        def bind_tools(self, tools, **kwargs):
            return self
        def stream(self, messages, config):
            assert current_model_call_context().user_id == '42'
            yield AIMessageChunk(content='已查看。')
    monkeypatch.setattr(module, 'build_chat_model_for_task', lambda *a, **k: Model())
    app = FastAPI()
    app.include_router(module.router)
    with TestClient(app) as client:
        response = client.post('/local-workspace/step/stream', json=request().model_dump())
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    assert 'event: result' in response.text and 'event: error' not in response.text
