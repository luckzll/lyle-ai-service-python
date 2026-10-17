from uuid import uuid4

import pytest

from app.core.config import settings
from app.memory import store


@pytest.fixture(autouse=True)
def isolated_memory(monkeypatch):
    for flag in ('postgres_enabled', 'redis_enabled', 'qdrant_enabled', 'milvus_enabled'):
        monkeypatch.setattr(settings, flag, False)


@pytest.mark.parametrize(('text', 'key', 'value'), [
    ('我是程序员，有什么副业适合我吗？', 'career', '程序员'),
    ('怎么学习 AI？我是程序员。', 'career', '程序员'),
    ('我喜欢西瓜，有什么挑选建议？', 'food_preference', '西瓜'),
    ('请记住：我不喜欢帽子，可以吗？', 'negative_preference', '帽子'),
    ('我的说话风格是简短直接。以后能这样回答吗？', 'preferences.communication_style', '简短直接'),
    ('我正在做模型量化，帮我创建待办。', 'current_project', '模型量化'),
])
def test_explicit_fact_survives_an_adjacent_question_or_task(text, key, value):
    candidates = store._extract_structured_candidates(text, extraction_mode='rule')
    assert any(item_key == key and item_value == value for item_key, item_value, _ in candidates)


@pytest.mark.parametrize('text', [
    '我是程序员吗？',
    '你觉得我喜欢西瓜吗？',
    '我的职业是什么',
    '如果我是程序员，我喜欢西瓜，有什么建议吗？',
    '请假设我是程序员，我喜欢西瓜，有什么建议？',
    '假装我是程序员，我喜欢西瓜，有什么建议？',
    '小明说：我是程序员，我喜欢西瓜，怎么回复？',
    '小明说，\n我是程序员。\n我喜欢西瓜。\n怎么回复？',
    '翻译成英文：我是程序员，我喜欢西瓜，应该怎么说？',
    '示例：\n我是程序员。\n我喜欢西瓜。\n这段怎么翻译？',
    '附件原文：\n我是程序员。\n我喜欢西瓜。\n这段怎么翻译？',
    '创建待办：\n我想做一个个人 AI 助理',
    '他说“我是程序员，我喜欢西瓜”，这句话怎么理解？',
    '看看这段代码：```我是程序员，我喜欢西瓜```，是什么意思？',
    '你把近期比较火的开源模型的链接发给我，都是做什么的，然后我去看看一下',
])
def test_questions_and_source_material_do_not_become_personal_facts(monkeypatch, text):
    monkeypatch.setattr(store, 'invoke_structured_memory_chain',
                        lambda *args: pytest.fail('no personal assertion to extract'))
    assert store._extract_structured_candidates(text, extraction_mode='hybrid') == []


def test_mixed_question_keeps_preference_uncertainty():
    candidates = store._extract_structured_candidates(
        '我喜欢咖啡，但不太确定，有什么推荐？', extraction_mode='rule',
    )
    assert candidates == [('food_preference', '咖啡', .65)]


def test_model_receives_only_assertions_from_mixed_question(monkeypatch):
    received = []
    def extract(text, history):
        received.append(text)
        return [{'key': 'current_project', 'value': 'AI 智能体', 'confidence': .9}]
    monkeypatch.setattr(store, 'invoke_structured_memory_chain', extract)
    candidates = store._extract_structured_candidates(
        '最近在做 AI 智能体，怎么推进比较好？', extraction_mode='llm',
    )
    assert candidates == [('current_project', 'AI 智能体', .9)]
    assert len(received) == 1
    assert '最近在做 AI 智能体' in received[0]
    assert '怎么推进' not in received[0]


def test_pending_candidate_keeps_original_message_as_evidence():
    user_id = 'mixed-evidence-' + uuid4().hex
    text = '请记住：我不喜欢帽子，可以吗？'
    candidates = store.extract_structured_memory(
        user_id, 'conversation-one', text, extraction_mode='rule', source_message_id='message-one',
    )
    assert len(candidates) == 1
    assert candidates[0]['status'] == 'pending'
    assert candidates[0]['source'] == f'source_ref:conversation:conversation-one;message:message-one\n{text}'
    assert store.recall_memory(user_id, 'new-conversation', '我的偏好', include_vector=False)['profiles'] == []


def test_explicit_remember_instruction_can_use_model_without_first_person(monkeypatch):
    monkeypatch.setattr(store, 'invoke_structured_memory_chain', lambda *args: [
        {'key': 'preferences.output_style', 'value': '回答简短直接', 'confidence': .9},
    ])
    candidates = store._extract_structured_candidates('请记住：以后回答简短直接。', extraction_mode='llm')
    assert any(key == 'preferences.output_style' for key, _, _ in candidates)
