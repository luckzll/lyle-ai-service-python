from app.memory.overview import build_memory_overview_answer
from app.tools.tool_registry import select_tools


def _mixed_state():
    memory = {
        "profiles": [{
            "id": 11,
            "schemaPath": "preferences.visual.dislikes",
            "label": "视觉负向偏好",
            "value": "帽子",
            "summary": "用户在视觉形象上不喜欢帽子。",
            "status": "active",
            "confidence": 0.95,
        }],
        "structured": [],
    }
    tool_results = [{
        "tool": "work_summary",
        "ok": True,
        "summary": {
            "todos": {
                "items": [{"id": 22, "title": "V2 验收任务", "status": "open"}],
            },
            "operations": {
                "items": [{
                    "id": 33,
                    "summary": "创建日程「验收复盘」",
                    "operationType": "calendar.create",
                    "status": "rejected",
                }],
            },
        },
    }]
    return memory, tool_results


def test_memory_overview_formats_each_object_with_its_actual_status():
    memory, tool_results = _mixed_state()

    answer = build_memory_overview_answer("你记得我什么？", memory, tool_results)

    assert "帽子" in answer
    assert "已确认记忆" in answer
    assert "V2 验收任务（进行中）" in answer
    assert "创建日程「验收复盘」（已拒绝，未执行）" in answer
    assert "待确认状态" not in answer
    assert "确认后才会正式生效" not in answer


def test_memory_overview_keeps_genuinely_pending_operation_pending():
    memory, tool_results = _mixed_state()
    tool_results[0]["summary"]["operations"]["items"][0]["status"] = "pending"

    answer = build_memory_overview_answer("记忆概览", memory, tool_results)

    assert "创建日程「验收复盘」（待确认，尚未执行）" in answer
    assert "帽子" in answer and "已确认记忆" in answer


def test_memory_overview_uses_deterministic_memory_and_work_tools():
    assert select_tools("你记得我什么？") == ["memory_snapshot", "work_summary"]
