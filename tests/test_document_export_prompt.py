from app.llm.prompts import build_user_prompt


def test_document_export_explains_actual_frontend_capabilities_without_claiming_disk_write():
    prompt = build_user_prompt('把刚才的整理做成 Word 放桌面', [], '')
    assert '保存为文件' in prompt
    assert '完整文档正文' in prompt
    assert '选择位置并保存' in prompt
    assert '不得声称' in prompt
    assert '不支持生成 PPT' in prompt
