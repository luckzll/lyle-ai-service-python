"""Conservative capability hints, independent from tool/data authorization.

Selecting a knowledge base makes it available; it does not make every request
about its documents. Obvious self-contained tasks need no specialist workers.
"""
import re

_MEMORY = re.compile(
    r"我(?:之前|以前|上次)|(?:你|还)记得|我的记忆|记忆(?:里|中|概览|快照|数量|统计)|"
    r"(?:查看|回顾|导出|删除|忘掉|检查)(?:我的|已保存的|长期)?记忆|"
    r"我的(?:偏好|喜好|兴趣|爱好|名字|职业|背景|习惯|历史|目标|计划|项目)|"
    r"我(?:有什么|有哪些)(?:兴趣|爱好|偏好|喜好)|"
    r"我(?:最)?(?:喜欢|讨厌)|我(?:叫什么|是谁)|上次|之前(?:说过|聊过|提到)|按我的风格|"
    r"\b(?:remember me|my (?:preferences|interests|hobbies|history|name|goals)|last time|previous conversations?)\b", re.I,
)
_KNOWLEDGE = re.compile(
    r"知识库|这(?:份|篇|个)文档|那(?:份|篇|个)文档|文档(?:中|里|检索)|资料(?:中|里)|"
    r"上一段|这部分|里面(?:有|的|提到|说)|继续.{0,8}文档|"
    r"(?:根据|依据|基于|按照|查阅|检索|搜索|查看).{0,16}(?:文档|资料|文件)|"
    r"附件|上传(?:的|了|文件)|\b(?:knowledge\s*base|attached|attachments?|uploaded|"
    r"(?:this|that|the) (?:document|file)|according to the documents?)\b", re.I,
)
_EXTERNAL = re.compile(r"联网|网上|网页|上网|互联网|\b(?:web|internet|online)\b", re.I)
_LIVE = re.compile(r"搜索|搜一下|查一下|最新|实时|新闻|天气|股价|汇率|\b(?:search|latest|news|weather)\b", re.I)
_GREETING = re.compile(r"(?:你好|您好|嗨|早上好|下午好|晚上好|谢谢|好的|随便聊聊|聊聊天|你是谁|你能做什么|你在干什么呢|hi|hello|thanks|ok)[\s!！。,.，?？]*", re.I)
_SELF_CONTAINED = re.compile(
    r"^(?:(?:请|帮我|麻烦)(?:你)?\s*)?(?:什么是|解释|介绍一下|科普|为什么|怎么理解|"
    r"计算|算一下|求导|积分|解方程|证明|改写|润色|翻译|写一首|写个笑话|"
    r"如何|怎样|怎么(?:使用|用|理解|实现|配置)|"
    r"what is\b|how to\b|explain\b|why\b|calculate\b|translate\b|rewrite\b|polish\b)", re.I,
)
_TOOL_ACTION = re.compile(
    r"创建|新建|添加|删除|修改|发送|安排|提醒|日程|待办|检查|诊断|调研|对比|汇总|"
    r"读取|查看|列出|调用|连接|导出|下载|生成图片|再来|再生成|继续|"
    r"\b(?:create|delete|update|send|schedule|remind|list|fetch|read|diagnose|research|compare|export|download)\b", re.I,
)
_TASK_CREATE = re.compile(r"^(?:(?:请|帮我|请帮我)\s*)?(?:创建|新建|新增|添加)(?:一个|一条)?(?:待办|任务|日程|提醒)")
_TRANSFORM = re.compile(r"翻译|改写|润色|\b(?:translate|rewrite|polish)\b", re.I)


def _request_text(question: str) -> str:
    # Text after a transformation delimiter is the material to translate/edit,
    # not another request to recall memories or execute the quoted actions.
    parts = re.split(r"[:：\n]", question, maxsplit=1)
    return parts[0] if len(parts) == 2 and _TRANSFORM.search(parts[0]) else question


def needs_personal_memory(question: str) -> bool:
    question = _request_text(question)
    return not _TASK_CREATE.search(question.strip()) and bool(_MEMORY.search(question))


def references_knowledge(question: str) -> bool:
    question = _request_text(question)
    return not _TASK_CREATE.search(question.strip()) and bool(_KNOWLEDGE.search(question))


def requests_external_tools(question: str) -> bool:
    question = _request_text(question)
    return bool(_EXTERNAL.search(question) or (_LIVE.search(question) and not references_knowledge(question)))


def is_self_contained(question: str) -> bool:
    value = question.strip()
    if not value:
        return True
    if needs_personal_memory(value) or references_knowledge(value) or requests_external_tools(value):
        return False
    return bool(_GREETING.fullmatch(value) or _SELF_CONTAINED.search(value)
                or re.fullmatch(r"[\d\s.+\-*/()=×÷%^]+(?:等于几|等于多少|是多少)?[?？。]*", value))


def needs_tool_planning(question: str, *, has_attachments: bool = False) -> bool:
    return has_attachments or (not is_self_contained(question) and bool(requests_external_tools(question)
                                                                      or _TOOL_ACTION.search(_request_text(question))))
