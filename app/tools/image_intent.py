import re


_IMAGE_COMMAND = re.compile(
    r"(?:^|[，,。；;！!\n])\s*"
    r"(?:好的[，,\s]*)?"
    r"(?:(?:请|麻烦)(?:帮我|给我|替我|为我)?|帮我|给我|替我|为我|我想|我要|能否|可以)?\s*"
    r"(?:(?:参考|基于|结合|按照).{0,36}?)?"
    r"(?:批量\s*)?"
    r"(?P<action>生成|生图|画|绘制|制作|设计|出图|重绘)"
    r"(?P<tail>[^。？！\n]{0,100})",
    re.IGNORECASE,
)

_IMAGE_TARGET = re.compile(
    r"(?:\d{1,2}|[一两二三四五六七八九十])\s*(?:张|幅)|"
    r"图片|图像|照片|海报|头像|壁纸|插画|封面|配图|效果图|宣传图|概念图",
    re.IGNORECASE,
)

_FOLLOWUP_COMMAND = re.compile(
    r"(?:^|[，,。；;！!\n])\s*(?:请|帮我|请帮我)?\s*"
    r"(?:再来|再生成|重新生成|换成|改成|换为|改为)"
    r"[^。？！\n]{0,80}"
    r"(?:图片|图像|照片|海报|头像|壁纸|插画|封面|猫|狗|小猫|小狗|风格|"
    r"(?:\d{1,2}|[一两二三四五六七八九十])\s*张)",
    re.IGNORECASE,
)

_DISCUSSION_PATTERNS = (
    re.compile(r"^(?:请[^，。]{0,10})?(?:说明|解释|介绍|分析|讨论|评价|总结).{0,30}(?:批量生图|生图|文生图|图片生成)"),
    re.compile(r"^(?:请问)?(?:批量生图|生图|文生图|图片生成).{0,24}(?:价值|意义|用途|应用|优缺点|原理|流程|功能|能力|是什么|什么意思)"),
    re.compile(r"^(?:请问)?(?:如何|怎么|为什么|是否|支持不支持|支持吗|能不能).{0,30}(?:批量生图|生图|文生图|图片生成)"),
    re.compile(r"(?:生成|制作|输出).{0,20}(?:张图片|张图).{0,10}(?:成本|价格|耗时|需要多久|是否|吗|\?)"),
)


def is_image_generation_request(question: str) -> bool:
    """Only return true for an explicit request to create image output.

    Mentions of image generation as a product concept must stay in normal chat so
    they cannot accidentally invoke a billable image model.
    """

    text = " ".join(str(question or "").strip().split())
    if not text:
        return False

    lowered = text.lower()
    if re.search(r"(?:^|[,.!;]\s*)(?:please\s+)?(?:generate|create|draw)\s+(?:an?\s+|\d+\s+)?image\b", lowered):
        return True
    if any(pattern.search(text) for pattern in _DISCUSSION_PATTERNS):
        return False
    if _FOLLOWUP_COMMAND.search(text):
        return True

    for matched in _IMAGE_COMMAND.finditer(text):
        action = matched.group("action")
        tail = matched.group("tail").strip()
        if action in {"生图", "画", "绘制", "出图", "重绘"}:
            return bool(tail) or action == "生图"
        if _IMAGE_TARGET.search(tail):
            return True
    return False
