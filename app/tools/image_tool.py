from __future__ import annotations

import base64
import mimetypes
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextvars import copy_context
from pathlib import Path
from typing import Any, Iterator

import httpx

from app.core.config import settings
from app.llm.prompts import FINAL_ANSWER_INSTRUCTIONS
from app.tools.image_intent import is_image_generation_request
from app.usage.manual import begin_external_call, estimate_text_tokens, finish_external_call


def analyze_image_from_question(question: str) -> dict[str, object]:
    image_refs = extract_image_refs(question)
    if not image_refs:
        return {
            "ok": False,
            "content": "没有收到可识别的图片。请先上传图片后再提问。",
        }
    if not _image_api_key():
        return {"ok": False, "content": "图片识别模型 API Key 未配置。"}

    image_urls = [_to_image_url(value) for value in image_refs]
    prompt = _clean_analyze_prompt(question, image_refs)
    payload = {
        "model": settings.vision_model_name,
        "messages": [
            {"role": "system", "content": FINAL_ANSWER_INSTRUCTIONS + (
                "当前仅执行图片识别与分析，没有生成或编辑图片的能力。"
                "不得声称已生成、修改或上传图片，不得编造图片链接。"
            )},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "请综合描述这些图片的内容、共同点和差异。"},
                    *[
                        {"type": "image_url", "image_url": {"url": image_url}}
                        for image_url in image_urls
                    ],
                ],
            }
        ],
    }
    data = _post_json("/chat/completions", payload)
    content = _extract_chat_content(data)
    return {
        "ok": bool(content),
        "content": content or "图片识别没有返回内容。",
        "imageRef": image_refs[0],
        "imageRefs": image_refs,
        "model": settings.vision_model_name,
    }


def generate_image_from_question(question: str) -> dict[str, object]:
    if not _image_api_key():
        return {"ok": False, "content": "生图模型 API Key 未配置。"}
    spec = build_image_batch_spec(question, "generate")
    return build_image_batch_result(spec, list(iter_image_batch_results(spec)))


def edit_image_from_question(question: str) -> dict[str, object]:
    if not _image_api_key():
        return {"ok": False, "content": "图片编辑模型 API Key 未配置。"}

    image_refs = extract_image_refs(question)
    if not image_refs:
        return {"ok": False, "content": "没有收到要修改的图片。请先上传图片，并说明想怎么改。"}
    spec = build_image_batch_spec(question, "edit")
    return build_image_batch_result(spec, list(iter_image_batch_results(spec)))


def build_image_batch_spec(question: str, mode: str) -> dict[str, Any]:
    normalized_mode = "edit" if mode == "edit" else "generate"
    refs = extract_image_refs(question)
    max_count = max(1, min(int(settings.image_batch_max_count), 16))
    count = parse_image_count(question, max_count=max_count)
    prompt = (
        _clean_edit_prompt(question, refs)
        if normalized_mode == "edit"
        else _clean_generate_prompt(question)
    )
    return {
        "batchId": uuid.uuid4().hex,
        "mode": normalized_mode,
        "prompt": prompt,
        "count": count,
        "referenceImageRefs": refs,
        "referenceImageUrls": [_to_image_url(value) for value in refs],
        "model": settings.image_model_name,
        "size": "2048x2048",
    }


def iter_image_batch_results(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    count = max(1, int(spec.get("count") or 1))
    concurrency = max(1, min(int(settings.image_batch_concurrency), count, 32))
    executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="image-batch")
    futures: dict[Future[dict[str, Any]], int] = {
        executor.submit(copy_context().run, _execute_image_item, spec, index): index
        for index in range(count)
    }
    try:
        for future in as_completed(futures):
            index = futures[future]
            try:
                yield future.result()
            except Exception as exc:
                yield {
                    "index": index,
                    "status": "FAILED",
                    "error": _safe_error(exc),
                }
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def build_image_batch_result(
    spec: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, object]:
    ordered = sorted(items, key=lambda item: int(item.get("index") or 0))
    completed = [item for item in ordered if item.get("status") == "COMPLETED" and item.get("imageUrl")]
    failed = [item for item in ordered if item.get("status") != "COMPLETED"]
    mode = str(spec.get("mode") or "generate")
    action = "修改" if mode == "edit" else "生成"
    refs = list(spec.get("referenceImageRefs") or [])
    if not completed:
        return {
            "ok": False,
            "content": f"图片{action}失败，没有获得可用图片。",
            "prompt": spec.get("prompt"),
            "images": ordered,
            "imageUrls": [],
            "model": settings.image_model_name,
            "batchId": spec.get("batchId"),
            "batchStatus": "FAILED",
            "requestedCount": spec.get("count"),
            "completedCount": 0,
            "failedCount": len(failed),
            "sourceImageRef": refs[0] if refs else None,
            "sourceImageRefs": refs,
        }

    image_urls = [str(item["imageUrl"]) for item in completed]
    summary = f"已{action} {len(completed)} 张图片"
    if failed:
        summary += f"，另有 {len(failed)} 张失败"
    content = summary + "。"
    base_alt = "修改后图片" if mode == "edit" else "生成图片"
    content += "".join(
        f"\n\n![{base_alt if len(image_urls) == 1 else f'{base_alt} {index + 1}'}]({url})"
        for index, url in enumerate(image_urls)
    )
    storage_errors = [
        str(item.get("storageError") or "")
        for item in completed
        if str(item.get("storageError") or "")
    ]
    return {
        "ok": True,
        "partial": bool(failed),
        "content": content,
        "prompt": spec.get("prompt"),
        "images": ordered,
        "imageUrl": image_urls[0],
        "imageUrls": image_urls,
        "model": settings.image_model_name,
        "batchId": spec.get("batchId"),
        "batchStatus": "PARTIALLY_COMPLETED" if failed else "COMPLETED",
        "requestedCount": spec.get("count"),
        "completedCount": len(completed),
        "failedCount": len(failed),
        "sourceImageRef": refs[0] if refs else None,
        "sourceImageRefs": refs,
        "storageRequested": settings.generated_image_storage,
        "storageUsed": completed[0].get("storageUsed"),
        "storageError": "；".join(storage_errors),
    }


def _execute_image_item(spec: dict[str, Any], index: int) -> dict[str, Any]:
    prompt = str(spec.get("prompt") or "")
    if int(spec.get("count") or 1) > 1:
        prompt += (
            f"\n这是同一批次中的第 {index + 1} 张，请保持核心要求一致，"
            "同时让构图与细节具有独立变化。"
        )
    payload: dict[str, object] = {
        "model": settings.image_model_name,
        "prompt": prompt,
        "size": str(spec.get("size") or "2048x2048"),
        "response_format": "b64_json",
        "watermark": settings.image_watermark,
    }
    max_attempts = max(1, min(int(settings.image_batch_retry_attempts) + 1, 4))
    data: dict[str, object] = {}
    for attempt in range(max_attempts):
        try:
            if spec.get("mode") == "edit":
                data = _post_image_edit_json(payload, list(spec.get("referenceImageUrls") or []))
            else:
                data = _post_json("/images/generations", payload, timeout=120)
            break
        except Exception as exc:
            if attempt + 1 >= max_attempts or not _is_retryable_image_error(exc):
                raise
            delay = max(0.0, float(settings.image_batch_retry_base_seconds)) * (2**attempt)
            time.sleep(delay)
    images = _extract_generated_images(data)
    if not images:
        raise RuntimeError("生图模型没有返回图片。")
    first = images[0]
    remote_url = first.get("url") or ""
    b64 = first.get("b64_json") or ""
    image_src, storage_used, storage_error = _publish_generated_image_result(b64, remote_url)
    if not image_src:
        raise RuntimeError("图片生成成功，但结果无法持久化。")
    return {
        "index": index,
        "status": "COMPLETED",
        "imageUrl": image_src,
        "remoteUrl": remote_url,
        "revisedPrompt": first.get("revisedPrompt") or "",
        "storageUsed": storage_used,
        "storageError": storage_error,
    }


def _estimate_image_request_tokens(payload: dict[str, object]) -> int:
    """Reserve text plus a conservative per-image allowance, never Base64 bytes.

    Image encoding length is unrelated to vision tokens. The allowance is only
    for preflight budgeting; provider-reported usage remains authoritative.
    """
    tokens = estimate_text_tokens(str(payload.get("prompt") or ""))
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            tokens += estimate_text_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    tokens += estimate_text_tokens(str(part.get("text") or ""))
                elif part.get("type") == "image_url":
                    tokens += 4096
    return tokens


def _post_json(path: str, payload: dict[str, object], timeout: float | None = None) -> dict[str, object]:
    endpoint = f"{settings.ark_base_url.rstrip('/')}{path}"
    is_image_call = path.startswith("/images/")
    estimated_input = _estimate_image_request_tokens(payload)
    reserved_cost = (
        settings.ai_usage_image_cost_micros_per_call
        if is_image_call
        else round(estimated_input * settings.ai_usage_input_price_cny_per_million)
    )
    ticket = begin_external_call(
        estimated_input_tokens=estimated_input,
        estimated_cost_micros=reserved_cost,
    )
    task_type = (
        "IMAGE_EDIT"
        if is_image_call and any(key in payload for key in ("image", "image_urls", "image_input", "images"))
        else "IMAGE_GENERATE"
        if is_image_call
        else "IMAGE_ANALYZE"
    )
    try:
        with httpx.Client(timeout=timeout or settings.llm_timeout_seconds) as client:
            response = client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {_image_api_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        finish_external_call(
            ticket,
            call_type="IMAGE_GENERATION" if is_image_call else "LLM_VISION",
            task_type=task_type,
            provider=settings.image_model_provider if is_image_call else settings.vision_model_provider,
            model=str(payload.get("model") or settings.image_model_name),
            input_tokens=estimated_input,
            estimated_cost_micros=reserved_cost,
            status="FAILED",
            error_category=type(exc).__name__,
        )
        if isinstance(exc, httpx.HTTPStatusError):
            detail = exc.response.text[:1000] if exc.response is not None else str(exc)
            raise RuntimeError(f"图片模型请求失败：HTTP {exc.response.status_code} {detail}") from exc
        raise
    usage = data.get("usage") if isinstance(data, dict) else {}
    provider_input = int((usage or {}).get("prompt_tokens") or (usage or {}).get("input_tokens") or 0)
    provider_output = int((usage or {}).get("completion_tokens") or (usage or {}).get("output_tokens") or 0)
    finish_external_call(
        ticket,
        call_type="IMAGE_GENERATION" if is_image_call else "LLM_VISION",
        task_type=task_type,
        provider=settings.image_model_provider if is_image_call else settings.vision_model_provider,
        model=str(payload.get("model") or settings.image_model_name),
        input_tokens=provider_input or estimated_input,
        output_tokens=provider_output,
        usage_source="PROVIDER_REPORTED" if provider_input or provider_output else "ESTIMATED",
        estimated_cost_micros=(
            reserved_cost
            if is_image_call
            else round(
                (provider_input or estimated_input) * settings.ai_usage_input_price_cny_per_million
                + provider_output * settings.ai_usage_output_price_cny_per_million
            )
        ),
    )
    return data if isinstance(data, dict) else {}


def _post_image_edit_json(payload: dict[str, object], image_urls: list[str]) -> dict[str, object]:
    if not image_urls:
        raise RuntimeError("图片编辑至少需要一张参考图。")
    candidates = [
        {"image": image_urls},
        {"image_urls": image_urls},
        {"image_input": image_urls},
        {"images": image_urls},
    ]
    last_error = ""
    for candidate in candidates:
        try:
            return _post_json("/images/generations", {**payload, **candidate}, timeout=120)
        except RuntimeError as exc:
            message = str(exc)
            last_error = message
            if "HTTP 400" in message or "HTTP 422" in message:
                continue
            raise
    raise RuntimeError(f"图片编辑请求失败：{last_error}")


def _extract_chat_content(data: dict[str, object]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "").strip()
    return str(first.get("text") or "").strip()


def _extract_generated_images(data: dict[str, object]) -> list[dict[str, str]]:
    items = data.get("data")
    if not isinstance(items, list):
        return []
    images: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        b64 = str(item.get("b64_json") or item.get("base64") or "").strip()
        revised_prompt = str(item.get("revised_prompt") or "").strip()
        image: dict[str, str] = {}
        if url:
            image["url"] = url
        if b64:
            image["b64_json"] = b64
        if revised_prompt:
            image["revisedPrompt"] = revised_prompt
        if image:
            images.append(image)
    return images


def _publish_generated_image(b64: str) -> tuple[str, str, str]:
    if not b64:
        return "", "none", ""
    if settings.generated_image_storage.lower() == "tos":
        try:
            return _upload_generated_image_to_tos(b64), "tos", ""
        except Exception as exc:
            local_url = _save_generated_image_locally(b64)
            return local_url, "local", str(exc)
    return _save_generated_image_locally(b64), "local", ""


def _publish_generated_image_result(b64: str, url: str) -> tuple[str, str, str]:
    if b64:
        return _publish_generated_image(b64)
    if not url:
        return "", "none", ""
    if settings.generated_image_storage.lower() == "tos":
        try:
            return _upload_remote_image_to_tos(url), "tos", ""
        except Exception as exc:
            return url, "remote", str(exc)
    return url, "remote", ""


def _upload_remote_image_to_tos(url: str) -> str:
    if not settings.tos_bucket:
        raise RuntimeError("TOS bucket 未配置。")
    if not settings.tos_public_url_head:
        raise RuntimeError("TOS public URL head 未配置。")

    with httpx.Client(timeout=settings.tos_upload_timeout_seconds) as client:
        response = client.get(url)
        response.raise_for_status()
        image_bytes = response.content
    filename = f"{uuid.uuid4().hex}.jpg"
    object_key = _join_tos_key(settings.tos_save_path_head, filename)
    if settings.tos_access_key_id and settings.tos_secret_access_key and settings.tos_endpoint:
        _upload_generated_image_to_tos_sdk(image_bytes, object_key)
        return f"{settings.tos_public_url_head.rstrip('/')}/{object_key}"

    _upload_generated_image_to_tosutil(image_bytes, object_key)
    return f"{settings.tos_public_url_head.rstrip('/')}/{object_key}"


def _save_generated_image_locally(b64: str) -> str:
    if not b64:
        return ""
    image_dir = Path(settings.generated_image_dir).expanduser()
    image_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.jpg"
    (image_dir / filename).write_bytes(_decode_base64_image(b64))
    return f"{settings.generated_image_public_base_url.rstrip('/')}/{filename}"


def _upload_generated_image_to_tos(b64: str) -> str:
    if not settings.tos_bucket:
        raise RuntimeError("TOS bucket 未配置。")
    if not settings.tos_public_url_head:
        raise RuntimeError("TOS public URL head 未配置。")

    filename = f"{uuid.uuid4().hex}.jpg"
    object_key = _join_tos_key(settings.tos_save_path_head, filename)
    image_bytes = _decode_base64_image(b64)
    if settings.tos_access_key_id and settings.tos_secret_access_key and settings.tos_endpoint:
        _upload_generated_image_to_tos_sdk(image_bytes, object_key)
        return f"{settings.tos_public_url_head.rstrip('/')}/{object_key}"

    _upload_generated_image_to_tosutil(image_bytes, object_key)
    return f"{settings.tos_public_url_head.rstrip('/')}/{object_key}"


def _upload_generated_image_to_tos_sdk(image_bytes: bytes, object_key: str) -> None:
    try:
        import tos
    except ImportError as exc:
        raise RuntimeError("TOS Python SDK 未安装，请执行 pip install tos。") from exc

    client = tos.TosClientV2(
        settings.tos_access_key_id,
        settings.tos_secret_access_key,
        settings.tos_endpoint,
        settings.tos_region,
        request_timeout=settings.tos_upload_timeout_seconds,
    )
    client.put_object(
        bucket=settings.tos_bucket,
        key=object_key,
        content=image_bytes,
        content_type="image/jpeg",
        acl=tos.ACLType.ACL_Public_Read,
    )


def _upload_generated_image_to_tosutil(image_bytes: bytes, object_key: str) -> None:
    tosutil = shutil.which(settings.tosutil_path) or settings.tosutil_path
    if not Path(tosutil).exists() and "/" in tosutil:
        raise RuntimeError(f"tosutil 不存在：{settings.tosutil_path}")
    if shutil.which(tosutil) is None and "/" not in tosutil:
        raise RuntimeError(f"tosutil 不在 PATH 中：{settings.tosutil_path}")

    tos_uri = f"tos://{settings.tos_bucket}/{object_key}"
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as temp:
        temp.write(image_bytes)
        temp.flush()
        command = [tosutil, "cp", temp.name, tos_uri]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=settings.tos_upload_timeout_seconds,
            check=False,
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or f"tosutil 上传失败，exit={result.returncode}")


def _join_tos_key(prefix: str, filename: str) -> str:
    normalized = prefix.strip("/")
    return f"{normalized}/{filename}" if normalized else filename


def _decode_base64_image(b64: str) -> bytes:
    if not b64:
        return b""
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[-1]
    return base64.b64decode(b64)


def extract_image_refs(question: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(
        r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+",
        question,
    ):
        values.append(match.group(0))
    for match in re.finditer(
        r"https?://[^\s，。!！)）]+\.(?:png|jpe?g|webp|gif)(?:\?[^\s，。!！)）]*)?",
        question,
        re.IGNORECASE,
    ):
        values.append(match.group(0).rstrip("，。,.!！)）"))
    for match in re.finditer(
        r"(?<![A-Za-z0-9:])(/[^\s，。,.!！?？]+\.(?:png|jpe?g|webp|gif))",
        question,
        re.IGNORECASE,
    ):
        values.append(match.group(1))
    deduplicated: list[str] = []
    for value in values:
        if value not in deduplicated:
            deduplicated.append(value)
    max_refs = max(1, min(int(settings.image_reference_max_count), 16))
    return deduplicated[:max_refs]


def _extract_image_ref(question: str) -> str:
    refs = extract_image_refs(question)
    return refs[0] if refs else ""


def parse_image_count(question: str, max_count: int | None = None) -> int:
    upper = max(1, max_count or int(settings.image_batch_max_count))
    number_token = r"(\d{1,2}|[一两二三四五六七八九十])"
    requested = re.search(
        rf"(?:生成|生图|画|输出|制作|做)(?:[^，。,\n]{{0,12}}?){number_token}\s*张",
        question,
    )
    if requested:
        return max(1, min(_parse_image_count_token(requested.group(1)), upper))
    all_counts = re.findall(rf"{number_token}\s*张", question)
    if all_counts:
        return max(1, min(_parse_image_count_token(all_counts[-1]), upper))
    if (
        is_image_generation_request(question)
        and any(value in question for value in ["批量生图", "批量生成", "多张图片", "多张图"])
    ):
        return min(4, upper)
    return 1


def _parse_image_count_token(value: str) -> int:
    chinese_numbers = {
        "一": 1,
        "两": 2,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    return int(value) if value.isdigit() else chinese_numbers.get(value, 1)


def _to_image_url(image_ref: str) -> str:
    if image_ref.startswith(("http://", "https://", "data:")):
        return image_ref
    path = Path(image_ref).expanduser()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _clean_analyze_prompt(question: str, image_refs: list[str]) -> str:
    cleaned = question
    for image_ref in image_refs:
        cleaned = cleaned.replace(image_ref, "")
    return re.sub(r"图片\s*\d*\s*[：:]\s*", "", cleaned).strip(" ：:，。\n")


def _clean_generate_prompt(question: str) -> str:
    cleaned = question.strip()
    cleaned = re.sub(
        r"^(?:请|帮我|给我)?\s*(?:生成|生图|画|输出|制作|做)\s*"
        r"(?:\d{1,2}|[一两二三四五六七八九十])\s*张\s*",
        "",
        cleaned,
    )
    prefixes = [
        "帮我生成一张",
        "生成一张",
        "帮我画一张",
        "画一张",
        "帮我画",
        "画一个",
        "画",
        "生图",
        "生成图片",
    ]
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip(" ：:，。")
            break
    return cleaned or question.strip()


def _clean_edit_prompt(question: str, image_refs: list[str]) -> str:
    cleaned = question
    for image_ref in image_refs:
        cleaned = cleaned.replace(image_ref, "")
    cleaned = re.sub(r"图片\s*\d*\s*[：:]\s*", "", cleaned)
    cleaned = re.sub(
        r"(?:生成|生图|输出|制作|做)\s*"
        r"(?:\d{1,2}|[一两二三四五六七八九十])\s*张\s*",
        "",
        cleaned,
    )
    cleaned = cleaned.strip(" ：:，。\n")
    prefixes = [
        "帮我改一下",
        "帮我修改",
        "修改图片",
        "编辑图片",
        "改图",
        "图生图",
        "把这张图",
        "把这个图",
        "把图片",
        "请把",
    ]
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip(" ：:，。")
            break
    return cleaned or "请根据用户要求编辑这张图片。"


def _safe_error(exc: Exception) -> str:
    value = re.sub(r"(?i)(api[_ -]?key|authorization|token)\s*[:=]\s*\S+", r"\1=[REDACTED]", str(exc))
    return value[:1000] or exc.__class__.__name__


def _is_retryable_image_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.RequestError):
        return True
    return bool(re.search(r"HTTP\s+(?:408|429|5\d\d)\b", str(exc), re.IGNORECASE))


def _image_api_key() -> str:
    return settings.ark_api_key
