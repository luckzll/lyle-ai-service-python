from __future__ import annotations

import base64
import mimetypes
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import httpx

from app.core.config import settings


def analyze_image_from_question(question: str) -> dict[str, object]:
    image_ref = _extract_image_ref(question)
    if not image_ref:
        return {
            "ok": False,
            "content": "没有收到可识别的图片。请先上传图片后再提问。",
        }
    if not _image_api_key():
        return {"ok": False, "content": "图片识别模型 API Key 未配置。"}

    image_url = _to_image_url(image_ref)
    prompt = _clean_analyze_prompt(question, image_ref)
    payload = {
        "model": settings.vision_model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "请描述这张图片的内容。"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }
    data = _post_json("/chat/completions", payload)
    content = _extract_chat_content(data)
    return {
        "ok": bool(content),
        "content": content or "图片识别没有返回内容。",
        "imageRef": image_ref,
        "model": settings.vision_model_name,
    }


def generate_image_from_question(question: str) -> dict[str, object]:
    if not _image_api_key():
        return {"ok": False, "content": "生图模型 API Key 未配置。"}

    prompt = _clean_generate_prompt(question)
    payload = {
        "model": settings.image_model_name,
        "prompt": prompt,
        "size": "2048x2048",
        "response_format": "b64_json",
        "watermark": settings.image_watermark,
    }
    data = _post_json("/images/generations", payload, timeout=120)
    images = _extract_generated_images(data)
    if not images:
        return {
            "ok": False,
            "content": "生图模型没有返回图片。",
            "raw": data,
            "model": settings.image_model_name,
        }
    first = images[0]
    url = first.get("url") or ""
    b64 = first.get("b64_json") or ""
    image_src, storage_used, storage_error = _publish_generated_image_result(b64, url)
    content = "图片已生成。"
    if url:
        content += f"\n\n原图链接：{url}"
    if storage_error:
        content += f"\n\n图片上传 TOS 失败，已回退到本地存储：{storage_error}"
    if image_src:
        content += f"\n\n![生成图片]({image_src})"
    return {
        "ok": True,
        "content": content,
        "prompt": prompt,
        "images": images,
        "model": settings.image_model_name,
        "imageUrl": image_src,
        "storageRequested": settings.generated_image_storage,
        "storageUsed": storage_used,
        "storageError": storage_error,
    }


def edit_image_from_question(question: str) -> dict[str, object]:
    if not _image_api_key():
        return {"ok": False, "content": "图片编辑模型 API Key 未配置。"}

    image_ref = _extract_image_ref(question)
    if not image_ref:
        return {"ok": False, "content": "没有收到要修改的图片。请先上传图片，并说明想怎么改。"}

    image_url = _to_image_url(image_ref)
    prompt = _clean_edit_prompt(question, image_ref)
    payload = {
        "model": settings.image_model_name,
        "prompt": prompt,
        "size": "2048x2048",
        "response_format": "b64_json",
        "watermark": settings.image_watermark,
    }
    data = _post_image_edit_json(payload, image_url)
    images = _extract_generated_images(data)
    if not images:
        return {
            "ok": False,
            "content": "图片编辑模型没有返回图片。",
            "raw": data,
            "model": settings.image_model_name,
            "sourceImageRef": image_ref,
        }

    first = images[0]
    url = first.get("url") or ""
    b64 = first.get("b64_json") or ""
    image_src, storage_used, storage_error = _publish_generated_image_result(b64, url)
    content = "图片已修改。"
    if storage_error:
        content += f"\n\n图片上传 TOS 失败，已回退到本地存储：{storage_error}"
    if image_src:
        content += f"\n\n![修改后图片]({image_src})"
    return {
        "ok": True,
        "content": content,
        "prompt": prompt,
        "images": images,
        "model": settings.image_model_name,
        "imageUrl": image_src,
        "sourceImageRef": image_ref,
        "storageRequested": settings.generated_image_storage,
        "storageUsed": storage_used,
        "storageError": storage_error,
    }


def _post_json(path: str, payload: dict[str, object], timeout: float | None = None) -> dict[str, object]:
    endpoint = f"{settings.ark_base_url.rstrip('/')}{path}"
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
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:1000] if exc.response is not None else str(exc)
        raise RuntimeError(f"图片模型请求失败：HTTP {exc.response.status_code} {detail}") from exc
    return data if isinstance(data, dict) else {}


def _post_image_edit_json(payload: dict[str, object], image_url: str) -> dict[str, object]:
    candidates = [
        {"image": [image_url]},
        {"image_urls": [image_url]},
        {"image_input": [image_url]},
        {"images": [image_url]},
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


def _extract_image_ref(question: str) -> str:
    data_url_match = re.search(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", question)
    if data_url_match:
        return re.sub(r"\s+", "", data_url_match.group(0))
    url_match = re.search(r"https?://\S+\.(?:png|jpe?g|webp|gif)(?:\?\S*)?", question, re.IGNORECASE)
    if url_match:
        return url_match.group(0).rstrip("，。,.!！)")
    path_match = re.search(r"(/[^\s，。,.!！?？]+\.(?:png|jpe?g|webp|gif))", question, re.IGNORECASE)
    if path_match:
        return path_match.group(1)
    return ""


def _to_image_url(image_ref: str) -> str:
    if image_ref.startswith(("http://", "https://", "data:")):
        return image_ref
    path = Path(image_ref).expanduser()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _clean_analyze_prompt(question: str, image_ref: str) -> str:
    return question.replace(image_ref, "").strip(" ：:，。")


def _clean_generate_prompt(question: str) -> str:
    cleaned = question.strip()
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


def _clean_edit_prompt(question: str, image_ref: str) -> str:
    cleaned = question.replace(image_ref, "")
    cleaned = re.sub(r"图片\s*[：:]\s*", "", cleaned)
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


def _image_api_key() -> str:
    return settings.ark_api_key
