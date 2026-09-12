"""Deterministic retrieval context, kept separate from citable document text.

Only the supplied document name, section name and explicit metadata at the
document's beginning are copied. No model, entity inference or neighboring
document is used. Old chunks remain on their original retrieval representation.
"""
from dataclasses import dataclass
import re


CONTEXT_VERSION = "deterministic-v1"
LEGACY_CONTEXT_VERSION = "raw-v1"
_METADATA = re.compile(
    r"^(?:产品(?:名称)?|项目(?:名称)?|文档(?:名称|版本)?|适用(?:产品|项目|版本)|版本(?:号)?|"
    r"product|project|document(?:[ _]title)?|version)\s*[:：]\s*\S", re.IGNORECASE,
)


@dataclass(frozen=True)
class ChunkContext:
    retrieval_text: str
    version: str
    sources: tuple[str, ...]


def document_metadata(text: str) -> tuple[str, ...]:
    """Copy explicit leading metadata, never versions from unrelated sections."""
    result = []
    title_seen = False
    for raw in text[:2400].splitlines()[:24]:
        line = raw.strip()
        if not line or line == "---":
            continue
        if _METADATA.match(line):
            if len(line) <= 180 and line not in result:
                result.append(line)
            if len(result) >= 4:
                break
            continue
        # A single Markdown document title may precede the metadata block.
        if not result and not title_seen and re.match(r"^#\s+[^#]", line):
            title_seen = True
            continue
        break
    return tuple(result)


def build_chunk_context(document_title: str, section_title: str, raw_chunk: str,
                        metadata: tuple[str, ...] = (), *, enabled: bool = True,
                        max_chars: int = 480) -> ChunkContext:
    if not enabled:
        return ChunkContext("", LEGACY_CONTEXT_VERSION, ())
    # Keep the prefix small for the configured 512-token embedding model. This
    # bounds context characters; it is not a tokenizer-specific token guarantee.
    budget = max(0, min(160, int(max_chars)))
    candidates = [("document_title", "文档", document_title)]
    if section_title and section_title not in {"文档内容", "长段落"}:
        candidates.append(("section_title", "章节", section_title))
    candidates.extend(("document_metadata", "文首原文", value) for value in metadata)
    lines, sources = [], []
    used = 0
    for source, label, value in candidates:
        # Whitespace normalization is only for copied context, never raw_chunk.
        value = " ".join(str(value).split())
        if not value:
            continue
        line = f"{label}：{value}"
        if used + len(line) + 1 > budget:
            continue  # Do not truncate version identifiers or factual clauses.
        lines.append(line)
        used += len(line) + 1
        if source not in sources:
            sources.append(source)
    if not lines:
        return ChunkContext("", LEGACY_CONTEXT_VERSION, ())
    return ChunkContext("\n".join(lines) + "\n\n原文片段：\n" + raw_chunk, CONTEXT_VERSION, tuple(sources))


def keyword_text(chunk) -> str:
    # Preserve the legacy title/heading formula until a document is re-ingested.
    return getattr(chunk, "retrieval_text", "") or f"{chunk.title}\n{chunk.parent_title}\n{chunk.chunk}"


def embedding_text(chunk) -> str:
    return getattr(chunk, "retrieval_text", "") or chunk.chunk
