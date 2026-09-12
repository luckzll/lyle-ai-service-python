"""Identity of a citable passage, including its location when available."""
from collections.abc import Mapping
from typing import Any


def _value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _field(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = _value(row.get(name))
        if value:
            return value
    return ""


def citation_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    chunk = row.get("chunk")
    base = (_field(row, "knowledge_base_id"), _field(row, "doc_id"), str(chunk) if chunk is not None else "")
    location = (
        _field(row, "chunkIndex", "chunk_index"),
        _field(row, "startChar", "start_char"),
        _field(row, "endChar", "end_char"),
        _field(row, "pageNumber", "page", "page_number"),
    )
    # Old sources with no location still deduplicate as before. A located and
    # an unlocated source remain separate: we cannot assume the same occurrence.
    return (*base, *location) if any(location) else base
