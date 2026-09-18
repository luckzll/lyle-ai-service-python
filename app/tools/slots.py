"""Typed extraction contracts. Identity and permissions never come from model arguments."""
from datetime import date as Date, datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class WeatherArgs(Arguments):
    city: str = Field(min_length=1, max_length=120)
    date: Date | None = None


class SearchArgs(Arguments):
    query: str = Field(min_length=1, max_length=2000)
    time_range: str | None = Field(default=None, max_length=120)


class MemoryArgs(Arguments):
    query: str = Field(default="", max_length=2000)


class WorkSummaryArgs(Arguments):
    query: str = Field(default="工作汇总", max_length=2000)


class FileArgs(Arguments):
    file_ref: str = Field(min_length=1, max_length=2048)
    operation: Literal["read", "parse"] = "parse"


class ImageArgs(Arguments):
    image_refs: list[str] = Field(min_length=1, max_length=5)
    instruction: str = Field(min_length=1, max_length=4000)


class ImageGenerateArgs(Arguments):
    prompt: str = Field(min_length=1, max_length=4000)


class TodoArgs(Arguments):
    title: str = Field(min_length=1, max_length=200)
    due_at: datetime | None = None

    @field_validator("due_at")
    @classmethod
    def aware(cls, value):
        if value is not None and value.tzinfo is None:
            raise ValueError("时间必须带时区；有歧义时请先澄清。")
        return value


class CalendarArgs(Arguments):
    title: str = Field(min_length=1, max_length=200)
    starts_at: datetime
    timezone: str = "Asia/Shanghai"
    participants: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("starts_at")
    @classmethod
    def aware(cls, value):
        return TodoArgs.aware(value)

    @field_validator("timezone")
    @classmethod
    def valid_zone(cls, value):
        ZoneInfo(value)
        return value


class ClarifyArgs(Arguments):
    question: str = Field(min_length=1, max_length=500)
    missing_slots: list[str] = Field(default_factory=list, max_length=20)


class EmptyArgs(Arguments):
    pass


class AttachmentListArgs(Arguments):
    offset: int = Field(default=0, ge=0, le=200)
    limit: int = Field(default=20, ge=1, le=50)


class AttachmentReadArgs(Arguments):
    attachmentId: str | None = Field(default=None, min_length=1, max_length=128)
    relativePath: str | None = Field(default=None, min_length=1, max_length=512)
    startLine: int = Field(default=1, ge=1, le=10_000_000)
    maxLines: int = Field(default=100, ge=1, le=200)
    startColumn: int = Field(default=0, ge=0, le=5_242_880)

    @model_validator(mode="after")
    def exactly_one_file(self):
        if bool(self.attachmentId) == bool(self.relativePath):
            raise ValueError("提供一个 attachmentId 或唯一 relativePath。")
        return self


class AttachmentSearchArgs(Arguments):
    query: str = Field(min_length=1, max_length=200)
    attachmentId: str | None = Field(default=None, min_length=1, max_length=128)
    relativePath: str | None = Field(default=None, min_length=1, max_length=512)
    fileOffset: int = Field(default=0, ge=0, le=200)
    startLine: int = Field(default=1, ge=1, le=10_000_000)
    fileLimit: int = Field(default=10, ge=1, le=20)
    limit: int = Field(default=20, ge=1, le=30)
    caseSensitive: bool = False

    @model_validator(mode="after")
    def unambiguous_file(self):
        if self.attachmentId and self.relativePath:
            raise ValueError("attachmentId 与 relativePath 只能提供一个。")
        return self


ATTACHMENT_ARGUMENT_MODELS = {
    "attachment_list": AttachmentListArgs, "attachment_read": AttachmentReadArgs,
    "attachment_search": AttachmentSearchArgs,
}


ARGUMENT_MODELS = {
    "weather": WeatherArgs, "web_search": SearchArgs,
    "memory_snapshot": MemoryArgs, "work_summary": WorkSummaryArgs,
    "file_parse": FileArgs, "image_analyze": ImageArgs, "image_edit": ImageArgs,
    "image_generate": ImageGenerateArgs, "todo": TodoArgs, "calendar": CalendarArgs,
    "health_check": EmptyArgs, "memory_stats": EmptyArgs, "list_tools": EmptyArgs,
}


def slot_records(arguments: Arguments, user_text: str) -> dict:
    """Conservative provenance, never treat a model's own confidence as authorization.

    Normalized dates or rewritten queries are inferred, even when derived from explicit
    text. Exact substring evidence is only an attribution heuristic, not permission.
    """
    records = {}
    for name, value in arguments.model_dump(mode="json").items():
        defaulted = name not in arguments.model_fields_set
        explicit = isinstance(value, str) and bool(value) and value in user_text
        records[name] = {
            "value": value,
            "source": "default" if defaulted else "user_explicit" if explicit else "inferred",
            "confidence": 1.0 if explicit else 0.0 if value is None else 0.5,
        }
    return records
