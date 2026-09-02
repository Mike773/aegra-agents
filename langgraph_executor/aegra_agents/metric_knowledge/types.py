"""Типы кэша трактовок показателей."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Версия схемы/промпта трактовки. Бампить, когда меняется набор правил или
# формулировка вопроса к модели: строки старой версии пойдут на перерасчёт.
KNOWLEDGE_VERSION = 1

# Виды показателей: относительные проценты осмысленны только у «уровня».
VALID_KINDS = ("уровень", "вклад", "индекс")
DEFAULT_KIND = "уровень"

STATUS_FOUND = "found"
STATUS_UNKNOWN = "unknown"


@dataclass(frozen=True)
class CatalogEntry:
    """Показатель датасета — то, о чём спрашиваем базу знаний."""

    metric_key: str
    name_key: str
    metric_name: str
    metric_description: str = ""
    metric_ext_id: str | None = None
    direction: str | None = None
    unit: str | None = None
    parent_name: str | None = None
    depth: int | None = None


@dataclass
class KnowledgeRow:
    """Трактовка показателя из базы знаний (строка кэша)."""

    metric_key: str
    name_key: str
    metric_name: str
    metric_description: str = ""
    aliases: list[str] = field(default_factory=list)
    summary: str = ""
    rules: dict[str, Any] = field(default_factory=dict)
    status: str = STATUS_FOUND
    confidence: float = 0.0
    source_section_ids: list[str] = field(default_factory=list)
    model: str | None = None
    version: int = KNOWLEDGE_VERSION
    updated_at: datetime | None = None


__all__ = [
    "DEFAULT_KIND",
    "KNOWLEDGE_VERSION",
    "STATUS_FOUND",
    "STATUS_UNKNOWN",
    "VALID_KINDS",
    "CatalogEntry",
    "KnowledgeRow",
]
