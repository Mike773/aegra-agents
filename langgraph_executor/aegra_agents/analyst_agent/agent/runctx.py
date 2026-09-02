"""Контекст одного запуска агента.

Живёт в замыкании узла и его инструментов: база, журнал отклонений, флаги и
накопленные за ход обращения к wiki. Модульных глобалов нет — граф вызывают
конкурентно.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunContext:
    db: Any
    person_key: str
    direction_key: str
    ledger: Any
    # Подграф базы знаний и его настройки (None → инструмент не привязывается).
    easyrag_graph: Any = None
    easyrag_enabled: bool = True
    easyrag_top_k: int = 5
    gap_on_unanswered: bool = True
    use_peer_aggregates: bool = True
    text2sql_enabled: bool = True
    # Что нашлось в wiki за этот ход — уезжает в поля easyrag_* стейта, чтобы
    # раздел «Исходные данные» и describe_answer работали как в v4.
    wiki_queries: list[str] = field(default_factory=list)
    wiki_snippets: list[dict] = field(default_factory=list)
    wiki_stub_pages: list[dict] = field(default_factory=list)
    wiki_error: str | None = None
    used_data_tools: bool = False
    used_wiki_tool: bool = False


__all__ = ["RunContext"]
