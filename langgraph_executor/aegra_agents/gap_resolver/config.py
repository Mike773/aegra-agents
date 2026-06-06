"""Параметры графа gap_resolver.

Без pydantic — статичные дефолты (как у wiki_ingest.config/doc_manager.config).
Любой из них можно переопределить на вызов через ``config.configurable``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Сколько чанков исходных документов тянуть на один gap при поиске.
    top_k: int = 5

    # Префильтр: чанки с близостью ниже порога не отдаём LLM-судье (экономим
    # вызовы и не шумим заведомо нерелевантным). Решение «найден ли ответ» по
    # отобранным кандидатам принимает judge.answer_in_sources, а не сам порог.
    candidate_thresh: float = 0.35

    # Заводить пустую wiki-страницу-заглушку по теме gap'а, если её ещё нет.
    create_stub_pages: bool = True

    # Помечать gap resolved_at, если ответ нашёлся в источниках — чтобы
    # enrichment-loop не разбирал его на каждом прогоне заново.
    mark_resolved: bool = True

    # Потолок числа уникальных вопросов (gap-групп) за один прогон.
    max_gaps: int = 50


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


__all__ = ["Settings", "get_settings"]
