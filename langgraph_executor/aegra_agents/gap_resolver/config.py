"""Параметры графа gap_resolver.

Без pydantic — статичные дефолты (как у wiki_ingest.config/doc_manager.config).
Любой из них можно переопределить на вызов через ``config.configurable``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Сколько чанков направления проверять на один вопрос. Это НЕ отбор по
    # релевантности (векторов у чанков нет), а предохранитель: судья вызывается
    # на каждый чанк, поэтому число прямо задаёт стоимость разбора пробела.
    # Ориентир масштаба: примерно один чанк на 1000 знаков документа.
    max_chunks_per_gap: int = 400

    # Сколько вызовов судьи идёт параллельно. Больше брать нельзя: GigaChat
    # троттлит уже на нескольких одновременных запросах.
    judge_concurrency: int = 4

    # Общий потолок вызовов судьи на прогон. Вопрос без ответа стоит столько
    # вызовов, сколько чанков в направлении, а таких вопросов обычно
    # большинство — без потолка фоновый прогон может идти часами.
    max_judge_calls: int = 2000

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
