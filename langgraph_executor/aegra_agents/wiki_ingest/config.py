"""Пороги и параметры ingest-пайплайна.

Лёгкий аналог easyRag ``Settings`` (та же семантика и дефолты), без pydantic —
значения статичны и не зависят от окружения. Если позже понадобится override
через env, можно заменить на pydantic-settings.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Резолвер: эмбеддинги кандидатов считаются по чистому ``name`` (см.
    # pipeline._embed_text), поэтому пороги ниже, чем для ``name. descriptor``.
    # sim ≥ high → уверенный merge; < low → уверенно новая; между — LLM-судья.
    resolve_thresh_high: float = 0.85
    resolve_thresh_low: float = 0.45

    # Сколько символов от начала документа подавать в analyze_document.
    doc_brief_window: int = 4000

    # Чанкер (chars). См. chunker.chunk_text.
    chunk_target_size: int = 1200
    chunk_max_size: int = 1800
    chunk_overlap: int = 150

    # Сколько существующих страниц передавать в merge/relink-промпт как каталог.
    merge_catalog_limit: int = 300

    # Back-link: после resolve пройтись по существующим страницам и проставить
    # [[…]] на свежие сущности. prefilter пропускает страницы без substring-
    # совпадения с title/alias свежих сущностей (экономит LLM-вызовы).
    backlink_enabled: bool = True
    backlink_prefilter: bool = True

    # Параметры LLM (как в easyRag). max_tokens — потолок вывода: при дефолтном
    # (небольшом) лимите GigaChat усекает ответ save_entities/merge → меньше
    # сущностей. temperature=0 — детерминированное извлечение. None = не трогать
    # значение на инстансе клиента.
    llm_max_tokens: int | None = 50000
    llm_temperature: float | None = 0.0


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


__all__ = ["Settings", "get_settings"]
