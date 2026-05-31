"""Параметры графа doc_manager.

Без pydantic — статичные дефолты (как у wiki_ingest.config). Если позже
понадобится override через env — заменить на pydantic-settings.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # MIME по умолчанию для документов, пришедших текстом в сообщении.
    default_mime: str = "text/markdown"

    # Мягкая дедупликация загрузки по (direction_key, sha256). У source_doc нет
    # unique-констрейнта, поэтому проверка чисто advisory: не даём повторно
    # залить идентичный текст (и повторно прогнать его через wiki_ingest).
    upload_dedup: bool = True

    # Длина/число переносов, после которых сообщение считаем целым документом
    # (upload) и НЕ гоняем его через LLM-классификатор.
    upload_heuristic_max_len: int = 600
    upload_heuristic_max_newlines: int = 3

    # Заголовок (uri) документа — первая непустая строка, обрезанная до лимита.
    title_max_len: int = 120

    # Резолвер ссылки на удаление: порог fuzzy-совпадения по uri и минимальный
    # отрыв топ-1 от топ-2, ниже которого считаем результат неоднозначным.
    delete_fuzzy_threshold: float = 0.6
    delete_fuzzy_ambiguity_gap: float = 0.05


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


__all__ = ["Settings", "get_settings"]
