"""Параметры графа external_doc_loader.

Без pydantic — статичные дефолты (как у gap_resolver.config/doc_manager.config).
Любой из них можно переопределить на вызов через ``config.configurable``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Порог сходства текста (SequenceMatcher.ratio), выше которого документ
    # считаем «похожим» на уже загруженный (та же тема, изменённая версия).
    # Точный дубль ловится раньше по sha256 и до сравнения текста не доходит.
    similar_threshold: float = 0.9

    # Что делать с «похожим»: False — пропустить и показать в отчёте (дефолт,
    # ничего не удаляем), True — удалить старый документ (каскад на чанки и
    # кандидатов) и записать новый как pending.
    replace_similar: bool = False

    # Потолок числа документов из источника за один прогон.
    max_docs: int = 100


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


__all__ = ["Settings", "get_settings"]
