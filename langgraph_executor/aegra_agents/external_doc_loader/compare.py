"""Сравнение нового документа с уже загруженными документами направления.

Чистая логика без БД — на вход текст и список существующих документов
(уже отфильтрованных по direction_key), на выход вердикт:

* ``duplicate`` — точное совпадение по sha256, грузить нечего;
* ``similar``  — сходство текста с каким-то документом выше порога
  (та же тема, изменённая версия);
* ``new``      — совпадений нет.
"""
from __future__ import annotations

import hashlib
from typing import NamedTuple

from ..shared.text_similarity import similarity_ratio
from .repository import ExistingDoc


class MatchVerdict(NamedTuple):
    kind: str  # duplicate | similar | new
    match: ExistingDoc | None
    similarity: float | None


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ratio_upper_bound(len_a: int, len_b: int) -> float:
    """Верхняя граница SequenceMatcher.ratio() по одним только длинам —
    дешёвый префильтр, чтобы не гонять полное сравнение над заведомо
    непохожими текстами."""
    total = len_a + len_b
    if total == 0:
        return 1.0
    return 2.0 * min(len_a, len_b) / total


def match_against_existing(
    text: str,
    sha256: str,
    existing: list[ExistingDoc],
    *,
    similar_threshold: float,
) -> MatchVerdict:
    """Вердикт по новому документу относительно документов его направления."""
    for doc in existing:
        if doc.sha256 == sha256:
            return MatchVerdict("duplicate", doc, 1.0)

    best: ExistingDoc | None = None
    best_ratio = 0.0
    for doc in existing:
        if _ratio_upper_bound(len(text), len(doc.content)) < similar_threshold:
            continue
        ratio = similarity_ratio(text, doc.content)
        if ratio > best_ratio:
            best, best_ratio = doc, ratio

    if best is not None and best_ratio >= similar_threshold:
        return MatchVerdict("similar", best, best_ratio)
    return MatchVerdict("new", None, best_ratio if best is not None else None)


__all__ = ["MatchVerdict", "content_sha256", "match_against_existing"]
