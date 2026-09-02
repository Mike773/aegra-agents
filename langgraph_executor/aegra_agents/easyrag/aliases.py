"""Сопоставление запроса с алиасами страниц.

Зачем отдельно от векторов: у страницы «Среднее время обработки» алиас «AHT»,
и модель эмбеддингов с аббревиатурами работает плохо — в векторе секции короткий
алиас растворяется среди сотен слов текста. Точное совпадение слова из запроса с
алиасом ловит аббревиатуру гарантированно и без обращения к модели.

Здесь только разбор строк — обращения к базе живут в ``retrieval``.
"""
from __future__ import annotations

import re

# Кандидатов из запроса берём ограниченно: это параметры SQL-запроса, а не поиск
# по всему тексту.
MAX_CANDIDATES = 64
# Односимвольные токены отбрасываем — предлоги и союзы алиасами не бывают.
# Двухсимвольные оставляем: SL, KPI-подобные сокращения реально короткие.
MIN_TOKEN = 2

_WORD_RE = re.compile(r"[\w-]+", re.UNICODE)


def alias_norm(value: object) -> str:
    """Каноническая форма алиаса: без краевых пробелов, схлопнутые пробелы, регистр."""
    if value is None:
        return ""
    return " ".join(str(value).split()).casefold()


def query_alias_candidates(query: str) -> list[str]:
    """Строки запроса, которые имеет смысл сравнить с алиасами.

    Берём весь запрос целиком (короткий вопрос вида «AHT» — это и есть алиас),
    отдельные слова и пары соседних слов: алиасы часто двусловные («время
    обработки»). Порядок — от точного к частному, дубликаты убираются.
    """
    text = (query or "").strip()
    if not text:
        return []

    out: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        norm = alias_norm(value)
        if norm and len(norm) >= MIN_TOKEN and norm not in seen:
            seen.add(norm)
            out.append(norm)

    add(text)
    words = _WORD_RE.findall(text)
    for word in words:
        add(word)
    for first, second in zip(words, words[1:]):
        add(f"{first} {second}")
    return out[:MAX_CANDIDATES]


__all__ = ["MAX_CANDIDATES", "MIN_TOKEN", "alias_norm", "query_alias_candidates"]
