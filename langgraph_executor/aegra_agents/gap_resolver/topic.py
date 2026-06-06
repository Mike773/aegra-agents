"""Короткая тема gap-вопроса — заголовок для wiki-заглушки.

Один батч-вызов LLM на все вопросы прогона; при сбое или рассинхроне длины —
эвристический фолбэк (обрезка вопроса). Тема нужна лишь как заголовок заглушки и
строка отчёта, поэтому её неточность некритична.
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..wiki_ingest.llm import LLMClient

_RETRIES = 3

_TOPIC_TOOL = "save_topics"
_TOPIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "description": "Короткие темы СТРОГО в том же порядке и количестве, что и вопросы.",
            "items": {"type": "string"},
        }
    },
    "required": ["topics"],
    "additionalProperties": False,
}
_SYSTEM = (
    "Тебе дают пронумерованный список вопросов пользователей. Для КАЖДОГО верни "
    "короткую тему — сущность или понятие, о котором вопрос: 1–5 слов, в "
    "именительном падеже, без знаков вопроса и кавычек. Порядок и количество тем "
    "— строго как у вопросов. Ответ только через вызов tool save_topics."
)


def _heuristic(query: str) -> str:
    q = (query or "").strip().rstrip("?.!").strip()
    words = q.split()
    return " ".join(words[:6])[:80] or q[:80] or "Без темы"


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


async def extract_topics(
    queries: list[str], *, llm: LLMClient | None = None
) -> list[str]:
    """Темы для списка вопросов (выровнены по индексу). Фолбэк — эвристика."""
    fallback = [_heuristic(q) for q in queries]
    if not queries:
        return []
    client = llm or LLMClient()
    user = "Вопросы:\n" + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))
    for attempt in range(_RETRIES):
        try:
            raw = await client.call_json(
                system=_SYSTEM,
                user=user,
                tool_name=_TOPIC_TOOL,
                tool_description="Сохранить короткие темы вопросов (по одной на вопрос, тот же порядок).",
                input_schema=_TOPIC_SCHEMA,
            )
            topics = raw.get("topics") if isinstance(raw, dict) else None
            if isinstance(topics, list) and len(topics) == len(queries):
                return [(_clean(t) or fallback[i]) for i, t in enumerate(topics)]
        except Exception:
            pass
        if attempt + 1 < _RETRIES:
            await asyncio.sleep(1.5 * (attempt + 1))
    return fallback


__all__ = ["extract_topics"]
