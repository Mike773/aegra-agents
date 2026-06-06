"""LLM-проверка: содержат ли фрагменты исходных документов ОТВЕТ на вопрос.

Векторный отбор (retrieval.search_source_chunks) даёт лишь тематически близкие
чанки — близость ≠ наличие ответа («на каком авто ездил Колобок» близко к тексту
про Колобка, но ответа там нет). Поэтому решение «найдено/не найдено» принимает
LLM по реальному тексту кандидатов. Возвращаем ``None`` как вердикт, если судью
не удалось вызвать — тогда узел падает на порог близости.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ..wiki_ingest.llm import LLMClient

_RETRIES = 3

_ANSWER_TOOL = "save_answer_check"
_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {
            "type": "boolean",
            "description": "true только если фрагменты содержат фактический ответ на вопрос.",
        },
        "quote": {
            "type": "string",
            "description": "Короткая дословная цитата-обоснование из фрагментов (при found=true).",
        },
    },
    "required": ["found"],
    "additionalProperties": False,
}
_SYSTEM = (
    "Тебе дают вопрос и фрагменты исходных документов одного направления. Определи, "
    "содержат ли ЭТИ фрагменты фактический ответ на вопрос. found=true ставь только "
    "если ответ прямо следует из текста фрагментов; одной лишь тематической близости "
    "(тот же объект/тема, но без ответа) НЕДОСТАТОЧНО — тогда found=false. При "
    "found=true приведи в quote короткую дословную цитату из фрагментов. "
    "Ответ только через вызов tool save_answer_check."
)


@dataclass(frozen=True)
class Verdict:
    found: bool | None  # None — судью не удалось вызвать
    quote: str = ""


async def answer_in_sources(
    query: str, chunk_texts: list[str], *, llm: LLMClient | None = None
) -> Verdict:
    if not chunk_texts:
        return Verdict(found=False)
    client = llm or LLMClient()
    fragments = "\n---\n".join(chunk_texts)
    user = f"Вопрос: {query}\n\nФрагменты исходных документов:\n{fragments}"
    # GigaChat иногда не возвращает tool_call — ретраим. Если так и не вышло,
    # found=None: узел трактует это как «не подтверждено» (не помечает решённым),
    # и gap переберётся на следующем прогоне.
    for attempt in range(_RETRIES):
        try:
            raw = await client.call_json(
                system=_SYSTEM,
                user=user,
                tool_name=_ANSWER_TOOL,
                tool_description="Сохранить вердикт: есть ли во фрагментах ответ на вопрос.",
                input_schema=_ANSWER_SCHEMA,
            )
            if isinstance(raw, dict):
                quote = raw.get("quote")
                return Verdict(
                    found=bool(raw.get("found")),
                    quote=quote.strip() if isinstance(quote, str) else "",
                )
        except Exception:
            pass
        if attempt + 1 < _RETRIES:
            await asyncio.sleep(1.5 * (attempt + 1))
    return Verdict(found=None)


__all__ = ["Verdict", "answer_in_sources"]
