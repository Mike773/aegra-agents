"""LLM-проверка: содержит ли фрагмент исходного документа ОТВЕТ на вопрос.

Отбора по близости больше нет — у чанков нет эмбеддинга (миграция 0004), и узел
зовёт судью на КАЖДЫЙ чанк направления отдельно. Судья и раньше был тем, кто
решает «найдено/не найдено»: тематическая близость ответа не гарантирует («на
каком авто ездил Колобок» близко к тексту про Колобка, но ответа там нет), а
теперь он же и единственный фильтр.

Функция принимает список фрагментов, но узел передаёт ровно один: так вердикт
привязан к конкретному чанку, и в отчёт попадает точный источник ответа.
Вердикт ``None`` — судью не удалось вызвать; вопрос остаётся открытым.
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
