"""Классификация намерения пользователя (upload / list / delete).

Использует ``wiki_ingest.llm.LLMClient.call_json`` (tool-binding поверх GigaChat) —
надёжнее, чем парсить свободный JSON или ``with_structured_output`` (у GigaChat он
даёт пустой ``parameters``). Перед LLM — дешёвая эвристика: длинное многострочное
сообщение почти всегда документ (upload), его незачем гнать в классификатор.
"""
from __future__ import annotations

from ..wiki_ingest.llm import LLMClient, get_llm
from .config import get_settings
from .prompts import INTENT_CLASSIFY_PROMPT

_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["upload", "list", "delete"]},
        "title": {
            "type": "string",
            "description": "Краткий заголовок документа (для upload), иначе пусто",
        },
        "reference": {
            "type": "string",
            "description": (
                "Ссылка на документ для удаления: номер из списка, начало id или "
                "название; иначе пусто"
            ),
        },
    },
    "required": ["intent"],
}


async def classify(text: str, llm: LLMClient | None = None) -> dict:
    """Вернуть {"intent", "title", "reference"} по тексту сообщения."""
    cfg = get_settings()
    head = (text or "").strip()
    if (
        len(head) > cfg.upload_heuristic_max_len
        or head.count("\n") >= cfg.upload_heuristic_max_newlines
    ):
        return {"intent": "upload", "title": "", "reference": ""}

    client = llm or get_llm()
    result = await client.call_json(
        system=INTENT_CLASSIFY_PROMPT,
        user=head,
        tool_name="classify_doc_command",
        tool_description="Классифицировать команду пользователя по управлению документами.",
        input_schema=_CLASSIFY_SCHEMA,
    )
    intent = str(result.get("intent") or "").strip().lower()
    if intent not in {"upload", "list", "delete"}:
        intent = "upload"
    return {
        "intent": intent,
        "title": str(result.get("title") or "").strip(),
        "reference": str(result.get("reference") or "").strip(),
    }


__all__ = ["classify"]
