"""Инструмент ``suggest_followups`` — варианты следующего вопроса руководителя.

Ничего не считает: принимает список от модели, чистит его и кладёт в контекст
запуска, откуда узел агента переносит его в ``additional_kwargs`` итога
(``orchestrator_suggestions``). Привязывается только при
``interactive_suggestions=true``.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..agent.runctx import RunContext
from ..prompts.suggestions import (
    MAX_LABEL_LEN,
    MAX_SUGGESTIONS,
    SUGGESTIONS_TOOL_DESCRIPTION,
)

TOOL_NAME = "suggest_followups"
# Тип подсказки в контракте клиента: подсказка — готовое сообщение в чат.
SUGGESTION_TYPE = "message"

_EMPTY_HINT = (
    "Варианты не приняты: нужен список объектов с полями label (подпись кнопки) "
    "и content (полный вопрос руководителя). Заполни оба поля и вызови ещё раз "
    "либо переходи к итоговому ответу."
)


class Suggestion(BaseModel):
    """Один вариант: подпись кнопки и вопрос, который она отправит."""

    label: str = Field(..., description="Подпись кнопки — суть варианта одной фразой")
    content: str = Field(..., description="Полный вопрос от лица руководителя")


class SuggestArgs(BaseModel):
    suggestions: list[Suggestion] = Field(
        ..., description=f"Варианты следующего вопроса, до {MAX_SUGGESTIONS} штук"
    )


def clean_suggestions(raw: Any) -> list[dict[str, str]]:
    """Список модели → список подсказок контракта: без пустых, с обрезкой."""
    out: list[dict[str, str]] = []
    for item in raw or []:
        if isinstance(item, BaseModel):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        content = str(item.get("content") or "").strip()
        if not label or not content:
            continue
        if len(label) > MAX_LABEL_LEN:
            label = label[: MAX_LABEL_LEN - 1].rstrip() + "…"
        out.append({"type": SUGGESTION_TYPE, "label": label, "content": content})
        if len(out) >= MAX_SUGGESTIONS:
            break
    return out


def make_suggest_followups(ctx: RunContext) -> StructuredTool:
    def suggest_followups(suggestions: list[Any]) -> str:
        cleaned = clean_suggestions(suggestions)
        if not cleaned:
            return _EMPTY_HINT
        # Повторный вызов перезаписывает список: у ответа один набор кнопок.
        ctx.suggestions = cleaned
        return (
            f"Варианты приняты: {len(cleaned)}. Теперь напиши итоговый ответ "
            "без раздела «Варианты для углубления»."
        )

    return StructuredTool.from_function(
        func=suggest_followups,
        args_schema=SuggestArgs,
        name=TOOL_NAME,
        description=SUGGESTIONS_TOOL_DESCRIPTION,
    )


__all__ = ["SUGGESTION_TYPE", "TOOL_NAME", "clean_suggestions", "make_suggest_followups"]
