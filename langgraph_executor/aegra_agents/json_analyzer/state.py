from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class JsonAnalyzerState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # Вход: JSON-датасет (распарсенный dict или сырая строка) и вопрос.
    raw_json: str | dict | None
    question: str
    # Изоляция pgvector-кэша эмбеддингов между направлениями.
    direction_key: str
    # Промежуточные результаты узла gather.
    parsed_rows: list[dict[str, Any]]
    gathered_facts: str
    # Структурированные шаги tool-loop (Блок A.4 ТЗ): список
    # {"tool", "args", "result_summary"} — оркестратор маппит их в TraceStep.
    tool_steps: list[dict[str, Any]]
    completed: bool
    # Итог.
    answer: str
