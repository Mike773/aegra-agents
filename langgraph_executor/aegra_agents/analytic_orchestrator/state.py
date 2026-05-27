from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Intent = Literal["analytics", "wiki", "chat", "done"]


class OrchestratorState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

    boss_tabnum: str
    employee_tabnum: str
    position: str | None
    direction_key: str | None

    metrics: Any
    metrics_error: str | None
    loaded: bool

    intent: Intent | None

    # Результат вызова easyrag-подграфа (свежий, под последний вопрос пользователя).
    easyrag_query: str | None
    easyrag_snippets: list[dict]
    easyrag_error: str | None

    # Результат вызова json_analyzer-подграфа под последний аналитический вопрос.
    # Sticky: не сбрасывается между циклами, пока следующий analytics-цикл не
    # перезапишет — респондер использует самый свежий ответ.
    analytics_question: str | None
    analytics_answer: str | None
    analytics_error: str | None
