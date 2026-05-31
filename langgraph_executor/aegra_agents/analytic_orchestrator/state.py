from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Intent = Literal[
    "analytics",
    "wiki",
    "chat",
    "done",
    "assignments",
    "assignments_select",
]


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
    # Релевантные запросу страницы-заглушки (type='stub', пустые) — заполняется,
    # только когда обычная выборка пуста: сущность заведена, но не наполнена.
    easyrag_stub_pages: list[dict]

    # Результат вызова json_analyzer-подграфа под последний аналитический вопрос.
    # Sticky: не сбрасывается между циклами, пока следующий analytics-цикл не
    # перезапишет — респондер использует самый свежий ответ.
    analytics_question: str | None
    analytics_answer: str | None
    analytics_error: str | None

    # Поручения по проблемным зонам сотрудника.
    # candidate_assignments — последняя извлечённая «корзина» кандидатов.
    # pending_assignments — кандидаты, по которым прямо сейчас ждём выбор пользователя.
    # Пока pending не пуст, роутер форсит intent="assignments_select".
    # selected_assignments — выбранное к фиксации в текущей итерации.
    # last_committed_assignments — что реально ушло в mock-сервис в прошлый раз.
    candidate_assignments: list[dict]
    pending_assignments: list[dict]
    selected_assignments: list[dict]
    last_committed_assignments: list[dict]
