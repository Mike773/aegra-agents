from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class KbChatState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

    # Параметры (из configurable, затем из state).
    direction_key: str
    top_k: int

    # Классификация текущей реплики: искать в базе знаний или просто поболтать.
    intent: Literal["kb", "chat"]

    # Результат retrieve по последнему вопросу (перезаписывается каждый ход).
    snippets: list[dict]
    snippet_error: str | None
