from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class KnowledgeState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    snippets: list[dict]
    answer: str
