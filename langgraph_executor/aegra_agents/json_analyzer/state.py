from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class JsonAnalyzerState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    raw_json: str | dict | list | None
    parsed: Any
    findings: list[str]
    summary: str
