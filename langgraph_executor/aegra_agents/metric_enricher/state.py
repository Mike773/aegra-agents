"""Состояние фонового обогатителя трактовок показателей."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class MetricEnricherState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

    # Вход (можно и через configurable).
    direction_key: str
    max_metrics: int
    top_k: int
    concurrency: int

    # Что взяли в работу и что получилось.
    pending: list[dict[str, Any]]
    found: int
    unknown: int
    errors: list[str]
    report: str


__all__ = ["MetricEnricherState"]
