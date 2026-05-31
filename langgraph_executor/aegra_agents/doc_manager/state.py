"""Состояние графа doc_manager.

Message-driven (как json_analyzer/analytic_orchestrator): несёт ``messages`` с
``add_messages`` и отвечает ``AIMessage``. ``last_listed`` хранит последнюю выдачу
списка между тёрнами (через чекпойнтер) — чтобы удалять по порядковому номеру.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class DocManagerState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

    # Вход. direction_key читается из config.configurable, затем из state.
    direction_key: str

    # Результат классификации текущего тёрна.
    intent: Literal["upload", "list", "delete", "unknown"]
    upload_title: str
    upload_content: str
    delete_reference: str

    # Кэш последней выдачи списка (перезаписывается узлом list, переживает тёрн
    # через чекпойнтер). id — строками, даты — ISO-строками: иначе чекпойнтер не
    # сериализует UUID/datetime.
    # Элемент: {"id", "uri", "mime", "processed_at", "ingested_at", "chunks"}
    last_listed: list[dict[str, Any]]

    report: str
