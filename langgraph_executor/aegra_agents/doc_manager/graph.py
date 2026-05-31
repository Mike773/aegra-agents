"""Сборка графа doc_manager.

classify → routing на upload/list/delete. Намерение ``unknown`` означает, что
``classify_node`` уже сам ответил (нет direction_key / пустое сообщение) — тогда
сразу END, второй ответ не нужен.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    after_classify,
    classify_node,
    do_delete,
    do_list,
    do_upload,
)
from .state import DocManagerState


def build_graph():
    g = StateGraph(DocManagerState)

    g.add_node("classify", classify_node)
    g.add_node("upload", do_upload)
    g.add_node("list", do_list)
    g.add_node("delete", do_delete)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        after_classify,
        {
            "upload": "upload",
            "list": "list",
            "delete": "delete",
            "unknown": END,
        },
    )
    g.add_edge("upload", END)
    g.add_edge("list", END)
    g.add_edge("delete", END)

    return g.compile()


graph = build_graph()

__all__ = ["build_graph", "graph"]
