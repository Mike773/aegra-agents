"""Сборка графа external_doc_loader.

fetch_documents → (load_documents | finalize) → END. Если внешний источник
пуст или недоступен — сразу finalize (короткий отчёт).
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import after_fetch, fetch_documents, finalize, load_documents
from .state import ExternalDocLoaderState


def build_graph():
    g = StateGraph(ExternalDocLoaderState)

    g.add_node("fetch_documents", fetch_documents)
    g.add_node("load_documents", load_documents)
    g.add_node("finalize", finalize)

    g.add_edge(START, "fetch_documents")
    g.add_conditional_edges(
        "fetch_documents",
        after_fetch,
        {"load_documents": "load_documents", "finalize": "finalize"},
    )
    g.add_edge("load_documents", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


graph = build_graph()

__all__ = ["build_graph", "graph"]
