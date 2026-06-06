"""Сборка графа gap_resolver.

load_gaps → (investigate | finalize) → END. Узел load_gaps выбирает нерешённые
gap'ы направления; если их нет — сразу finalize (короткий отчёт). investigate
ищет ответы в исходных документах, заводит заглушки и помечает решённое.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import after_load, finalize, investigate, load_gaps
from .state import GapResolverState


def build_graph():
    g = StateGraph(GapResolverState)

    g.add_node("load_gaps", load_gaps)
    g.add_node("investigate", investigate)
    g.add_node("finalize", finalize)

    g.add_edge(START, "load_gaps")
    g.add_conditional_edges(
        "load_gaps",
        after_load,
        {"investigate": "investigate", "finalize": "finalize"},
    )
    g.add_edge("investigate", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


graph = build_graph()

__all__ = ["build_graph", "graph"]
