"""Сборка графа metric_enricher.

load_catalog → (enrich | finalize) → END. Фоновая работа: наполняет кэш
трактовок показателей, чтобы агент-аналитик не тратил на это время хода.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import after_load, enrich, finalize, load_catalog
from .state import MetricEnricherState


def build_graph():
    g = StateGraph(MetricEnricherState)
    g.add_node("load_catalog", load_catalog)
    g.add_node("enrich", enrich)
    g.add_node("finalize", finalize)
    g.add_edge(START, "load_catalog")
    g.add_conditional_edges(
        "load_catalog", after_load, {"enrich": "enrich", "finalize": "finalize"}
    )
    g.add_edge("enrich", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


graph = build_graph()

__all__ = ["build_graph", "graph"]
