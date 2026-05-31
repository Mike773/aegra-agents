from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import after_load, finalize, load_pending, process
from .state import WikiIngestState


def build_graph():
    g = StateGraph(WikiIngestState)

    g.add_node("load_pending", load_pending)
    g.add_node("process", process)
    g.add_node("finalize", finalize)

    g.add_edge(START, "load_pending")
    g.add_conditional_edges(
        "load_pending",
        after_load,
        {"process": "process", "finalize": "finalize"},
    )
    g.add_edge("process", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


graph = build_graph()
