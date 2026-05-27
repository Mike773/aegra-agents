from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..shared.clients import create_gigachat_embeddings
from .nodes import (
    after_retrieve,
    make_embed_query_node,
    make_maybe_record_gap_node,
    make_retrieve_node,
)
from .state import EasyRagState


def build_graph(embedder=None):
    g = StateGraph(EasyRagState)
    embedder = embedder if embedder is not None else create_gigachat_embeddings()

    g.add_node("embed_query", make_embed_query_node(embedder))
    g.add_node("retrieve", make_retrieve_node())
    g.add_node("maybe_record_gap", make_maybe_record_gap_node())

    g.add_edge(START, "embed_query")
    g.add_edge("embed_query", "retrieve")
    g.add_conditional_edges(
        "retrieve",
        after_retrieve,
        {"maybe_record_gap": "maybe_record_gap", "__end__": END},
    )
    g.add_edge("maybe_record_gap", END)

    return g.compile()


graph = build_graph()
