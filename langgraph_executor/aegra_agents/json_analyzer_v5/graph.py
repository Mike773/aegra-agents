from __future__ import annotations

from langchain_gigachat import GigaChat
from langgraph.graph import END, START, StateGraph

from ..shared.clients import create_gigachat_client
from .nodes import make_gather_node, make_synthesize_node
from .state import JsonAnalyzerState


def build_graph(llm: GigaChat):
    g = StateGraph(JsonAnalyzerState)
    g.add_node("gather", make_gather_node(llm))
    g.add_node("synthesize", make_synthesize_node(llm))
    g.add_edge(START, "gather")
    g.add_edge("gather", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


llm = create_gigachat_client().get_llm()
graph = build_graph(llm)
