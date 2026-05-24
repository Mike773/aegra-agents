from __future__ import annotations

from langchain_gigachat import GigaChat
from langgraph.graph import END, START, StateGraph

from ..shared.clients import create_gigachat_client
from .nodes import make_analyze_node, make_parse_input_node
from .state import JsonAnalyzerState


def build_graph(llm: GigaChat):
    g = StateGraph(JsonAnalyzerState)
    g.add_node("parse_input", make_parse_input_node())
    g.add_node("analyze", make_analyze_node(llm))
    g.add_edge(START, "parse_input")
    g.add_edge("parse_input", "analyze")
    g.add_edge("analyze", END)
    return g.compile()


llm = create_gigachat_client().get_llm()
graph = build_graph(llm)
