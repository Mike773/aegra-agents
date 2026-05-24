from __future__ import annotations

from langchain_gigachat import GigaChat
from langgraph.graph import END, START, StateGraph

from ..shared.clients import create_gigachat_client
from .nodes import make_generate_node, make_retrieve_node
from .state import KnowledgeState


def build_graph(llm: GigaChat):
    g = StateGraph(KnowledgeState)
    g.add_node("retrieve", make_retrieve_node())
    g.add_node("generate", make_generate_node(llm))
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


llm = create_gigachat_client().get_llm()
graph = build_graph(llm)
