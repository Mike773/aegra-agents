"""Сборка графа kb_chat.

route → (kb: retrieve → respond | chat: respond) → END. Один вызов графа = один
ход диалога; история между ходами живёт в per-thread чекпойнтере (в проде — aegra).
"""
from __future__ import annotations

from langchain_gigachat import GigaChat
from langgraph.graph import END, START, StateGraph

from ..easyrag.graph import graph as easyrag_graph
from ..shared.clients import create_gigachat_client
from .nodes import (
    after_route,
    make_respond_node,
    make_retrieve_node,
    make_route_node,
)
from .state import KbChatState


def build_graph(llm: GigaChat):
    g = StateGraph(KbChatState)

    g.add_node("route", make_route_node(llm))
    g.add_node("retrieve", make_retrieve_node(easyrag_graph))
    g.add_node("respond", make_respond_node(llm))

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        after_route,
        {"retrieve": "retrieve", "respond": "respond"},
    )
    g.add_edge("retrieve", "respond")
    g.add_edge("respond", END)

    return g.compile()


llm = create_gigachat_client().get_llm()
graph = build_graph(llm)

__all__ = ["build_graph", "graph"]
