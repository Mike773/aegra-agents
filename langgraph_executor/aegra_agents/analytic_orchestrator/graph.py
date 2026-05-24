from __future__ import annotations

from langchain_gigachat import GigaChat
from langgraph.graph import END, START, StateGraph

from ..json_analyzer.graph import graph as json_analyzer_graph
from ..knowledge_agent.graph import graph as knowledge_agent_graph
from ..shared.clients import create_gigachat_client
from .nodes import (
    after_finalize,
    make_ask_user_node,
    make_finalize_node,
    make_json_node,
    make_knowledge_node,
    make_route_node,
    route_intent,
)
from .state import OrchestratorState


def build_graph(llm: GigaChat):
    g = StateGraph(OrchestratorState)

    g.add_node("ask_user", make_ask_user_node())
    g.add_node("route", make_route_node(llm))
    g.add_node("knowledge", make_knowledge_node(knowledge_agent_graph))
    g.add_node("json", make_json_node(json_analyzer_graph))
    g.add_node("finalize", make_finalize_node(llm))

    g.add_edge(START, "ask_user")
    g.add_edge("ask_user", "route")

    g.add_conditional_edges(
        "route",
        route_intent,
        {"knowledge": "knowledge", "json": "json", "finalize": "finalize"},
    )

    g.add_edge("knowledge", "finalize")
    g.add_edge("json", "finalize")

    g.add_conditional_edges(
        "finalize",
        after_finalize,
        {"ask_user": "ask_user", "__end__": END},
    )

    return g.compile()


llm = create_gigachat_client().get_llm()
graph = build_graph(llm)
