from __future__ import annotations

from langchain_gigachat import GigaChat
from langgraph.graph import END, START, StateGraph

from ..easyrag.graph import graph as easyrag_graph
from ..json_analyzer.graph import graph as json_analyzer_graph
from ..shared.clients import create_gigachat_client
from .nodes import (
    after_route,
    make_ask_user_node,
    make_call_easyrag_node,
    make_call_json_analyzer_node,
    make_initial_analysis_node,
    make_load_data_node,
    make_respond_node,
    make_route_node,
    need_load,
)
from .state import OrchestratorState


def build_graph(llm: GigaChat):
    g = StateGraph(OrchestratorState)

    g.add_node("load_data", make_load_data_node())
    g.add_node("initial_analysis", make_initial_analysis_node(llm))
    g.add_node("ask_user", make_ask_user_node())
    g.add_node("route", make_route_node(llm))
    g.add_node("call_json_analyzer", make_call_json_analyzer_node(json_analyzer_graph))
    g.add_node("call_easyrag", make_call_easyrag_node(easyrag_graph))
    g.add_node("respond", make_respond_node(llm))

    g.add_conditional_edges(
        START,
        need_load,
        {"load_data": "load_data", "ask_user": "ask_user"},
    )
    g.add_edge("load_data", "initial_analysis")
    g.add_edge("initial_analysis", "ask_user")
    g.add_edge("ask_user", "route")
    g.add_conditional_edges(
        "route",
        after_route,
        {
            "call_json_analyzer": "call_json_analyzer",
            "call_easyrag": "call_easyrag",
            "respond": "respond",
            "__end__": END,
        },
    )
    g.add_edge("call_json_analyzer", "respond")
    g.add_edge("call_easyrag", "respond")
    g.add_edge("respond", "ask_user")

    return g.compile()


llm = create_gigachat_client().get_llm()
graph = build_graph(llm)
