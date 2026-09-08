"""Сборка графа analyst_agent — только топология.

Один граф вместо связки «оркестратор + подграф-аналитик»: роутера-модели,
отдельного сбора фактов и стадии пересказа больше нет. За ход модель вызывается
один раз циклом инструментов, и её финальный текст и есть ответ руководителю.

Ход turn-based, без interrupt(): один вызов графа = одно входящее сообщение →
ответ → END. Состояние между ходами держит чекпойнтер aegra по thread_id.
"""
from __future__ import annotations

from langchain_gigachat import GigaChat
from langgraph.graph import END, START, StateGraph

from ..easyrag.graph import graph as easyrag_graph
from ..shared.clients import create_gigachat_client
from .nodes.agent_node import make_agent_node
from .nodes.insight import make_auto_insight_node
from .nodes.load import make_load_data_node
from .nodes.plan import make_plan_tasks_node, make_summarize_node
from .nodes.prepare import make_begin_turn_node, make_prepare_node
from .nodes.routing import after_agent, after_prepare, need_load
from .state import AnalystOutput, AnalystState


def build_graph(llm: GigaChat, checkpointer=None):
    # output_schema без metrics/aggregates: сырой датасет остаётся каналом
    # стейта (чекпойнтится, живёт между ходами), но наружу не отдаётся.
    g = StateGraph(AnalystState, output_schema=AnalystOutput)

    g.add_node("load_data", make_load_data_node())
    g.add_node("prepare", make_prepare_node())
    g.add_node("begin_turn", make_begin_turn_node())
    g.add_node("plan_tasks", make_plan_tasks_node(llm))
    g.add_node("agent", make_agent_node(llm, easyrag_graph))
    g.add_node("summarize", make_summarize_node(llm))
    g.add_node("auto_insight", make_auto_insight_node())

    # Первый ход треда грузит данные и готовит базу; дальше состояние берётся
    # из чекпойнтера, и ход начинается сразу с агента.
    g.add_conditional_edges(
        START, need_load, {"load_data": "load_data", "begin_turn": "begin_turn"}
    )
    # Долгосрочная память отключена: узлы load_memory/save_memory из
    # ..long_term_memory в граф не подключены, memory_context остаётся пустым.
    g.add_edge("load_data", "prepare")
    g.add_conditional_edges(
        "prepare",
        after_prepare,
        {"plan_tasks": "plan_tasks", "agent": "agent", "auto_insight": "auto_insight"},
    )
    g.add_edge("plan_tasks", "agent")
    g.add_edge("begin_turn", "agent")

    # Составной разбор: узел агента отрабатывает одну задачу за вызов, граф
    # возвращает управление сюда же, пока задачи не кончатся. Ребро в самого
    # себя, а не Send: задачи упорядочены и следующая опирается на предыдущие.
    g.add_conditional_edges(
        "agent",
        after_agent,
        {
            "agent": "agent",
            "summarize": "summarize",
            "auto_insight": "auto_insight",
            END: END,
        },
    )
    g.add_edge("summarize", "auto_insight")
    g.add_edge("auto_insight", END)

    return g.compile(checkpointer=checkpointer)


llm = create_gigachat_client().get_llm()
graph = build_graph(llm)

__all__ = ["build_graph", "graph"]
