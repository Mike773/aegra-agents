from __future__ import annotations

from langchain_gigachat import GigaChat
from langgraph.graph import END, START, StateGraph

from ..easyrag.graph import graph as easyrag_graph
from ..json_analyzer.graph import graph as json_analyzer_graph
from ..shared.clients import create_gigachat_client
from .nodes import (
    after_route,
    make_call_easyrag_node,
    make_call_json_analyzer_node,
    make_commit_assignments_node,
    make_extract_assignments_node,
    make_ground_wiki_node,
    make_initial_analysis_node,
    make_load_data_node,
    make_propose_assignments_node,
    make_respond_node,
    make_route_node,
    make_select_assignments_node,
    need_load,
)
from .state import OrchestratorState


def build_graph(llm: GigaChat):
    g = StateGraph(OrchestratorState)

    g.add_node("load_data", make_load_data_node())
    g.add_node(
        "initial_analysis",
        make_initial_analysis_node(llm, json_analyzer_graph),
    )
    g.add_node(
        "extract_assignments",
        make_extract_assignments_node(llm, json_analyzer_graph),
    )
    g.add_node("propose_assignments", make_propose_assignments_node())
    g.add_node("route", make_route_node(llm))
    # Один и тот же узел wiki-grounding на двух позициях графа (у узла фиксированные
    # out-edges, поэтому два инстанса проще условных рёбер): перед первичным
    # анализом и после json_analyzer на analytics-ходу.
    g.add_node("ground_wiki_initial", make_ground_wiki_node(llm, easyrag_graph))
    g.add_node("ground_wiki_analytics", make_ground_wiki_node(llm, easyrag_graph))
    g.add_node("call_json_analyzer", make_call_json_analyzer_node(json_analyzer_graph))
    g.add_node("call_easyrag", make_call_easyrag_node(easyrag_graph))
    g.add_node("select_assignments", make_select_assignments_node(llm))
    g.add_node("commit_assignments", make_commit_assignments_node())
    g.add_node("respond", make_respond_node(llm))

    # Turn-based чат: один вызов графа = одно входящее сообщение → ответ → END.
    # Состояние между ходами держит per-thread чекпоинтер aegra (по thread_id),
    # поэтому никаких interrupt() — следующая реплика приходит обычным входом
    # {"messages": [...]}, а не через Command(resume=...).
    g.add_conditional_edges(
        START,
        need_load,
        {"load_data": "load_data", "route": "route"},
    )
    # Первый ход: входящее сообщение — инструкция-триггер (брифинг). Грузим
    # данные, делаем первичный анализ и завершаем ход. Контракт сообщений хода:
    # рабочие узлы по пути кладут короткие «шаговые» сообщения (помечены
    # additional_kwargs.orchestrator_step), а терминальный лист — ИТОГОВЫЙ ответ
    # (additional_kwargs.orchestrator_final). Итог всегда последний элемент
    # messages: его и показывает пользователю вызывающая система (messages[-1]
    # либо по флагу orchestrator_final). Прогресс можно отключить флагом
    # configurable.emit_progress_messages=false (тогда за ход одно сообщение).
    # Поручения на первом ходу НЕ формируем: если анализ нашёл проблему, он
    # лишь предлагает их оформить (текстом, в том же сообщении). Реальный разбор
    # кандидатов запускает роутер по intent 'assignments' на последующем ходу.
    g.add_edge("load_data", "ground_wiki_initial")
    g.add_edge("ground_wiki_initial", "initial_analysis")
    g.add_edge("initial_analysis", END)
    # Ветвь оформления поручений (по запросу пользователя на последующих ходах).
    g.add_edge("extract_assignments", "propose_assignments")
    g.add_edge("propose_assignments", END)
    # Последующие ходы: классифицируем реплику и отвечаем.
    g.add_conditional_edges(
        "route",
        after_route,
        {
            "call_json_analyzer": "call_json_analyzer",
            "call_easyrag": "call_easyrag",
            "extract_assignments": "extract_assignments",
            "select_assignments": "select_assignments",
            "respond": "respond",
        },
    )
    g.add_edge("call_json_analyzer", "ground_wiki_analytics")
    g.add_edge("ground_wiki_analytics", "respond")
    g.add_edge("call_easyrag", "respond")
    g.add_edge("select_assignments", "commit_assignments")
    g.add_edge("commit_assignments", END)
    g.add_edge("respond", END)

    return g.compile()


llm = create_gigachat_client().get_llm()
graph = build_graph(llm)
