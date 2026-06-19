from __future__ import annotations

from langchain_gigachat import GigaChat
from langgraph.graph import END, START, StateGraph

from ..easyrag.graph import graph as easyrag_graph
# Бизнес-оркестратор использует аналитика json_analyzer_v3 (= v2 + инструмент
# element_slice): без аналитики по бенчмарку, pop-сравнения только у метрик с
# указанным планом.
from ..json_analyzer_v3.graph import graph as json_analyzer_graph
from ..shared.clients import create_gigachat_client
from .nodes import (
    after_route,
    make_call_easyrag_node,
    make_call_json_analyzer_node,
    make_form_insights_node,
    make_ground_wiki_node,
    make_initial_analysis_node,
    make_load_data_node,
    make_respond_node,
    make_route_node,
    make_save_insights_node,
    need_load,
)
from .state import OrchestratorOutput, OrchestratorState


def build_graph(llm: GigaChat, checkpointer=None):
    # output_schema без metrics: полный датасет остаётся каналом стейта
    # (чекпойнтится, живёт между ходами), но не отдаётся наружу в run/stream.
    # checkpointer: в проде его подставляет aegra (per-thread); для локального
    # multi-turn прогона можно передать MemorySaver, иначе состояние между
    # ходами не сохранится (need_load полагается на персистентный loaded).
    g = StateGraph(OrchestratorState, output_schema=OrchestratorOutput)

    g.add_node("load_data", make_load_data_node())
    g.add_node(
        "initial_analysis",
        make_initial_analysis_node(llm, json_analyzer_graph),
    )
    g.add_node("route", make_route_node(llm))
    # Один и тот же узел wiki-grounding на двух позициях графа (у узла фиксированные
    # out-edges, поэтому два инстанса проще условных рёбер): перед первичным
    # анализом и после json_analyzer на analytics-ходу.
    g.add_node("ground_wiki_initial", make_ground_wiki_node(llm, easyrag_graph))
    g.add_node("ground_wiki_analytics", make_ground_wiki_node(llm, easyrag_graph))
    g.add_node("call_json_analyzer", make_call_json_analyzer_node(json_analyzer_graph))
    g.add_node("call_easyrag", make_call_easyrag_node(easyrag_graph))
    # Завершение анализа (post_insights): форма → подтверждение → сохранение.
    g.add_node("form_insights", make_form_insights_node(llm))
    g.add_node("save_insights", make_save_insights_node())
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
    # Первый ход: входящее сообщение — триггер. Грузим данные, делаем первичный
    # многоуровневый разбор по бизнес-методологии и завершаем ход блоком «Что
    # делаем дальше?». Контракт сообщений хода: рабочие узлы кладут короткие
    # «шаговые» сообщения (additional_kwargs.orchestrator_step), терминальный лист —
    # ИТОГОВЫЙ ответ (additional_kwargs.orchestrator_final, всегда последний).
    # Прогресс отключается флагом configurable.emit_progress_messages=false.
    g.add_edge("load_data", "ground_wiki_initial")
    g.add_edge("ground_wiki_initial", "initial_analysis")
    g.add_edge("initial_analysis", END)
    # Последующие ходы: классифицируем реплику и ведём её по нужной ветви.
    g.add_conditional_edges(
        "route",
        after_route,
        {
            "call_json_analyzer": "call_json_analyzer",
            "call_easyrag": "call_easyrag",
            "form_insights": "form_insights",
            "save_insights": "save_insights",
            "respond": "respond",
        },
    )
    g.add_edge("call_json_analyzer", "ground_wiki_analytics")
    g.add_edge("ground_wiki_analytics", "respond")
    g.add_edge("call_easyrag", "respond")
    # form_insights показывает инсайты и ждёт «Все верно?» (следующим ходом);
    # save_insights пишет в сервис и закрывает завершение. Оба — листья хода.
    g.add_edge("form_insights", END)
    g.add_edge("save_insights", END)
    g.add_edge("respond", END)

    return g.compile(checkpointer=checkpointer)


llm = create_gigachat_client().get_llm()
graph = build_graph(llm)
