from __future__ import annotations

from langchain_gigachat import GigaChat
from langgraph.graph import END, START, StateGraph

from ..easyrag.graph import graph as easyrag_graph
# Бизнес-оркестратор использует аналитика json_analyzer_v5: peer-агрегаты уровней
# и персональные rankings, поля ex (% выполнения плана) и rr (RunRate) в выдаче
# инструментов; без аналитики по бенчмарку, pop-сравнения только у метрик с планом.
from ..json_analyzer_v5.graph import graph as json_analyzer_graph
from ..long_term_memory.nodes import make_load_memory_node, make_save_memory_node
from ..shared.clients import create_gigachat_client
from .nodes import (
    after_route,
    make_auto_insight_node,
    make_call_easyrag_node,
    make_call_json_analyzer_node,
    make_ground_wiki_node,
    make_initial_analysis_node,
    make_load_data_node,
    make_respond_node,
    make_route_node,
    make_save_insight_node,
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
    # анализом и перед json_analyzer на analytics-ходу — сниппеты уходят
    # аналитику на вход (wiki_context) и дальше респондеру.
    g.add_node("ground_wiki_initial", make_ground_wiki_node(llm, easyrag_graph))
    g.add_node("ground_wiki_analytics", make_ground_wiki_node(llm, easyrag_graph))
    g.add_node("call_json_analyzer", make_call_json_analyzer_node(json_analyzer_graph))
    g.add_node("call_easyrag", make_call_easyrag_node(easyrag_graph))
    # Инсайты: стартовый пишется автоматически после первичного разбора,
    # остальные — по явной просьбе руководителя. Экрана подтверждения нет.
    g.add_node("auto_insight", make_auto_insight_node(llm))
    g.add_node("save_insight", make_save_insight_node(llm))
    g.add_node("respond", make_respond_node(llm))

    # Долгосрочная память (пока заглушка long_term_memory): загрузка контекста
    # прошлых диалогов на первом ходе и сохранение итогов в конце хода.
    g.add_node("load_memory", make_load_memory_node(llm))
    g.add_node("save_memory", make_save_memory_node())

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
    # многоуровневый разбор по бизнес-методологии и завершаем ход предложением,
    # что разобрать дальше. Контракт сообщений хода: рабочие узлы кладут короткие
    # «шаговые» сообщения (additional_kwargs.orchestrator_step), терминальный лист —
    # ИТОГОВЫЙ ответ (additional_kwargs.orchestrator_final, всегда последний).
    # Прогресс отключается флагом configurable.emit_progress_messages=false.
    g.add_edge("load_data", "load_memory")
    g.add_edge("load_memory", "ground_wiki_initial")
    g.add_edge("ground_wiki_initial", "initial_analysis")
    # Итог хода отдаёт initial_analysis; auto_insight — «хвост» первого хода:
    # пишет стартовый инсайт в сервис и НЕ добавляет сообщений пользователю.
    g.add_edge("initial_analysis", "auto_insight")
    g.add_edge("auto_insight", END)
    # Последующие ходы: классифицируем реплику и ведём её по нужной ветви.
    g.add_conditional_edges(
        "route",
        after_route,
        {
            "call_json_analyzer": "ground_wiki_analytics",
            "call_easyrag": "call_easyrag",
            "save_insight": "save_insight",
            "respond": "respond",
        },
    )
    g.add_edge("ground_wiki_analytics", "call_json_analyzer")
    g.add_edge("call_json_analyzer", "respond")
    g.add_edge("call_easyrag", "respond")
    # Ходы save_insight/respond завершаются сохранением долгосрочной памяти.
    g.add_edge("save_insight", "save_memory")
    g.add_edge("respond", "save_memory")
    g.add_edge("save_memory", END)

    return g.compile(checkpointer=checkpointer)


llm = create_gigachat_client().get_llm()
graph = build_graph(llm)
