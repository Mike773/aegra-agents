"""Структура графа бизнес-оркестратора (analytic_orchestrator_v2).

Первый ход (load_data → wiki → первичный разбор) завершается ОДНИМ сообщением, а
хвостом хода auto_insight пишет стартовый инсайт. Фиксация по просьбе — лист
save_insight; экрана подтверждения нет. Граф строится детерминированно с заглушкой
llm (узлы ленивы), поэтому ни сети, ни GigaChat-кредов не нужно; фабрика клиента на
уровне модуля требует креды — даём фиктивные.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from langgraph_executor.aegra_agents.analytic_orchestrator_v2.graph import build_graph
from langgraph_executor.aegra_agents.analytic_orchestrator_v2.nodes import after_route


def _edges():
    compiled = build_graph(object())
    return {(e.source, e.target) for e in compiled.get_graph().edges}


def test_first_turn_tail_writes_start_insight():
    edges = _edges()
    # Итог хода отдаёт initial_analysis, а auto_insight — «немой» хвост первого
    # хода: пишет стартовый инсайт в сервис и закрывает ход.
    assert ("initial_analysis", "auto_insight") in edges
    assert ("auto_insight", "__end__") in edges
    assert ("initial_analysis", "__end__") not in edges


def test_first_turn_grounds_wiki_before_analysis():
    edges = _edges()
    assert ("load_data", "ground_wiki_initial") in edges
    assert ("ground_wiki_initial", "initial_analysis") in edges
    assert ("load_data", "initial_analysis") not in edges


def test_save_insight_is_leaf_and_confirmation_removed():
    edges = _edges()
    assert ("route", "save_insight") in edges
    assert ("save_insight", "__end__") in edges
    # Экран подтверждения «Все верно?» убран целиком.
    nodes_in_graph = {n for e in edges for n in e}
    assert "form_insights" not in nodes_in_graph
    assert "save_insights" not in nodes_in_graph


def test_analytics_path_through_wiki_grounding():
    edges = _edges()
    assert ("call_json_analyzer", "ground_wiki_analytics") in edges
    assert ("ground_wiki_analytics", "respond") in edges
    assert ("call_json_analyzer", "respond") not in edges
    assert ("call_easyrag", "respond") in edges


def test_after_route_intent_mapping():
    # Содержательный анализ.
    assert after_route({"intent": "analytics"}) == "call_json_analyzer"
    assert after_route({"intent": "wiki"}) == "call_easyrag"
    # Явная просьба зафиксировать вывод.
    assert after_route({"intent": "save_insight"}) == "save_insight"
    # Прочее и неожиданное — к респондеру.
    assert after_route({"intent": "chat"}) == "respond"
    assert after_route({"intent": "done"}) == "respond"
    assert after_route({"intent": "finish"}) == "respond"


def test_metrics_excluded_from_output_channels():
    g = build_graph(object())
    assert "metrics" not in g.output_channels
    assert "messages" in g.output_channels
    assert "metrics_error" in g.output_channels
