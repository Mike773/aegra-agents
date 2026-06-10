"""Структура графа бизнес-оркестратора (analytic_orchestrator_v2).

Первый ход (load_data → wiki → первичный разбор) завершается ОДНИМ сообщением.
Завершение анализа (post_insights) — отдельные листья form_insights/save_insights.
Граф строится детерминированно с заглушкой llm (узлы ленивы), поэтому ни сети, ни
GigaChat-кредов не нужно; фабрика клиента на уровне модуля требует креды — даём
фиктивные.
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


def test_first_turn_emits_single_message():
    edges = _edges()
    assert ("initial_analysis", "__end__") in edges
    # Первый ход НЕ продолжается в авто-формирование выводов.
    assert ("initial_analysis", "form_insights") not in edges


def test_first_turn_grounds_wiki_before_analysis():
    edges = _edges()
    assert ("load_data", "ground_wiki_initial") in edges
    assert ("ground_wiki_initial", "initial_analysis") in edges
    assert ("load_data", "initial_analysis") not in edges


def test_finish_branches_are_leaves():
    edges = _edges()
    # Завершение анализа: форма и сохранение — отдельные листья хода.
    assert ("route", "form_insights") in edges
    assert ("route", "save_insights") in edges
    assert ("form_insights", "__end__") in edges
    assert ("save_insights", "__end__") in edges


def test_analytics_path_through_wiki_grounding():
    edges = _edges()
    assert ("call_json_analyzer", "ground_wiki_analytics") in edges
    assert ("ground_wiki_analytics", "respond") in edges
    assert ("call_json_analyzer", "respond") not in edges
    assert ("call_easyrag", "respond") in edges


def test_after_route_intent_mapping():
    # Содержательный анализ.
    assert after_route({"intent": "analytics"}) == "call_json_analyzer"
    assert after_route({"intent": "more_analysis"}) == "call_json_analyzer"
    assert after_route({"intent": "wiki"}) == "call_easyrag"
    # Завершение: форма/переформа и сохранение/отмена.
    assert after_route({"intent": "finish"}) == "form_insights"
    assert after_route({"intent": "finish_reform"}) == "form_insights"
    assert after_route({"intent": "finish_save"}) == "save_insights"
    assert after_route({"intent": "finish_cancel"}) == "save_insights"
    # Прочее — к респондеру.
    assert after_route({"intent": "ask_question"}) == "respond"
    assert after_route({"intent": "chat"}) == "respond"
    assert after_route({"intent": "done"}) == "respond"


def test_metrics_excluded_from_output_channels():
    g = build_graph(object())
    assert "metrics" not in g.output_channels
    assert "messages" in g.output_channels
    assert "metrics_error" in g.output_channels
