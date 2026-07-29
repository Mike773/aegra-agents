"""Структура графа бизнес-оркестратора (analytic_orchestrator_v4).

Отличия от v2/v3: первый ход продолжается узлом auto_insight (пишет стартовый
инсайт в сервис и НЕ добавляет сообщений), процесса подтверждения
form_insights → «Все верно?» → save_insights больше нет, вместо него — ветка
save_insight по явной просьбе руководителя.

Граф строится детерминированно с заглушкой llm (узлы ленивы), поэтому ни сети,
ни GigaChat-кредов не нужно; фабрика клиента на уровне модуля требует креды —
даём фиктивные.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from langgraph_executor.aegra_agents.analytic_orchestrator_v4.graph import build_graph
from langgraph_executor.aegra_agents.analytic_orchestrator_v4.nodes import after_route


def _edges():
    compiled = build_graph(object())
    return {(e.source, e.target) for e in compiled.get_graph().edges}


def test_first_turn_ends_with_auto_insight():
    edges = _edges()
    assert ("initial_analysis", "auto_insight") in edges
    assert ("auto_insight", "__end__") in edges
    # Итог хода отдаёт initial_analysis, но лист хода — auto_insight.
    assert ("initial_analysis", "__end__") not in edges


def test_first_turn_grounds_wiki_before_analysis():
    edges = _edges()
    assert ("load_data", "ground_wiki_initial") in edges
    assert ("ground_wiki_initial", "initial_analysis") in edges
    assert ("load_data", "initial_analysis") not in edges


def test_confirmation_flow_removed():
    nodes = {n for e in _edges() for n in e}
    assert "form_insights" not in nodes
    assert "save_insights" not in nodes
    # Вместо него — фиксация по явной просьбе, лист хода.
    assert ("route", "save_insight") in _edges()
    assert ("save_insight", "__end__") in _edges()


def test_analytics_path_through_wiki_grounding():
    edges = _edges()
    assert ("call_json_analyzer", "ground_wiki_analytics") in edges
    assert ("ground_wiki_analytics", "respond") in edges
    assert ("call_json_analyzer", "respond") not in edges
    assert ("call_easyrag", "respond") in edges


def test_after_route_intent_mapping():
    assert after_route({"intent": "analytics"}) == "call_json_analyzer"
    assert after_route({"intent": "wiki"}) == "call_easyrag"
    assert after_route({"intent": "save_insight"}) == "save_insight"
    assert after_route({"intent": "chat"}) == "respond"
    assert after_route({"intent": "done"}) == "respond"
    # Интентов завершения из v2 больше нет — любой неизвестный к респондеру.
    assert after_route({"intent": "finish"}) == "respond"
    assert after_route({"intent": "more_analysis"}) == "respond"
