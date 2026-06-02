"""Структура графа оркестратора: инварианты первого хода и ветки поручений.

Регресс на поведение, заданное в graph.py: первый ход (load_data → анализ)
завершается ОДНИМ ответным сообщением — вызывающая система умеет работать только
с одним, поэтому initial_analysis уходит сразу в END, а не в extract_assignments.
При этом оформление поручений не теряется: оно достижимо роутером по intent
'assignments' на последующих ходах.

Тест строит граф детерминированно (фиктивный llm — узлы создаются лениво и в
момент сборки его не вызывают), поэтому ни сети, ни GigaChat-кредов не нужно.
Фабрика клиента на уровне модуля всё же конструирует GigaChat при импорте, но
без обращения к сети — отдаём заведомо валидную заглушку креды.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
# Импорт graph.py исполняет `llm = create_gigachat_client().get_llm()` на уровне
# модуля. GigaChat при конструировании в сеть не ходит, но требует креды — даём
# фиктивные, чтобы импорт не падал в чистом CI.
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from langgraph_executor.aegra_agents.analytic_orchestrator.graph import build_graph
from langgraph_executor.aegra_agents.analytic_orchestrator.nodes import after_route


def _edges():
    # Узлы замыкаются на llm лениво — для проверки топологии хватит заглушки.
    compiled = build_graph(object())
    return {(e.source, e.target) for e in compiled.get_graph().edges}


def test_first_turn_emits_single_message():
    edges = _edges()
    # Первый ход завершается ровно одним сообщением: анализ → END.
    assert ("initial_analysis", "__end__") in edges
    # И НЕ продолжается в авто-формирование поручений (старое поведение).
    assert ("initial_analysis", "extract_assignments") not in edges


def test_assignments_branch_reachable_via_router():
    edges = _edges()
    # Роутер ведёт в ветку поручений, и она доходит до конца.
    assert ("route", "extract_assignments") in edges
    assert ("extract_assignments", "propose_assignments") in edges
    assert ("propose_assignments", "__end__") in edges


def test_after_route_maps_assignments_intent():
    # Intent 'assignments' (его выставляет роутер по фразе «оформи поручения»,
    # которую и предлагает INITIAL_OFFER_HINT) маршрутизируется в extract_assignments.
    assert after_route({"intent": "assignments"}) == "extract_assignments"


def test_ground_wiki_initial_in_first_turn_path():
    edges = _edges()
    # Первый ход заземляется в wiki ДО первичного анализа.
    assert ("load_data", "ground_wiki_initial") in edges
    assert ("ground_wiki_initial", "initial_analysis") in edges
    # Прямого ребра load_data → initial_analysis больше нет.
    assert ("load_data", "initial_analysis") not in edges


def test_ground_wiki_analytics_after_json_analyzer():
    edges = _edges()
    # На analytics-ходу wiki-grounding идёт ПОСЛЕ аналитика (использует его выводы)
    # и перед респондером.
    assert ("call_json_analyzer", "ground_wiki_analytics") in edges
    assert ("ground_wiki_analytics", "respond") in edges
    # Прямого ребра call_json_analyzer → respond больше нет.
    assert ("call_json_analyzer", "respond") not in edges


def test_wiki_intent_path_unchanged():
    edges = _edges()
    # Явный wiki-интент остаётся отдельным single-query путём.
    assert ("call_easyrag", "respond") in edges
