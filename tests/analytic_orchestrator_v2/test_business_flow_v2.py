"""Детерминированные тесты бизнес-поведения analytic_orchestrator_v2.

Импортируем только nodes.py (не graph.py) — GigaChat-креды/сеть не нужны.
Фейк-LLM возвращает заранее заданный .content; внешний сервис сохранения инсайтов
подменяем фейком. Асинхронные узлы гоняем через asyncio.run.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langchain_core.messages import AIMessage, HumanMessage

from langgraph_executor.aegra_agents.analytic_orchestrator_v2 import nodes
from langgraph_executor.aegra_agents.analytic_orchestrator_v2.nodes import (
    _org_structure_block,
    _parse_continuation_options,
    _resolve_more_analysis_question,
    make_form_insights_node,
    make_route_node,
    make_save_insights_node,
)
from langgraph_executor.aegra_agents.analytic_orchestrator_v2.prompts import (
    FORM_INSIGHTS_OUTRO,
    SAVE_CANCEL_PROMPT,
    SAVE_DONE_PROMPT,
)


# --- Фейки -----------------------------------------------------------------

class _AIMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content):
        self._content = content
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _AIMsg(self._content)


# Датасет в форме loader-а: {me, employees:[{fio, metrics:[{id, metric_name,...}]}]}.
def _dataset():
    return {
        "me": {"fio": "Босс Борис", "metrics": []},
        "employees": [{
            "fio": "Иванов Иван",
            "metrics": [
                {"id": "90022908", "metric_name": "Производительность",
                 "metric_description": "Выработка за смену", "metric_type": "прямая",
                 "measure_type": "у.е.", "date": "2026-05-11", "calc_period": "д",
                 "fact": 12, "plan": 18, "benchmark": 16, "element": None},
            ],
        }],
    }


# --- Хелперы продолжения / разрешения «2» -----------------------------------

CONT_TEXT = (
    "Резюме: всё стабильно.\n\n"
    "Что делаем дальше?\n"
    "1. Завершить анализ и сформулировать вопросы для отработки?\n"
    "2. Разобрать динамику производительности по дням недели\n"
    "3. Есть ли у тебя дополнительные вопросы по анализу?"
)


def test_parse_continuation_options():
    opts = _parse_continuation_options(CONT_TEXT)
    assert len(opts) == 3
    assert opts[0].startswith("Завершить анализ")
    assert opts[1] == "Разобрать динамику производительности по дням недели"
    assert opts[2].startswith("Есть ли у тебя")
    # Нет блока — пусто.
    assert _parse_continuation_options("Просто текст без вариантов") == []


def test_resolve_more_analysis_question():
    state = {"pending_options": _parse_continuation_options(CONT_TEXT)}
    # Голый «2» → текст второго варианта.
    assert _resolve_more_analysis_question(state, "2") == \
        "Разобрать динамику производительности по дням недели"
    assert _resolve_more_analysis_question(state, "вариант 2").startswith("Разобрать")
    # Свой вопрос не трогаем.
    own = "А что по AHT за последнюю неделю?"
    assert _resolve_more_analysis_question(state, own) == own
    # Нет сохранённых вариантов → возвращаем как есть.
    assert _resolve_more_analysis_question({}, "2") == "2"


def test_org_structure_block():
    block = _org_structure_block({"metrics": _dataset(), "position": "оператор"})
    assert block is not None
    assert "Босс Борис" in block
    assert "Иванов Иван" in block
    assert "оператор" in block
    # Нет датасета → None.
    assert _org_structure_block({"metrics": None}) is None


# --- Роутер: завершение и подтверждение -------------------------------------

def test_route_confirm_mapping():
    state = {
        "messages": [HumanMessage("да, всё верно")],
        "pending_confirmation": True,
    }
    route = make_route_node(FakeLLM("confirm"))
    assert route(state)["intent"] == "finish_save"
    route = make_route_node(FakeLLM("edit"))
    assert route(state)["intent"] == "finish_reform"
    route = make_route_node(FakeLLM("cancel"))
    assert route(state)["intent"] == "finish_cancel"


def test_route_normal_labels():
    state = {"messages": [HumanMessage("1")]}
    # «1» → finish (роутер-LLM так классифицирует).
    assert make_route_node(FakeLLM("finish"))(state)["intent"] == "finish"
    # Неизвестный ярлык → chat (безопасная деградация).
    assert make_route_node(FakeLLM("чтототакое"))(state)["intent"] == "chat"


# --- post_insights: форма → подтверждение → сохранение ----------------------

def _dialogue_state():
    return {
        "metrics": _dataset(),
        "employee_tabnum": "12345",
        "direction_key": "dir-1",
        "messages": [
            HumanMessage("что происходит?"),
            AIMessage("Производительность 12 при плане 18 — область развития."),
        ],
    }


def test_form_insights_sets_pending_and_asks_confirm():
    insights_json = json.dumps({"insights": [
        {"type": "main_problem", "metric_id": "90022908",
         "metric_name": "Производительность",
         "text": "Производительность 12 при плане 18 — фокус внимания."},
    ]})
    node = make_form_insights_node(FakeLLM(insights_json))
    out = asyncio.run(node(_dialogue_state(), {}))
    # Сформировал, показал на подтверждение, в БД ещё НЕ сохранил.
    assert out["pending_confirmation"] is True
    assert len(out["candidate_assignments"]) == 1
    assert out["candidate_assignments"][0]["metric_name"] == "Производительность"
    text = out["messages"][-1].content
    assert FORM_INSIGHTS_OUTRO in text          # «Все верно?»
    assert "Производительность" in text


def test_form_insights_empty_clears_pending():
    node = make_form_insights_node(FakeLLM(json.dumps({"insights": []})))
    out = asyncio.run(node(_dialogue_state(), {}))
    assert out["pending_confirmation"] is False
    assert out["candidate_assignments"] == []


def test_save_insights_confirm_submits(monkeypatch):
    captured = {}

    class FakeSend:
        def __init__(self, boss_tabnum, employee_tabnum, direction_key,
                     thread_id, insights):
            captured["boss"] = boss_tabnum
            captured["employee"] = employee_tabnum
            captured["thread_id"] = thread_id
            captured["insights"] = insights

        def submit(self):
            captured["submitted"] = True

    monkeypatch.setattr(nodes, "SendAssignmentsComponent", FakeSend)
    save = make_save_insights_node()
    state = {
        "intent": "finish_save",
        "candidate_assignments": [{"type": "problem", "metric_id": "1",
                                   "metric_name": "Производительность", "text": "..."}],
        "pending_confirmation": True,
        "employee_tabnum": "12345",
        "boss_tabnum": "999",
        "direction_key": "dir-1",
        "messages": [HumanMessage("да")],
    }
    out = save(state, {"configurable": {"thread_id": "thread-abc"}})
    assert captured.get("submitted") is True
    assert captured["boss"] == "999"
    assert captured["employee"] == "12345"
    assert captured["thread_id"] == "thread-abc"
    assert out["messages"][-1].content == SAVE_DONE_PROMPT
    assert out["pending_confirmation"] is False
    assert out["candidate_assignments"] == []
    assert len(out["last_committed_assignments"]) == 1


def test_save_insights_cancel_does_not_submit(monkeypatch):
    called = {"submitted": False}

    class FakeSend:
        def __init__(self, **kw):
            pass

        def submit(self):
            called["submitted"] = True

    monkeypatch.setattr(nodes, "SendAssignmentsComponent", FakeSend)
    save = make_save_insights_node()
    state = {
        "intent": "finish_cancel",
        "candidate_assignments": [{"type": "problem", "metric_name": "X", "text": "y"}],
        "pending_confirmation": True,
        "messages": [HumanMessage("нет, отмена")],
    }
    out = save(state, {})
    assert called["submitted"] is False
    assert out["messages"][-1].content == SAVE_CANCEL_PROMPT
    assert out["pending_confirmation"] is False
    assert out["candidate_assignments"] == []
