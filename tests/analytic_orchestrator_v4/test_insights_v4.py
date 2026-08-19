"""Инсайты analytic_orchestrator_v4: один главный инсайт из опорного профиля.

auto_insight шлёт в сервис ОДИН главный вывод (поле insight — объект, не
массив): топ-проблему из профиля сотрудника (кеш по source_id), без проблем —
топ-достижение. Профиля нет — фолбэк на LLM-классификацию первичного разбора.
Ручной фиксации больше нет: save_insight отвечает, что главный вывод уходит
автоматически, и ничего не отправляет.

Импортируем только nodes.py (не graph.py) — GigaChat-креды/сеть не нужны.
Фейк-LLM возвращает заранее заданный .content; внешний сервис инсайтов подменяем
фейком. Асинхронные узлы гоняем через asyncio.run.
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

from langgraph_executor.aegra_agents.analytic_orchestrator_v4 import nodes
from langgraph_executor.aegra_agents.analytic_orchestrator_v4.nodes import (
    make_auto_insight_node,
    make_save_insight_node,
)
from langgraph_executor.aegra_agents.analytic_orchestrator_v4.prompts import (
    SAVE_INSIGHT_AUTO_FIXED,
)
from langgraph_executor.aegra_agents.shared.assignments_service import (
    SendAssignmentsComponent,
)


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


def _profile(problems=True, achievements=False):
    prof = {
        "status": "зона риска",
        "confidence_note": "средняя",
        "key_facts": "факт 12 при плане 18",
        "problems": [],
        "achievements": [],
        "secondary": [],
    }
    if problems:
        prof["problems"] = [{
            "metric_id": "90022908", "metric_name": "Производительность",
            "headline": "Производительность на 33% ниже плана",
            "hypothesis": "…", "what_to_check": "…",
            "meeting_topic": "…", "assignment_draft": "…",
        }]
    if achievements:
        prof["achievements"] = [{
            "metric_id": "77", "metric_name": "Доля переводов",
            "headline": "Доля переводов стабильно выше медианы коллег",
        }]
    return prof


def _insights_json(kind="main_problem"):
    return json.dumps({"insights": [
        {"type": kind, "metric_id": "90022908",
         "metric_name": "Производительность",
         "text": "Производительность 12 при плане 18 — фокус внимания."},
    ]})


def _state(**over):
    state = {
        "metrics": _dataset(),
        "metrics_summary": "Производительность 12 при плане 18.",
        "employee_profile": _profile(),
        "boss_tabnum": "999",
        "employee_tabnum": "12345",
        "direction_key": "dir-1",
        "source_type": "meeting",
        "source_id": "mtg-77",
        "messages": [
            HumanMessage("что происходит?"),
            AIMessage("Производительность 12 при плане 18 — область развития."),
        ],
    }
    state.update(over)
    return state


class _Capture:
    """Фейк сервиса инсайтов: пишет вызовы в общий список."""

    calls: list[dict] = []

    def __init__(self, **kw):
        self.kw = kw

    def submit(self):
        type(self).calls.append(self.kw)


def _fake_service(monkeypatch):
    _Capture.calls = []
    monkeypatch.setattr(nodes, "SendAssignmentsComponent", _Capture)
    return _Capture.calls


# --- Стартовый инсайт из профиля --------------------------------------------

def test_auto_insight_takes_main_problem_from_profile(monkeypatch):
    calls = _fake_service(monkeypatch)
    llm = FakeLLM(_insights_json())
    node = make_auto_insight_node(llm)
    out = asyncio.run(node(_state(), {"configurable": {"thread_id": "t-1"}}))

    assert len(calls) == 1
    call = calls[0]
    assert call["source_type"] == "meeting"
    assert call["source_id"] == "mtg-77"
    assert call["thread_id"] == "t-1"
    assert call["boss_tabnum"] == "999"
    assert call["employee_tabnum"] == "12345"
    # Один главный инсайт объектом — из профиля, а не из LLM-классификации.
    assert "insights" not in call
    assert call["insight"] == {
        "type": "main_problem",
        "metric_id": "90022908",
        "metric_name": "Производительность",
        "text": "Производительность на 33% ниже плана",
    }
    assert llm.calls == []
    assert out["committed_insights"][0]["metric_name"] == "Производительность"
    assert any("опорный профиль" in s.get("summary", "")
               for s in out["reasoning_trace"])
    # Пользователю узел ничего не пишет — итог хода отдал initial_analysis.
    assert "messages" not in out


def test_auto_insight_profile_achievement_when_no_problems(monkeypatch):
    calls = _fake_service(monkeypatch)
    node = make_auto_insight_node(FakeLLM(_insights_json()))
    state = _state(employee_profile=_profile(problems=False, achievements=True))
    asyncio.run(node(state, {}))
    assert calls[0]["insight"]["type"] == "achievement"
    assert calls[0]["insight"]["text"] == (
        "Доля переводов стабильно выше медианы коллег"
    )


# --- Фолбэк без профиля: старая классификация -------------------------------

def test_auto_insight_without_profile_falls_back_to_classification(monkeypatch):
    calls = _fake_service(monkeypatch)
    llm = FakeLLM(_insights_json())
    node = make_auto_insight_node(llm)
    out = asyncio.run(node(_state(employee_profile=None), {}))
    assert len(llm.calls) == 1
    assert calls[0]["insight"]["type"] == "main_problem"
    assert calls[0]["insight"]["text"] == (
        "Производительность 12 при плане 18 — фокус внимания."
    )
    assert any("классификация разбора" in s.get("summary", "")
               for s in out["reasoning_trace"])


def test_auto_insight_fallback_to_norm(monkeypatch):
    """Главной проблемы нет («всё в норме») — пишем вывод о норме."""
    calls = _fake_service(monkeypatch)
    node = make_auto_insight_node(FakeLLM(_insights_json(kind="norm")))
    asyncio.run(node(_state(employee_profile=None), {}))
    assert len(calls) == 1
    assert calls[0]["insight"]["type"] == "norm"


def test_auto_insight_skipped_when_no_profile_and_no_analysis(monkeypatch):
    """Ни профиля, ни разбора (аналитик упал) — инсайт не выдумываем."""
    calls = _fake_service(monkeypatch)
    node = make_auto_insight_node(FakeLLM(_insights_json()))
    state = _state(employee_profile=None, metrics_summary=None, messages=[
        HumanMessage("что происходит?"),
        AIMessage("Данных по метрикам сейчас нет — уточните запрос."),
    ])
    out = asyncio.run(node(state, {}))
    assert calls == []
    assert "committed_insights" not in out


def test_auto_insight_classification_failure_is_not_fatal(monkeypatch):
    calls = _fake_service(monkeypatch)
    node = make_auto_insight_node(FakeLLM("не json"))
    out = asyncio.run(node(_state(employee_profile=None), {}))
    assert calls == []
    assert out["reasoning_trace"][-1]["kind"] == "decision"


# --- Общие гейты и сбои ------------------------------------------------------

def test_auto_insight_without_source_writes_nothing(monkeypatch):
    calls = _fake_service(monkeypatch)
    node = make_auto_insight_node(FakeLLM(_insights_json()))
    out = asyncio.run(node(_state(source_type=None, source_id=None), {}))
    assert calls == []
    assert "committed_insights" not in out
    assert out["reasoning_trace"][-1]["kind"] == "decision"


def test_auto_insight_service_failure_is_not_fatal(monkeypatch):
    class Boom:
        def __init__(self, **kw):
            pass

        def submit(self):
            raise RuntimeError("сервис лёг")

    monkeypatch.setattr(nodes, "SendAssignmentsComponent", Boom)
    node = make_auto_insight_node(FakeLLM(_insights_json()))
    out = asyncio.run(node(_state(), {}))
    assert "messages" not in out
    assert out["reasoning_trace"][-1]["kind"] == "error"


# --- Ручной фиксации больше нет ----------------------------------------------

# Тексты ответов сверяем ДОСЛОВНО — выключаем дефолтную HTML-конвертацию
# (answer_html), она проверяется отдельно в test_answer_html.py.
_NO_HTML = {"configurable": {"answer_html": False}}


def test_save_insight_explains_auto_fixation_and_sends_nothing(monkeypatch):
    calls = _fake_service(monkeypatch)
    llm = FakeLLM(_insights_json())
    node = make_save_insight_node(llm)
    state = _state(messages=[
        HumanMessage("что происходит?"),
        AIMessage("Производительность 12 при плане 18 — область развития."),
        HumanMessage("зафиксируй этот вывод"),
    ])
    out = asyncio.run(node(state, _NO_HTML))
    assert calls == []
    assert llm.calls == []
    assert out["messages"][-1].content == SAVE_INSIGHT_AUTO_FIXED
    assert "committed_insights" not in out


# --- Контракт заглушки сервиса -----------------------------------------------

def _component_kwargs(**over):
    kw = {
        "boss_tabnum": "999", "employee_tabnum": "12345",
        "direction_key": "dir-1", "thread_id": "t-1",
    }
    kw.update(over)
    return kw


def test_assignments_payload_single_insight():
    ins = {"type": "main_problem", "metric_id": "1",
           "metric_name": "Производительность", "text": "ниже плана"}
    payload = SendAssignmentsComponent(
        **_component_kwargs(insight=ins, source_type="meeting", source_id="m-1")
    ).submit()
    assert payload["content"] == {"insight": ins}
    assert payload["source_type"] == "meeting"
    assert payload["source_id"] == "m-1"


def test_assignments_payload_legacy_insights_list():
    """Старые оркестраторы (v1–v3) продолжают слать массив insights."""
    ins = [{"type": "norm", "metric_id": "2", "metric_name": "AHT", "text": "в плане"}]
    payload = SendAssignmentsComponent(**_component_kwargs(insights=ins)).submit()
    assert payload["content"] == {"insights": ins}
