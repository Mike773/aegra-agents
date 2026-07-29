"""Инсайты analytic_orchestrator_v4: привязка source_type/source_id, стартовая
запись без подтверждения и фиксация по явной просьбе.

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
    SAVE_INSIGHT_EMPTY,
    SAVE_INSIGHT_NO_SOURCE,
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


# --- Стартовый инсайт -------------------------------------------------------

def test_auto_insight_submits_main_problem_with_source(monkeypatch):
    calls = _fake_service(monkeypatch)
    node = make_auto_insight_node(FakeLLM(_insights_json()))
    out = asyncio.run(node(_state(), {"configurable": {"thread_id": "t-1"}}))

    assert len(calls) == 1
    call = calls[0]
    assert call["source_type"] == "meeting"
    assert call["source_id"] == "mtg-77"
    assert call["thread_id"] == "t-1"
    assert call["boss_tabnum"] == "999"
    assert call["employee_tabnum"] == "12345"
    # Ровно один инсайт — главная проблема.
    assert len(call["insights"]) == 1
    assert call["insights"][0]["type"] == "main_problem"
    assert out["committed_insights"][0]["metric_name"] == "Производительность"
    # Пользователю узел ничего не пишет — итог хода отдал initial_analysis.
    assert "messages" not in out


def test_auto_insight_falls_back_to_norm(monkeypatch):
    """Главной проблемы нет («всё в норме») — пишем вывод о норме."""
    calls = _fake_service(monkeypatch)
    node = make_auto_insight_node(FakeLLM(_insights_json(kind="norm")))
    asyncio.run(node(_state(), {}))
    assert len(calls) == 1
    assert calls[0]["insights"][0]["type"] == "norm"


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


def test_auto_insight_skipped_when_analysis_failed(monkeypatch):
    """Аналитик упал (metrics_summary пуст) — инсайт по тексту «данных нет» не пишем."""
    calls = _fake_service(monkeypatch)
    node = make_auto_insight_node(FakeLLM(_insights_json()))
    state = _state(metrics_summary=None, messages=[
        HumanMessage("что происходит?"),
        AIMessage("Данных по метрикам сейчас нет — уточните запрос."),
    ])
    out = asyncio.run(node(state, {}))
    assert calls == []
    assert "committed_insights" not in out


def test_auto_insight_classification_failure_is_not_fatal(monkeypatch):
    calls = _fake_service(monkeypatch)
    node = make_auto_insight_node(FakeLLM("не json"))
    out = asyncio.run(node(_state(), {}))
    assert calls == []
    assert out["reasoning_trace"][-1]["kind"] == "decision"


# --- Фиксация по явной просьбе ---------------------------------------------

def test_save_insight_submits_and_answers(monkeypatch):
    calls = _fake_service(monkeypatch)
    node = make_save_insight_node(FakeLLM(_insights_json(kind="problem")))
    state = _state(messages=[
        HumanMessage("что происходит?"),
        AIMessage("Производительность 12 при плане 18 — область развития."),
        HumanMessage("зафиксируй этот вывод"),
    ])
    out = asyncio.run(node(state, {"configurable": {"thread_id": "t-2"}}))
    assert len(calls) == 1
    assert calls[0]["source_id"] == "mtg-77"
    assert "Производительность" in out["messages"][-1].content
    assert len(out["committed_insights"]) == 1


def test_save_insight_without_source_explains(monkeypatch):
    calls = _fake_service(monkeypatch)
    node = make_save_insight_node(FakeLLM(_insights_json()))
    out = asyncio.run(node(_state(source_id=None), {}))
    assert calls == []
    assert out["messages"][-1].content == SAVE_INSIGHT_NO_SOURCE


def test_save_insight_nothing_to_save(monkeypatch):
    calls = _fake_service(monkeypatch)
    node = make_save_insight_node(FakeLLM(json.dumps({"insights": []})))
    out = asyncio.run(node(_state(), {}))
    assert calls == []
    assert out["messages"][-1].content == SAVE_INSIGHT_EMPTY


def test_save_insight_skips_already_committed(monkeypatch):
    """Повторная классификация снова выдаёт стартовый инсайт — не дублируем."""
    calls = _fake_service(monkeypatch)
    node = make_save_insight_node(FakeLLM(_insights_json()))
    committed = [{"type": "main_problem", "metric_name": "Производительность",
                  "metric_id": "90022908", "text": "..."}]
    out = asyncio.run(node(_state(committed_insights=committed), {}))
    assert calls == []
    assert out["messages"][-1].content == SAVE_INSIGHT_EMPTY


def test_save_insight_one_verdict_per_metric(monkeypatch):
    """Одна метрика за диалог = один вывод: противоречивых дублей быть не должно."""
    calls = _fake_service(monkeypatch)
    llm = FakeLLM(json.dumps({"insights": [
        {"type": "achievement", "metric_id": "1", "metric_name": "Доля переводов",
         "text": "Улучшилась на 61.6%."},
        {"type": "problem", "metric_id": "1", "metric_name": "Доля переводов",
         "text": "По переводам в рублях всплеск."},
        {"type": "norm", "metric_id": "2", "metric_name": "AHT", "text": "В плане."},
    ]}))
    asyncio.run(make_save_insight_node(llm)(_state(), {}))
    sent = calls[0]["insights"]
    assert [i["metric_name"] for i in sent] == ["Доля переводов", "AHT"]
    assert sent[0]["type"] == "achievement"  # оставляем первый вердикт


def test_save_insight_prefers_last_analytics_answer(monkeypatch):
    """«Этот вывод» — свежий разбор под последний вопрос, а не весь диалог."""
    _fake_service(monkeypatch)
    llm = FakeLLM(_insights_json(kind="problem"))
    node = make_save_insight_node(llm)
    asyncio.run(node(_state(analytics_answer="AHT 269.7 сек — в плане."), {}))
    prompt = llm.calls[0][-1].content
    assert "AHT 269.7 сек" in prompt
