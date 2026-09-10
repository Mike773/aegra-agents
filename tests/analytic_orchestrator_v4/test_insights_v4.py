"""Инсайты analytic_orchestrator_v4: один главный инсайт объектом content.insight.

В сервис пишет только auto_insight на первом ходе треда — один главный инсайт
(параметр insight, в payload — объект content.insight). Ручная фиксация
save_insight ничего не отправляет: отвечает, что главный вывод уже зафиксирован
автоматически. Заглушка сервиса двухрежимная: легаси-массив v1–v3 не тронут.

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


# --- Стартовый инсайт (единственная запись в сервис за тред) -----------------

def test_auto_insight_submits_single_main_insight(monkeypatch):
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
    # ОДИН главный инсайт объектом, легаси-массив не передаётся.
    assert call["insight"]["type"] == "main_problem"
    assert call["insight"]["metric_name"] == "Производительность"
    assert call["insight"]["fact"] == 12 and call["insight"]["plan"] == 18
    assert "insights" not in call
    # Дедуп-механики больше нет — узел не копит committed_insights.
    assert "committed_insights" not in out
    # Пользователю узел ничего не пишет — итог хода отдал initial_analysis.
    assert "messages" not in out


def test_auto_insight_falls_back_to_norm(monkeypatch):
    """Главной проблемы нет («всё в норме») — пишем вывод о норме."""
    calls = _fake_service(monkeypatch)
    node = make_auto_insight_node(FakeLLM(_insights_json(kind="norm")))
    asyncio.run(node(_state(), {}))
    assert len(calls) == 1
    assert calls[0]["insight"]["type"] == "norm"


def test_auto_insight_without_source_writes_nothing(monkeypatch):
    calls = _fake_service(monkeypatch)
    node = make_auto_insight_node(FakeLLM(_insights_json()))
    out = asyncio.run(node(_state(source_type=None, source_id=None), {}))
    assert calls == []
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
    assert out["reasoning_trace"][-1]["kind"] == "decision"


def test_auto_insight_classification_failure_is_not_fatal(monkeypatch):
    calls = _fake_service(monkeypatch)
    node = make_auto_insight_node(FakeLLM("не json"))
    out = asyncio.run(node(_state(), {}))
    assert calls == []
    assert out["reasoning_trace"][-1]["kind"] == "decision"


# --- Явная просьба «зафиксируй вывод»: ручной записи больше нет --------------

# Сверяем текст ответа ДОСЛОВНО — выключаем дефолтную HTML-конвертацию
# (answer_html), она проверяется отдельно в test_answer_html.py.
_NO_HTML = {"configurable": {"answer_html": False}}


def test_save_insight_answers_auto_fixed_and_sends_nothing(monkeypatch):
    calls = _fake_service(monkeypatch)
    llm = FakeLLM(_insights_json(kind="problem"))
    node = make_save_insight_node(llm)
    state = _state(messages=[
        HumanMessage("что происходит?"),
        AIMessage("Производительность 12 при плане 18 — область развития."),
        HumanMessage("зафиксируй этот вывод"),
    ])
    out = asyncio.run(node(state, _NO_HTML))
    assert calls == []
    # Без LLM: узел только объясняет, что главный вывод уходит автоматически.
    assert llm.calls == []
    assert out["messages"][-1].content == SAVE_INSIGHT_AUTO_FIXED
    assert "committed_insights" not in out


def test_save_insight_answers_same_without_source(monkeypatch):
    """Ответ одинаковый и без привязки — узел в сервис не ходит вовсе."""
    calls = _fake_service(monkeypatch)
    node = make_save_insight_node(FakeLLM(_insights_json()))
    out = asyncio.run(node(_state(source_id=None), _NO_HTML))
    assert calls == []
    assert out["messages"][-1].content == SAVE_INSIGHT_AUTO_FIXED


# --- Заглушка сервиса: двухрежимный payload ----------------------------------

def test_service_stub_insight_object_mode():
    payload = SendAssignmentsComponent(
        boss_tabnum="999", employee_tabnum="12345", direction_key="dir-1",
        thread_id="t-1",
        insight={"type": "main_problem", "metric_id": "90022908",
                 "metric_name": "Производительность", "text": "..."},
        source_type="meeting", source_id="mtg-77",
    ).submit()
    assert payload["content"] == {"insight": {
        "type": "main_problem", "metric_id": "90022908",
        "metric_name": "Производительность", "text": "...",
    }}
    assert "insights" not in payload["content"]
    assert payload["source_type"] == "meeting"
    assert payload["source_id"] == "mtg-77"


def test_service_stub_legacy_list_mode():
    """v1–v3 передают insights-массив — payload прежний."""
    payload = SendAssignmentsComponent(
        boss_tabnum="999", employee_tabnum="12345", direction_key="dir-1",
        thread_id="t-1",
        insights=[{"type": "problem", "metric_id": "1",
                   "metric_name": "AHT", "text": "..."}],
    ).submit()
    assert payload["content"] == {"insights": [{
        "type": "problem", "metric_id": "1", "metric_name": "AHT", "text": "...",
    }]}
    assert "insight" not in payload["content"]


# --- Сигнальный режим (run_mode=signal) --------------------------------------

class _QueueLLM:
    """Ответы по очереди: первый — классификатору инсайтов, второй — вердикту."""

    def __init__(self, *contents):
        self._contents = list(contents)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _AIMsg(self._contents.pop(0) if self._contents else "")


_SIGNAL_CFG = {"configurable": {"thread_id": "t-1", "run_mode": "signal"}}
_SIGNAL_KEYS = ("author", "confirmed", "signal", "signal_description")


def test_signal_mode_adds_fields_to_insight(monkeypatch):
    calls = _fake_service(monkeypatch)
    llm = _QueueLLM(_insights_json(), '{"signal": true}')
    out = asyncio.run(make_auto_insight_node(llm)(_state(), _SIGNAL_CFG))

    assert len(calls) == 1
    call = calls[0]
    assert call["source_type"] == "meeting" and call["source_id"] == "mtg-77"
    ins = call["insight"]
    assert ins["type"] == "main_problem"
    assert ins["metric_name"] == "Производительность"
    assert ins["author"] == "Agent"
    assert ins["confirmed"] is True
    assert ins["signal"] is True
    assert ins["signal_description"] == "Производительность 12 при плане 18."
    assert len(llm.calls) == 2
    verdict_prompt = llm.calls[1][-1].content
    assert "что происходит?" in verdict_prompt
    assert "Производительность 12" in verdict_prompt
    assert out["reasoning_trace"][-1]["detail"]["signal"] is True


def test_signal_mode_sends_norm_when_classifier_empty(monkeypatch):
    calls = _fake_service(monkeypatch)
    llm = _QueueLLM("[]", '{"signal": false}')
    asyncio.run(make_auto_insight_node(llm)(_state(), _SIGNAL_CFG))

    assert len(calls) == 1
    ins = calls[0]["insight"]
    assert ins["type"] == "norm"
    assert ins["metric_id"] is None and ins["metric_name"] is None
    assert ins["fact"] is None and ins["plan"] is None
    assert ins["signal"] is False
    assert ins["signal_description"] == "Производительность 12 при плане 18."


def test_signal_mode_verdict_failure_falls_back_to_type(monkeypatch):
    calls = _fake_service(monkeypatch)
    llm = _QueueLLM(_insights_json(), "затрудняюсь")
    asyncio.run(make_auto_insight_node(llm)(_state(), _SIGNAL_CFG))
    assert calls[0]["insight"]["signal"] is True


def test_signal_mode_is_case_insensitive(monkeypatch):
    calls = _fake_service(monkeypatch)
    llm = _QueueLLM(_insights_json(), '{"signal": true}')
    cfg = {"configurable": {"thread_id": "t-1", "run_mode": " SIGNAL "}}
    asyncio.run(make_auto_insight_node(llm)(_state(), cfg))
    assert calls[0]["insight"]["signal"] is True


def test_no_signal_fields_without_run_mode(monkeypatch):
    calls = _fake_service(monkeypatch)
    llm = FakeLLM(_insights_json())
    asyncio.run(make_auto_insight_node(llm)(_state(), {"configurable": {"thread_id": "t-1"}}))
    assert not any(k in calls[0]["insight"] for k in _SIGNAL_KEYS)
    assert len(llm.calls) == 1


def test_signal_mode_skips_when_no_summary(monkeypatch):
    calls = _fake_service(monkeypatch)
    llm = _QueueLLM()
    asyncio.run(make_auto_insight_node(llm)(_state(metrics_summary=""), _SIGNAL_CFG))
    assert calls == []
    assert llm.calls == []


def test_signal_mode_skips_without_source(monkeypatch):
    calls = _fake_service(monkeypatch)
    llm = _QueueLLM()
    asyncio.run(make_auto_insight_node(llm)(_state(source_id=None), _SIGNAL_CFG))
    assert calls == []
    assert llm.calls == []


# --- plan/fact метрики в инсайте ---------------------------------------------

def test_insight_fact_plan_take_latest_row_of_employee(monkeypatch):
    """Несколько дат и разрез по element: берём последний общий срез сотрудника."""
    calls = _fake_service(monkeypatch)
    data = _dataset()
    m = data["employees"][0]["metrics"]
    m.append({**m[0], "date": "2026-05-04", "fact": 9, "plan": 17})
    m.append({**m[0], "date": "2026-05-18", "fact": 14, "plan": 19})
    m.append({**m[0], "date": "2026-05-25", "fact": 3, "plan": 5, "element": "Офис А"})
    asyncio.run(make_auto_insight_node(FakeLLM(_insights_json()))(
        _state(metrics=data), {"configurable": {"thread_id": "t-1"}}
    ))
    ins = calls[0]["insight"]
    assert ins["fact"] == 14 and ins["plan"] == 19


def test_insight_fact_plan_none_when_metric_unknown(monkeypatch):
    calls = _fake_service(monkeypatch)
    payload = json.dumps({"insights": [
        {"type": "norm", "metric_id": "", "metric_name": "",
         "text": "Ситуация в норме."},
    ]})
    asyncio.run(make_auto_insight_node(FakeLLM(payload))(
        _state(), {"configurable": {"thread_id": "t-1"}}
    ))
    ins = calls[0]["insight"]
    assert ins["fact"] is None and ins["plan"] is None
