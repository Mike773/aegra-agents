"""Опорный профиль сотрудника (employee_diagnosis) и его кеш по source_id.

Первый запуск строит профиль (детерминированный диагност + LLM-формулировки) и
кладёт его в LangGraph Store; повторный запуск с тем же source_id читает кеш без
LLM. Сбой LLM не кешируется (диалог живёт на детерминированном фолбэке), сбой
Store не фатален, без source-привязки профиль строится без кеша.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from langgraph_executor.aegra_agents.analytic_orchestrator_v4 import nodes
from langgraph_executor.aegra_agents.analytic_orchestrator_v4.employee_profile import (
    profile_namespace,
)
from langgraph_executor.aegra_agents.analytic_orchestrator_v4.nodes import (
    _profile_system_block,
    make_employee_diagnosis_node,
    make_respond_node,
)

_DATES = ["2026-05-31", "2026-06-30", "2026-07-31"]


def _dataset():
    return {
        "me": None,
        "employees": [{
            "tabnum": "111", "fio": "Иванов Иван",
            "metrics": [
                {
                    "id": "m1", "metric_name": "Продажи", "metric_type": "прямая",
                    "measure_type": "рубль", "date": d, "calc_period": "Месяц",
                    "fact": f, "plan": 100, "benchmark": None, "element": None,
                    "child_metrics": [],
                }
                for d, f in zip(_DATES, [90, 75, 60])
            ],
        }],
    }


def _aggregates():
    def _slice(dt):
        return {
            "dt": dt, "calc_period": "Месяц", "mean_fact": 100.0,
            "mean_plan": 100.0, "mean_ex": 95.0, "median": 100.0,
            "hit_rate": 70.0, "top20_mean_fact": 130.0, "iqr": 10.0,
            "cv": 20.0, "total_objects": 120,
        }

    return [{
        "dataset": {
            "level": "OFFICE", "level_name": "по офису",
            "metrics": [{
                "metric_id": "m1", "metric_name": "Продажи",
                "aggregates": {**_slice(_DATES[-1]), "history": [
                    _slice(_DATES[0]), _slice(_DATES[1]),
                ]},
            }],
        },
    }]


_PROFILE_JSON = json.dumps({
    "status": "зона риска",
    "confidence_note": "средняя (перцентили не пришли)",
    "key_facts": "факт 60 при плане 100, медиана когорты 100",
    "problems": [{
        "metric_name": "Продажи",
        "headline": "Продажи на 40% ниже медианы когорты",
        "hypothesis": "индивидуальное снижение активности",
        "what_to_check": "разовое крупное событие, отпуск",
        "meeting_topic": "как стабилизировать продажи",
        "assignment_draft": "диагностика воронки продаж",
    }],
    "achievements": [],
    "secondary": [],
}, ensure_ascii=False)


class _AIMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content=_PROFILE_JSON, fail=False):
        self.content = content
        self.fail = fail
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("LLM недоступен")
        return _AIMsg(self.content)


def _state(**over):
    state = {
        "metrics": _dataset(),
        "aggregates": _aggregates(),
        "employee_tabnum": "111",
        "boss_tabnum": "999",
        "source_type": "meeting",
        "source_id": "mtg-1",
        "reasoning_trace": [],
    }
    state.update(over)
    return state


_NS = profile_namespace("meeting", "mtg-1")


def _run(monkeypatch, store, llm, state=None, configurable=None):
    monkeypatch.setattr(nodes, "_resolve_profile_store", lambda: store)
    node = make_employee_diagnosis_node(llm)
    return asyncio.run(node(state or _state(), {"configurable": configurable or {}}))


# --- первый запуск: строим и кешируем ---------------------------------------- #

def test_first_run_builds_and_caches(monkeypatch):
    store, llm = InMemoryStore(), FakeLLM()
    out = _run(monkeypatch, store, llm)
    assert out["employee_profile"]["status"] == "зона риска"
    assert out["employee_profile_cached"] is False
    # Метрика доведена по каталогу датасета: id восстановлен по имени.
    assert out["employee_profile"]["problems"][0]["metric_id"] == "m1"
    cached = asyncio.run(store.aget(_NS, "111"))
    assert cached is not None
    assert cached.value["profile"]["status"] == "зона риска"
    assert cached.value["diagnosis"]["person"]["fio"] == "Иванов Иван"
    assert cached.value["dataset_fingerprint"]
    # Пометричную диагностику в кеше не храним — только сводку
    # (проблемы/достижения и обвязку).
    assert "metrics" not in cached.value["diagnosis"]
    assert "binary_metrics" not in cached.value["diagnosis"]
    assert cached.value["diagnosis"]["top_problem"]
    assert any("закешировал" in s.get("summary", "")
               for s in out["reasoning_trace"])


def test_second_run_reads_cache_without_llm(monkeypatch):
    store = InMemoryStore()
    _run(monkeypatch, store, FakeLLM())          # прогрев кеша
    llm2 = FakeLLM()
    out = _run(monkeypatch, store, llm2)         # свежий стейт, тот же Store
    assert out["employee_profile_cached"] is True
    assert out["employee_profile"]["status"] == "зона риска"
    assert llm2.calls == []                      # ни одного LLM-вызова
    assert any("из кеша" in s.get("summary", "") for s in out["reasoning_trace"])


def test_refresh_flag_rebuilds(monkeypatch):
    store = InMemoryStore()
    _run(monkeypatch, store, FakeLLM())
    llm2 = FakeLLM()
    out = _run(monkeypatch, store, llm2,
               configurable={"refresh_employee_diagnosis": True})
    assert out["employee_profile_cached"] is False
    assert len(llm2.calls) == 1


# --- сбои и деградация -------------------------------------------------------- #

def test_llm_failure_not_cached(monkeypatch):
    store = InMemoryStore()
    out = _run(monkeypatch, store, FakeLLM(fail=True))
    assert "employee_profile" not in out
    assert out["employee_profile_error"]
    # Диагноз есть → диалог получает детерминированный фолбэк из меток.
    assert "Опорная диагностика" in out["employee_diagnosis_brief"]
    assert "Продажи" in out["employee_diagnosis_brief"]
    # Транзиентный сбой LLM кеш не отравляет.
    assert asyncio.run(store.aget(_NS, "111")) is None


def test_garbage_llm_output_not_cached(monkeypatch):
    store = InMemoryStore()
    out = _run(monkeypatch, store, FakeLLM(content="не json"))
    assert "employee_profile" not in out
    assert out["employee_diagnosis_brief"]
    assert asyncio.run(store.aget(_NS, "111")) is None


def test_no_source_computes_but_not_cached(monkeypatch):
    store, llm = InMemoryStore(), FakeLLM()
    out = _run(monkeypatch, store, llm,
               state=_state(source_type=None, source_id=None))
    assert out["employee_profile"]["status"] == "зона риска"
    assert asyncio.run(store.aget(_NS, "111")) is None
    assert any("без кеша" in s.get("summary", "") for s in out["reasoning_trace"])


def test_no_metrics_noop(monkeypatch):
    store, llm = InMemoryStore(), FakeLLM()
    out = _run(monkeypatch, store, llm, state=_state(metrics=None))
    assert "employee_profile" not in out
    assert llm.calls == []
    assert any("данные не загружены" in s.get("summary", "")
               for s in out["reasoning_trace"])


def test_disabled_flag_is_full_noop(monkeypatch):
    store, llm = InMemoryStore(), FakeLLM()
    out = _run(monkeypatch, store, llm,
               configurable={"employee_diagnosis_enabled": False})
    assert out == {}
    assert llm.calls == []


def test_store_errors_not_fatal(monkeypatch):
    class _BrokenStore:
        async def aget(self, *a, **kw):
            raise RuntimeError("store лёг")

        async def aput(self, *a, **kw):
            raise RuntimeError("store лёг")

    out = _run(monkeypatch, _BrokenStore(), FakeLLM())
    # Кеш не работает, но профиль построен и ход не упал.
    assert out["employee_profile"]["status"] == "зона риска"


# --- подмешивание в системный контекст ---------------------------------------- #

def test_profile_block_renders_formulations():
    block = _profile_system_block({"employee_profile": json.loads(_PROFILE_JSON)})
    assert "Опорная диагностика" in block
    assert "Продажи на 40% ниже медианы когорты" in block
    assert "Черновик поручения: диагностика воронки продаж" in block
    # Фолбэк из меток — когда профиля нет, но есть brief.
    assert _profile_system_block(
        {"employee_diagnosis_brief": "метки"}) == "метки"
    assert _profile_system_block({}) is None


def test_respond_gets_profile_block(monkeypatch):
    llm = FakeLLM(content="ответ руководителю")
    node = make_respond_node(llm)
    state = {
        "messages": [HumanMessage("как дела у Иванова?")],
        "employee_profile": json.loads(_PROFILE_JSON),
    }
    asyncio.run(node(state, {"configurable": {"describe_answer": False}}))
    system_text = llm.calls[0][0].content
    assert "Опорная диагностика" in system_text
    assert "Продажи на 40% ниже медианы когорты" in system_text
