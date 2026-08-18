"""Флаг configurable.use_peer_aggregates и роутер analytic_orchestrator_v4.

Флаг управляет ВСЕМИ peer-данными: batch-агрегатами уровней (их не запрашиваем в
load_data) и персональными rankings в основном датасете (их срезаем на входе в
аналитик, не мутируя metrics в стейте).
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

from langgraph_executor.aegra_agents.analytic_orchestrator_v4 import nodes
from langgraph_executor.aegra_agents.analytic_orchestrator_v4.nodes import (
    _collect_aggregate_ids,
    _strip_rankings,
    make_load_data_node,
    make_route_node,
)

RANKINGS = [{"rank": "458 из 500", "level": "ORG", "percentile": 8.4}]


class _AIMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, messages):
        return _AIMsg(self._content)


def _dataset_with_rankings():
    return {
        "me": None,
        "employees": [{
            "fio": "Иванов Иван",
            "metrics": [{
                "id": "1", "metric_name": "Производительность", "fact": 12,
                "rankings": RANKINGS,
                "child_metrics": [
                    {"id": "2", "metric_name": "Доля переводов", "fact": 14,
                     "rankings": RANKINGS},
                ],
            }],
        }],
    }


# --- Срезка персональных rankings ------------------------------------------

def test_strip_rankings_is_deep_and_non_mutating():
    data = _dataset_with_rankings()
    stripped = _strip_rankings(data)
    emp = stripped["employees"][0]["metrics"][0]
    assert "rankings" not in emp
    assert "rankings" not in emp["child_metrics"][0]
    # Метрики (имя/факт) на месте, исходный объект не тронут.
    assert emp["metric_name"] == "Производительность"
    assert data["employees"][0]["metrics"][0]["rankings"] == RANKINGS


def test_strip_rankings_handles_missing_me():
    assert _strip_rankings({"employees": []})["employees"] == []
    assert _strip_rankings("не словарь") == "не словарь"


# --- load_data: индивидуальная загрузка предагрегатов -----------------------
# Какие грузить, говорит сам датасет: aggregates_ids у персоны рядом с metrics.
# Один вызов на уникальный id (у босса и подчинённого id могут совпадать).

class _FakeOrg:
    def __init__(self, **kw):
        pass

    def direction_key(self):
        return "dir-1"

    def combined_json(self):
        return {}


def _dataset_with_ids():
    return {
        "me": {"fio": "Босс Борис", "aggregates_ids": ["182_m_777"], "metrics": []},
        "employees": [{
            "fio": "Иванов Иван",
            # Первый id общий с руководителем — вызов не должен дублироваться.
            "aggregates_ids": ["182_m_777", "182_m_888"],
            "metrics": [],
        }],
    }


def _agg_record(level="ORG"):
    return {"dataset": {"level": level, "metrics": []}}


def _patch_sources(monkeypatch, agg_calls, dataset, fail_ids=()):
    monkeypatch.setattr(nodes, "IsuEmployeeOrgstructureInfo", _FakeOrg)

    class _Dataset:
        def __init__(self, **kw):
            pass

        def build_json_output(self):
            return dataset

    class _Agg:
        def __init__(self, **kw):
            agg_calls.append(kw)
            self.kw = kw

        def build_json_output(self):
            object_id = self.kw["filters"][0]["object_id"]
            if object_id in fail_ids:
                raise RuntimeError("сервис агрегатов лёг")
            return [_agg_record()]

    monkeypatch.setattr(nodes, "GetBatchAgentDatasetByFiltersComponent", _Dataset)
    monkeypatch.setattr(nodes, "GetBatchAgentAggregateDatasetByFiltersComponent", _Agg)


def _load(monkeypatch, configurable, dataset=None, fail_ids=()):
    agg_calls: list[dict] = []
    _patch_sources(
        monkeypatch, agg_calls,
        dataset if dataset is not None else _dataset_with_ids(),
        fail_ids,
    )
    node = make_load_data_node()
    state = {"messages": [HumanMessage("разбери метрики")]}
    out = asyncio.run(node(state, {"configurable": configurable}))
    return out, agg_calls


_BASE_CFG = {"boss_tabnum": "999", "employee_tabnum": "12345"}


def test_one_call_per_unique_id_with_id_filters(monkeypatch):
    out, agg_calls = _load(monkeypatch, dict(_BASE_CFG))
    # Дедуп по персонам: 182_m_777 есть у обоих, но вызов один; порядок —
    # первого появления (me раньше employees).
    assert [c["filters"] for c in agg_calls] == [
        [{"object_type": "aggregate_id", "object_id": "182_m_777"}],
        [{"object_type": "aggregate_id", "object_id": "182_m_888"}],
    ]
    # Каждая запись помечена своим aggregate_id — по пометке аналитик
    # привязывает агрегаты к персонам.
    assert out["aggregates"] == [
        {**_agg_record(), "aggregate_id": "182_m_777"},
        {**_agg_record(), "aggregate_id": "182_m_888"},
    ]
    assert out["aggregates_error"] is None


def test_no_aggregate_ids_means_no_calls(monkeypatch):
    out, agg_calls = _load(
        monkeypatch, dict(_BASE_CFG), dataset={"me": None, "employees": []}
    )
    assert agg_calls == []
    assert out["aggregates"] is None


def test_ids_collected_from_json_string_dataset(monkeypatch):
    # Прод может отдать датасет JSON-строкой — id всё равно собираются.
    out, agg_calls = _load(
        monkeypatch, dict(_BASE_CFG),
        dataset=json.dumps(_dataset_with_ids(), ensure_ascii=False),
    )
    assert len(agg_calls) == 2
    assert [r["aggregate_id"] for r in out["aggregates"]] == [
        "182_m_777", "182_m_888",
    ]


def test_failed_id_does_not_drop_the_rest(monkeypatch):
    out, agg_calls = _load(monkeypatch, dict(_BASE_CFG), fail_ids={"182_m_777"})
    assert len(agg_calls) == 2
    assert out["aggregates"] == [{**_agg_record(), "aggregate_id": "182_m_888"}]
    assert "182_m_777" in out["aggregates_error"]


def test_aggregates_skipped_when_flag_off(monkeypatch):
    out, agg_calls = _load(
        monkeypatch, {**_BASE_CFG, "use_peer_aggregates": False}
    )
    assert agg_calls == []
    assert out["aggregates"] is None


def test_flag_accepts_string_false(monkeypatch):
    _, agg_calls = _load(monkeypatch, {**_BASE_CFG, "use_peer_aggregates": "false"})
    assert agg_calls == []


def test_collect_aggregate_ids_direct():
    assert _collect_aggregate_ids(_dataset_with_ids()) == ["182_m_777", "182_m_888"]
    # Не-dict персоны, пустые/None id, не-dict датасет — молча пропускаются.
    assert _collect_aggregate_ids({
        "me": "не словарь",
        "employees": [{"aggregates_ids": [None, " ", "182_m_777", 42]}],
    }) == ["182_m_777", "42"]
    assert _collect_aggregate_ids(None) == []
    assert _collect_aggregate_ids("битая строка") == []


def test_source_params_land_in_state(monkeypatch):
    out, _ = _load(
        monkeypatch,
        {**_BASE_CFG, "source_type": "meeting", "source_id": "mtg-77"},
    )
    assert out["source_type"] == "meeting"
    assert out["source_id"] == "mtg-77"


def test_source_params_absent_are_none(monkeypatch):
    out, _ = _load(monkeypatch, dict(_BASE_CFG))
    assert out["source_type"] is None
    assert out["source_id"] is None


# --- Роутер -----------------------------------------------------------------

def test_route_known_labels():
    state = {"messages": [HumanMessage("разбери долю переводов")]}
    assert make_route_node(FakeLLM("analytics"))(state)["intent"] == "analytics"
    assert make_route_node(FakeLLM("save_insight"))(state)["intent"] == "save_insight"
    assert make_route_node(FakeLLM("wiki"))(state)["intent"] == "wiki"


def test_route_removed_labels_degrade_to_chat():
    """Интентов подтверждения и «2/3» больше нет — они не должны проходить."""
    state = {"messages": [HumanMessage("1")]}
    for stale in ("finish", "more_analysis", "ask_question", "finish_save"):
        assert make_route_node(FakeLLM(stale))(state)["intent"] == "chat"
