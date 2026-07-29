"""Что уходит в подграф json_analyzer_v5 из analytic_orchestrator_v4.

Предагрегаты передаются ключом raw_aggregates; при выключенном флаге в аналитик
не уходят ни они, ни персональные rankings. Ошибки подграфа изолированы.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langgraph_executor.aegra_agents.analytic_orchestrator_v4.nodes import _run_analyzer

AGGS = [{"dataset": {"level": "ORG", "metrics": []}}]
METRICS = {
    "me": None,
    "employees": [{
        "fio": "Иванов Иван",
        "metrics": [{
            "id": "1", "metric_name": "Производительность", "fact": 12,
            "rankings": [{"rank": "458 из 500", "level": "ORG", "percentile": 8.4}],
        }],
    }],
}


class FakeGraph:
    def __init__(self, result=None, boom=None):
        self.result = result if result is not None else {"answer": "ок", "tool_steps": []}
        self.boom = boom
        self.payload = None

    async def ainvoke(self, payload):
        self.payload = payload
        if self.boom:
            raise self.boom
        return self.result


def _run(graph, **kw):
    return asyncio.run(
        _run_analyzer(graph, METRICS, "что происходит?", "dir-1", **kw)
    )


def test_aggregates_passed_through():
    graph = FakeGraph()
    answer, steps, err = _run(graph, aggregates=AGGS)
    assert (answer, err) == ("ок", None)
    assert graph.payload["raw_aggregates"] == AGGS
    # rankings сотрудника остались на месте.
    assert graph.payload["raw_json"]["employees"][0]["metrics"][0]["rankings"]


def test_flag_off_strips_all_peer_data():
    graph = FakeGraph()
    _run(graph, aggregates=AGGS, use_aggregates=False)
    assert graph.payload["raw_aggregates"] is None
    assert "rankings" not in graph.payload["raw_json"]["employees"][0]["metrics"][0]


def test_missing_aggregates_are_fine():
    graph = FakeGraph()
    answer, _, err = _run(graph)
    assert graph.payload["raw_aggregates"] is None
    assert (answer, err) == ("ок", None)


def test_subgraph_failure_is_isolated():
    graph = FakeGraph(boom=RuntimeError("подграф лёг"))
    answer, steps, err = _run(graph, aggregates=AGGS)
    assert answer is None
    assert steps == []
    assert "подграф лёг" in err
