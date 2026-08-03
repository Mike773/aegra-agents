"""Детерминированные тесты бизнес-поведения analytic_orchestrator_v2.

Импортируем только nodes.py (не graph.py) — GigaChat-креды/сеть не нужны.
Фейк-LLM возвращает заранее заданный .content. Асинхронные узлы гоняем через
asyncio.run. Инсайты вынесены в test_insights_v2.py.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langchain_core.messages import HumanMessage

from langgraph_executor.aegra_agents.analytic_orchestrator_v2.nodes import (
    _org_structure_block,
    make_route_node,
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


def test_org_structure_block():
    block = _org_structure_block({"metrics": _dataset(), "position": "оператор"})
    assert block is not None
    assert "Босс Борис" in block
    assert "Иванов Иван" in block
    assert "оператор" in block
    # Нет датасета → None.
    assert _org_structure_block({"metrics": None}) is None


# --- Роутер ------------------------------------------------------------------

def test_route_normal_labels():
    state = {"messages": [HumanMessage("зафиксируй этот вывод")]}
    assert make_route_node(FakeLLM("save_insight"))(state)["intent"] == "save_insight"
    assert make_route_node(FakeLLM("analytics"))(state)["intent"] == "analytics"
    assert make_route_node(FakeLLM("wiki"))(state)["intent"] == "wiki"
    # Ярлыки убранной ветки подтверждения и неизвестные ответы модели → chat
    # (безопасная деградация).
    assert make_route_node(FakeLLM("finish"))(state)["intent"] == "chat"
    assert make_route_node(FakeLLM("чтототакое"))(state)["intent"] == "chat"


def test_route_empty_reply_is_chat():
    out = make_route_node(FakeLLM("analytics"))({"messages": []})
    assert out["intent"] == "chat"
