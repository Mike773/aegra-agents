"""analytic_orchestrator_v4: блок правил про звезду подмешивается по данным.

Прозу пишет оркестратор, поэтому правила «как говорить про звезду» нужны именно
ему. Признак наличия звезды берём из СЫРОГО датасета (state['metrics']), а не из
текста выжимки: подпись раздела пишет модель и может её перефразировать.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_executor.aegra_agents.analytic_orchestrator_v4.nodes import (
    _has_star_data,
    make_initial_analysis_node,
    make_respond_node,
)
from langgraph_executor.aegra_agents.analytic_orchestrator_v4.prompts import (
    STAR_PROSE_BLOCK,
)


def _dataset(*metrics):
    return {"me": None, "employees": [{"fio": "Иванов Иван", "metrics": list(metrics)}]}


PLAIN = {"id": "1", "metric_name": "Производительность", "fact": 12}
# Звезда — узел со star_received, её влияющие показатели — в child_metrics.
STAR = {"id": "2", "metric_name": "Звезда качества", "star_received": False,
        "child_metrics": [{"id": "3", "metric_name": "CSI", "fact": 4.1,
                           "plan": 4.5, "is_star_metric": True}]}
# Флаг без узла звезды — вторичный признак: правила про звезду всё равно нужны.
STAR_NUMERIC = {"id": "4", "metric_name": "Доля переводов", "fact": 14,
                "is_star_metric": True}


# --- Детектор ---------------------------------------------------------------

def test_has_star_data_detects_star_node():
    assert _has_star_data(_dataset(PLAIN, STAR))


def test_star_prose_block_semantics():
    assert "Звёзд может быть несколько" in STAR_PROSE_BLOCK
    assert "называй каждую по имени" in STAR_PROSE_BLOCK
    assert "дочерние показатели" in STAR_PROSE_BLOCK


def test_has_star_data_detects_numeric_star_metric():
    assert _has_star_data(_dataset(PLAIN, STAR_NUMERIC))


def test_has_star_data_false_on_plain_dataset():
    assert not _has_star_data(_dataset(PLAIN))


def test_has_star_data_on_broken_input_is_false():
    """Датасет приходит от внешнего клиента — форма не гарантирована."""
    for broken in (None, "не json", {"employees": 5}, []):
        assert not _has_star_data(broken)


# --- Подмешивание в системный промпт респондера -----------------------------

class _AIMsg:
    def __init__(self, content):
        self.content = content


class CapturingLLM:
    """Ловит системный промпт, с которым позвали модель."""

    def __init__(self):
        self.system_text = ""

    def invoke(self, messages):
        for m in messages:
            if isinstance(m, SystemMessage):
                self.system_text = m.content
        return _AIMsg("ответ")


def _respond_system_text(metrics):
    llm = CapturingLLM()
    state = {
        "messages": [HumanMessage(content="что по звезде?")],
        "metrics": metrics,
        "metrics_summary": "ЗВЕЗДА: не заработана",
    }
    asyncio.run(make_respond_node(llm)(state, {}))
    return llm.system_text


def test_respond_injects_star_block_with_star_data():
    assert STAR_PROSE_BLOCK in _respond_system_text(_dataset(PLAIN, STAR))


def test_respond_omits_star_block_without_star_data():
    assert STAR_PROSE_BLOCK not in _respond_system_text(_dataset(PLAIN))


# --- Подмешивание на первом ходе --------------------------------------------

class _FakeAnalyzerGraph:
    """Подграф json_analyzer: отдаёт готовую выжимку без вызовов модели."""

    async def ainvoke(self, state, config=None):
        return {"answer": "ЗВЕЗДА: не заработана", "tool_steps": []}


def _initial_system_text(metrics):
    llm = CapturingLLM()
    node = make_initial_analysis_node(llm, _FakeAnalyzerGraph())
    state = {
        "messages": [HumanMessage(content="разбери метрики")],
        "metrics": metrics,
        "direction_key": "оператор",
        "briefing": "разбери метрики",
    }
    # format_first_answer выключен: тест целится в системный промпт ОТВЕТА, а
    # CapturingLLM запоминает последний SystemMessage (им стал бы промпт формы).
    asyncio.run(node(state, {"configurable": {"format_first_answer": False}}))
    return llm.system_text


def test_initial_analysis_injects_star_block_with_star_data():
    assert STAR_PROSE_BLOCK in _initial_system_text(_dataset(PLAIN, STAR))


def test_initial_analysis_omits_star_block_without_star_data():
    assert STAR_PROSE_BLOCK not in _initial_system_text(_dataset(PLAIN))
