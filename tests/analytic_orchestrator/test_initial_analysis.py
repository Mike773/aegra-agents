"""Первичный анализ: разбор метрик — основное тело ответа, предложение
поручений — приписка в самом конце системного промпта.

Регресс на баг, когда INITIAL_OFFER_HINT стоял ПЕРЕД данными и перетягивал
ответ на «предложу оформить поручения» в ущерб самому разбору метрик.

Импортируем только nodes.py — без GigaChat/сети. Асинхронность гоняем через
asyncio.run (pytest-asyncio в окружении нет).
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_executor.aegra_agents.analytic_orchestrator.nodes import (
    _FINAL_KEY,
    make_initial_analysis_node,
)
from langgraph_executor.aegra_agents.analytic_orchestrator.prompts import (
    INITIAL_OFFER_HINT,
)


class _AIMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content="разбор метрик"):
        self._content = content
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _AIMsg(self._content)


class FakeAnalyzer:
    """json_analyzer-подграф: отдаёт текстовый анализ под любой вопрос."""

    def __init__(self, answer="CSAT 70% — ниже плана; тренд вниз 3 мес."):
        self._answer = answer

    async def ainvoke(self, payload):
        return {"answer": self._answer, "tool_steps": []}


def _metrics():
    return {
        "me": {"tabnum": 1, "fio": "Босс", "metrics": []},
        "employees": [{
            "tabnum": 2, "fio": "Сотрудник",
            "metrics": [{"metric_name": "CSAT", "metric_description": "опис", "fact": 70}],
        }],
    }


def _state(**over):
    base = {
        "messages": [HumanMessage(content="Проанализируй метрики сотрудника.")],
        "metrics": _metrics(),
        "briefing": "Проанализируй метрики сотрудника.",
        "direction_key": "dir-1",
        "reasoning_trace": [],
    }
    base.update(over)
    return base


def _cfg(**configurable):
    return {"configurable": configurable}


def _system_text(llm: FakeLLM) -> str:
    sys_msg = next(m for m in llm.calls[0] if isinstance(m, SystemMessage))
    return sys_msg.content


def test_analyst_data_before_offer_hint():
    """Блок данных аналитика идёт РАНЬШЕ подсказки про поручения."""
    llm = FakeLLM()
    answer = "CSAT 70% — ниже плана; тренд вниз 3 мес."
    node = make_initial_analysis_node(llm, FakeAnalyzer(answer))
    asyncio.run(node(_state(), _cfg()))

    text = _system_text(llm)
    assert answer in text
    assert INITIAL_OFFER_HINT in text
    # порядок: сначала данные, потом приписка про поручения
    assert text.index(answer) < text.index(INITIAL_OFFER_HINT)


def test_human_message_asks_analysis_first():
    """HumanMessage явно требует сперва разбор, потом — предложение поручений."""
    llm = FakeLLM()
    node = make_initial_analysis_node(llm, FakeAnalyzer())
    asyncio.run(node(_state(), _cfg()))

    human = next(m for m in llm.calls[0] if isinstance(m, HumanMessage))
    assert "Сначала дай содержательный разбор" in human.content


def test_initial_message_is_final_tagged():
    """Итоговое сообщение первого хода помечено orchestrator_final."""
    llm = FakeLLM()
    node = make_initial_analysis_node(llm, FakeAnalyzer())
    out = asyncio.run(node(_state(), _cfg()))
    msg = out["messages"][0]
    assert msg.additional_kwargs.get(_FINAL_KEY) is True


def test_initial_sets_only_summary_not_narrow_answer():
    """Ход 1 пишет разбор в metrics_summary и НЕ дублирует его в analytics_answer.

    Узкое поле analytics_answer держит только свежий ответ реального analytics-хода,
    поэтому metrics_summary и analytics_answer никогда не пересекаются по содержимому
    (и _metrics_system_block не нужен дедуп).
    """
    answer = "CSAT 70% — ниже плана; тренд вниз 3 мес."
    node = make_initial_analysis_node(FakeLLM(), FakeAnalyzer(answer))
    out = asyncio.run(node(_state(), _cfg()))
    assert out["metrics_summary"] == answer
    assert not out.get("analytics_answer")
    assert not out.get("analytics_question")
