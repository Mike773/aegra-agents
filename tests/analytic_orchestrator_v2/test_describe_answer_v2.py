"""Юнит-тесты режима describe_answer (LLM-описание трассы + фоллбэк).

Импортируем только nodes.py/agent_base.py — GigaChat-креды/сеть не нужны.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langchain_core.messages import AIMessage, ToolMessage

from langgraph_executor.aegra_agents.analytic_orchestrator_v2.nodes import (
    _TRACE_SECTION_TITLE,
    _with_description,
    render_trace,
)
from langgraph_executor.aegra_agents.json_analyzer_v3.agent_base import (
    extract_tool_steps,
)


class _AIMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content, error: Exception | None = None):
        self._content = content
        self._error = error
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self._error is not None:
            raise self._error
        return _AIMsg(self._content)


_CFG_ON = {"configurable": {"describe_answer": True}}


def _trace_with_tool_call():
    return [
        {"stage": "route", "kind": "intent",
         "summary": "Определил тип запроса как «analytics»."},
        {"stage": "json_analyzer", "kind": "tool_call",
         "summary": "get_metric(name=Производительность) → факт 12, план 18",
         "detail": {"tool": "get_metric",
                    "args": {"name": "Производительность"},
                    "result_summary": "факт 12, план 18",
                    "reasoning": "Нужно текущее значение метрики."}},
        {"stage": "respond", "kind": "decision",
         "summary": "Сформировал ответ на основе: метрики."},
    ]


def test_flag_off_no_block():
    text = asyncio.run(_with_description(
        "Ответ.", {"reasoning_trace": _trace_with_tool_call()}, {},
        llm=FakeLLM("не должно вызваться"),
    ))
    assert text == "Ответ."


def test_llm_narrative_appended():
    llm = FakeLLM("Я запросил метрику и сравнил её с планом.")
    text = asyncio.run(_with_description(
        "Ответ.", {"reasoning_trace": _trace_with_tool_call()}, _CFG_ON,
        llm=llm, question="что происходит?",
    ))
    assert text.startswith("Ответ.")
    assert _TRACE_SECTION_TITLE in text
    assert "Я запросил метрику и сравнил её с планом." in text
    # Использован LLM-путь, а не механический список render_trace.
    assert "1. **" not in text
    # В контекст LLM ушли вопрос, ответ, шаги трассы и мотивация вызова.
    assert len(llm.calls) == 1
    sent = "\n".join(m.content for m in llm.calls[0])
    assert "что происходит?" in sent
    assert "get_metric" in sent
    assert "Мотивация вызова: Нужно текущее значение метрики." in sent


def test_llm_error_falls_back_to_render_trace():
    trace = _trace_with_tool_call()
    text = asyncio.run(_with_description(
        "Ответ.", {"reasoning_trace": trace}, _CFG_ON,
        llm=FakeLLM("", error=RuntimeError("boom")),
    ))
    assert _TRACE_SECTION_TITLE in text
    assert render_trace(trace) in text


def test_trivial_trace_uses_deterministic_render_without_llm():
    # Служебный ход (отмена сохранения): нет tool_call/kb_hit → LLM не дёргаем.
    trace = [
        {"stage": "route", "kind": "intent", "summary": "Подтверждение: cancel."},
        {"stage": "assignments", "kind": "decision",
         "summary": "Руководитель отменил сохранение — ничего не фиксирую."},
    ]
    llm = FakeLLM("не должно вызваться")
    text = asyncio.run(_with_description(
        "Хорошо, не сохраняю.", {"reasoning_trace": trace}, _CFG_ON, llm=llm,
    ))
    assert llm.calls == []
    assert render_trace(trace) in text


def test_no_llm_passed_falls_back():
    trace = _trace_with_tool_call()
    text = asyncio.run(_with_description(
        "Ответ.", {"reasoning_trace": trace}, _CFG_ON,
    ))
    assert render_trace(trace) in text


# --- Захват «почему» в json_analyzer_v3.extract_tool_steps -------------------

def _gather_messages(with_reasoning: bool):
    ai = AIMessage(
        content="Посмотрю значение метрики." if with_reasoning else "",
        tool_calls=[{"name": "get_metric",
                     "args": {"name": "Производительность", "element": None},
                     "id": "c1", "type": "tool_call"}],
    )
    return [ai, ToolMessage(content="факт 12, план 18", tool_call_id="c1")]


def test_extract_tool_steps_captures_reasoning():
    steps = extract_tool_steps(_gather_messages(with_reasoning=True))
    assert len(steps) == 1
    assert steps[0]["tool"] == "get_metric"
    assert steps[0]["args"] == {"name": "Производительность"}  # None отфильтрован
    assert steps[0]["result_summary"] == "факт 12, план 18"
    assert steps[0]["reasoning"] == "Посмотрю значение метрики."


def test_extract_tool_steps_without_reasoning_has_no_field():
    steps = extract_tool_steps(_gather_messages(with_reasoning=False))
    assert len(steps) == 1
    assert "reasoning" not in steps[0]
