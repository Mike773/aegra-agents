"""Форма финального ответа ПЕРВОГО хода: рассказ + главная проблема + продолжение.

После основного LLM-ответа initial_analysis делает второй вызов с
FIRST_ANSWER_FORMAT_PROMPT: черновик приводится к рассказу, если пользователь не
задал формат в запросе (это решает сам промпт — узел зовёт его всегда). Сбой или
пустой результат не фатальны — остаёмся на черновике. Ходы 2+ не затронуты.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_executor.aegra_agents.analytic_orchestrator_v4.nodes import (
    make_initial_analysis_node,
)
from langgraph_executor.aegra_agents.analytic_orchestrator_v4.prompts import (
    FIRST_ANSWER_FORMAT_PROMPT,
)

_DRAFT = "Черновик: план 100, факт 60."
_STORY = "Рассказ: главная проблема — продажи, факт 60 при плане 100. Есть и другие зоны."


class _AIMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """1-й вызов — черновик ответа; вызов с промптом формы — рассказ."""

    def __init__(self, story=_STORY, fail_format=False):
        self.story = story
        self.fail_format = fail_format
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        system = next(
            (m.content for m in messages if isinstance(m, SystemMessage)), ""
        )
        if system == FIRST_ANSWER_FORMAT_PROMPT:
            if self.fail_format:
                raise RuntimeError("LLM недоступен")
            return _AIMsg(self.story)
        return _AIMsg(_DRAFT)


class _FakeAnalyzerGraph:
    async def ainvoke(self, state, config=None):
        return {"answer": "ГЛАВНАЯ ЗОНА: продажи", "tool_steps": []}


def _state():
    return {
        "messages": [HumanMessage(content="разбери метрики")],
        "metrics": {"me": None, "employees": [{"fio": "Иванов Иван", "metrics": []}]},
        "direction_key": "оператор",
        "briefing": "разбери метрики",
    }


def _run(llm, configurable=None):
    node = make_initial_analysis_node(llm, _FakeAnalyzerGraph())
    return asyncio.run(node(_state(), {"configurable": configurable or {}}))


def _final_text(out):
    return out["messages"][-1].content


def _format_calls(llm):
    return [
        msgs for msgs in llm.calls
        if any(isinstance(m, SystemMessage)
               and m.content == FIRST_ANSWER_FORMAT_PROMPT for m in msgs)
    ]


def test_default_reformats_to_story():
    llm = FakeLLM()
    out = _run(llm, {"answer_html": False})
    assert _final_text(out) == _STORY
    fmt = _format_calls(llm)
    assert len(fmt) == 1
    # На вход редактору идут запрос руководителя и черновик.
    human = next(m.content for m in fmt[0] if isinstance(m, HumanMessage))
    assert "разбери метрики" in human
    assert _DRAFT in human
    # Правило детекции формата — в самом промпте, отдельного вызова нет.
    assert "верни черновой ответ ДОСЛОВНО" in FIRST_ANSWER_FORMAT_PROMPT
    assert any("к форме рассказа" in s.get("summary", "")
               for s in out["reasoning_trace"])


def test_flag_off_keeps_draft_single_call():
    llm = FakeLLM()
    out = _run(llm, {"answer_html": False, "format_first_answer": False})
    assert _final_text(out) == _DRAFT
    assert _format_calls(llm) == []


def test_format_llm_failure_falls_back_to_draft():
    llm = FakeLLM(fail_format=True)
    out = _run(llm, {"answer_html": False})
    assert _final_text(out) == _DRAFT
    assert any(s.get("kind") == "error"
               and "форме рассказа" in s.get("summary", "")
               for s in out["reasoning_trace"])


def test_empty_format_output_falls_back_to_draft():
    llm = FakeLLM(story="   ")
    out = _run(llm, {"answer_html": False})
    assert _final_text(out) == _DRAFT


def test_html_history_keeps_formatted_markdown():
    # answer_html по умолчанию: в истории (orchestrator_markdown) — рассказ.
    out = _run(FakeLLM())
    msg = out["messages"][-1]
    assert msg.additional_kwargs["orchestrator_markdown"] == _STORY
    assert "<p>" in msg.content
