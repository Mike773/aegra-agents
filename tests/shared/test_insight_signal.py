"""Общие помощники сигнального режима (run_mode=signal) для инсайтов."""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from langgraph_executor.aegra_agents.shared.insight_signal import (  # noqa: E402
    DEFAULT_INSIGHT_AUTHOR,
    detect_signal,
    empty_norm_insight,
    is_signal_mode,
    signal_insight,
)


class FakeLLM:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        if self.error:
            raise self.error
        return AIMessage(content=self.content)


@pytest.mark.parametrize(
    "config, expected",
    [
        (None, False),
        ({}, False),
        ({"configurable": {}}, False),
        ({"configurable": {"run_mode": None}}, False),
        ({"configurable": {"run_mode": "chat"}}, False),
        ({"configurable": {"run_mode": "signal"}}, True),
        ({"configurable": {"run_mode": " SIGNAL "}}, True),
    ],
)
def test_is_signal_mode(config, expected):
    assert is_signal_mode(config) is expected


@pytest.mark.parametrize(
    "content, expected",
    [
        ('{"signal": true}', True),
        ('{"signal": false}', False),
        ('Вот вердикт:\n```json\n{"signal": true}\n```', True),
        ('{"signal": "true"}', True),
        ("Да, проблема есть.", True),
        ("Нет, проблем не выявлено.", False),
        ("true", True),
        ("false", False),
    ],
)
def test_detect_signal_parses_verdict(content, expected):
    llm = FakeLLM(content)
    got = asyncio.run(detect_signal(llm, "Как продажи?", "Продажи ниже плана.", fallback=not expected))
    assert got is expected


def test_detect_signal_prompt_contains_question_and_answer():
    llm = FakeLLM('{"signal": false}')
    asyncio.run(detect_signal(llm, "Как звонки?", "Звонков достаточно.", fallback=True))
    human = llm.prompts[0][-1].content
    assert "Как звонки?" in human
    assert "Звонков достаточно." in human


def test_detect_signal_falls_back_on_garbage():
    llm = FakeLLM("¯\\_(ツ)_/¯")
    assert asyncio.run(detect_signal(llm, "q", "a", fallback=True)) is True
    assert asyncio.run(detect_signal(llm, "q", "a", fallback=False)) is False


def test_detect_signal_falls_back_on_llm_error():
    llm = FakeLLM(error=RuntimeError("boom"))
    assert asyncio.run(detect_signal(llm, "q", "a", fallback=True)) is True


def test_signal_insight_adds_fields_and_keeps_base():
    base = {"type": "main_problem", "metric_id": "m1", "metric_name": "Продажи", "text": "t"}
    got = signal_insight(base, signal=True, description="Ответ агента.")
    assert got == {
        **base,
        "author": DEFAULT_INSIGHT_AUTHOR,
        "confirmed": True,
        "signal": True,
        "signal_description": "Ответ агента.",
    }
    assert DEFAULT_INSIGHT_AUTHOR == "Agent"
    assert "author" not in base  # исходный словарь не мутируем


def test_empty_norm_insight_shape():
    ins = empty_norm_insight()
    assert ins["type"] == "norm"
    assert ins["metric_id"] is None and ins["metric_name"] is None
    assert ins["fact"] is None and ins["plan"] is None
    assert ins["text"]
