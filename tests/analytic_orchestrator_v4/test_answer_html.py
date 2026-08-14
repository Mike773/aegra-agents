"""answer_html: итоговый ответ конвертируется в HTML детерминированно
(markdown2, без LLM), по умолчанию включено; история для LLM — из исходного
markdown, чтобы модель не имитировала разметку.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from langgraph_executor.aegra_agents.analytic_orchestrator_v4.nodes import (
    _MARKDOWN_KEY,
    _final_message,
    _history_for_llm,
)

_MD = "### Итог\n\n**Производительность** ниже плана.\n\n---\n### Как я пришёл к выводу\n\n1. **Метрики.** Шаг."


def test_final_message_html_by_default():
    msg = _final_message("**Жирный** вывод.", None)
    assert "<strong>Жирный</strong>" in msg.content
    assert "**" not in msg.content
    # Исходный markdown сохранён для истории LLM, флаг итога на месте.
    assert msg.additional_kwargs[_MARKDOWN_KEY] == "**Жирный** вывод."
    assert msg.additional_kwargs["orchestrator_final"] is True


def test_final_message_headers_and_lists():
    msg = _final_message(_MD, {"configurable": {}})
    assert "<h3>Итог</h3>" in msg.content
    assert "<hr" in msg.content
    assert "<ol>" in msg.content


def test_final_message_markdown_when_disabled():
    msg = _final_message("**Жирный** вывод.", {"configurable": {"answer_html": False}})
    assert msg.content == "**Жирный** вывод."
    assert _MARKDOWN_KEY not in msg.additional_kwargs


def test_history_uses_markdown_and_strips_trace():
    msg = _final_message(_MD, None)
    (hist,) = _history_for_llm([msg])
    # В LLM уходит исходный markdown без HTML и без раздела трассы.
    assert "<h3>" not in hist.content
    assert hist.content == "### Итог\n\n**Производительность** ниже плана."


def test_history_plain_message_unchanged():
    msg = _final_message("Просто текст.", {"configurable": {"answer_html": False}})
    (hist,) = _history_for_llm([msg])
    assert hist.content == "Просто текст."
