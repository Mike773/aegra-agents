"""describe_answer: раздел «Как я пришёл к выводу» — краткое саммари.

Промпт требует сжатое саммари (2–4 предложения: какими данными пользовался и как
пришёл к выводу) вместо развёрнутого пошагового рассказа. Запреты прежние:
внутренние имена инструментов/полей и сгущение красок под запретом.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from langgraph_executor.aegra_agents.analytic_orchestrator_v4.prompts import (
    DESCRIBE_ANSWER_PROMPT,
)


def test_prompt_demands_brief_summary():
    assert "2–4 предложени" in DESCRIBE_ANSWER_PROMPT
    assert "саммари" in DESCRIBE_ANSWER_PROMPT.lower()
    # Пошаговая хронология больше не нужна — и старый лимит рассказа тоже.
    assert "без пошаговой хронологии" in DESCRIBE_ANSWER_PROMPT.lower()
    assert "1200" not in DESCRIBE_ANSWER_PROMPT


def test_prompt_keeps_bans():
    # Технические имена по-прежнему под запретом (перечислены как запрещённые).
    assert "situation_overview" in DESCRIBE_ANSWER_PROMPT
    assert "hit_rate" in DESCRIBE_ANSWER_PROMPT
    # Мягкие формулировки: «критический» под запретом, рекомендаций не давать.
    assert "критический" in DESCRIBE_ANSWER_PROMPT
    assert "рекомендаций" in DESCRIBE_ANSWER_PROMPT
