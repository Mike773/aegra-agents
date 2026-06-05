"""Контекст метрик для респондера и дообогащение вопроса к json_analyzer.

Регресс на баг «откуда берёт данные на 2-м+ ходе»: узкий sticky-ответ больше не
выдаётся за «все метрики» (есть стабильный опорный разбор + подписанный своим
вопросом свежий ответ), сырого обрезанного JSON в контексте нет, а вопрос к
аналитику дообогащается прошлыми вопросами-ответами для разрешения «эти/те».

Импортируем только nodes.py — чистые функции от state, без GigaChat/сети.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langchain_core.messages import AIMessage, HumanMessage

from langgraph_executor.aegra_agents.analytic_orchestrator.nodes import (
    _DIALOG_CTX_ANSWER_CAP,
    _STEP_KEY,
    _metrics_system_block,
    _question_with_dialogue_context,
)


# --- _metrics_system_block --------------------------------------------------

def test_block_composes_summary_and_labeled_answer():
    block = _metrics_system_block({
        "metrics_summary": "Широкий разбор: производительность ниже плана.",
        "analytics_answer": "Доля переводов = 8.2%.",
        "analytics_question": "Какой факт по доле переводов?",
        "briefing": "Проанализируй метрики сотрудника.",
        "metrics": {"x": 1},
    })
    # есть и опорный разбор, и свежий ответ — подписанный своим вопросом
    assert "Опорный разбор метрик" in block
    assert "Широкий разбор" in block
    assert "Ответ аналитика на вопрос «Какой факт по доле переводов?»" in block
    assert "Доля переводов = 8.2%." in block
    # сырого JSON быть не должно
    assert "JSON с метриками" not in block


def test_block_first_turn_no_duplicate_narrow():
    # На первом ходе analytics_answer == metrics_summary, вопрос == брифинг —
    # узкий блок не добавляем, чтобы не дублировать опорный разбор.
    summary = "Первичный разбор метрик сотрудника."
    block = _metrics_system_block({
        "metrics_summary": summary,
        "analytics_answer": summary,
        "analytics_question": "Проанализируй метрики сотрудника.",
        "briefing": "Проанализируй метрики сотрудника.",
        "metrics": {"x": 1},
    })
    assert "Опорный разбор метрик" in block
    assert block.count("Ответ аналитика на вопрос") == 0


def test_block_no_analysis_returns_honest_message_not_raw_json():
    block = _metrics_system_block({"metrics": {"big": "dataset"}})
    assert "разбор сейчас недоступен" in block
    assert "JSON с метриками" not in block


def test_block_metrics_error_and_none():
    assert _metrics_system_block({"metrics_error": "нет tabnum"}) == "Контекст: нет tabnum"
    assert _metrics_system_block({"metrics": None}) is None


def test_block_ignores_errored_sticky_answer():
    # Ответ с ошибкой не должен попадать как свежий блок.
    block = _metrics_system_block({
        "metrics_summary": "Опора.",
        "analytics_answer": "битый ответ",
        "analytics_question": "что-то",
        "analytics_error": "ConnectTimeout",
        "metrics": {"x": 1},
    })
    assert "битый ответ" not in block
    assert "Опора." in block


# --- _question_with_dialogue_context ----------------------------------------

def test_enrich_builds_context_and_keeps_current_question():
    state = {
        "metrics_summary": "Первичный: всё ниже плана.",
        "messages": [
            HumanMessage(content="Какой факт по производительности?"),
            AIMessage(content="Производительность = 12.05 у.е."),
            HumanMessage(content="А доля переводов?"),
            AIMessage(content="Доля переводов = 8.2%."),
            HumanMessage(content="Сделай детализацию по западающим."),  # текущий
        ],
    }
    q = _question_with_dialogue_context(state, "Сделай детализацию по западающим.")
    assert "Предыдущие взаимодействия" in q
    assert "Первичный: всё ниже плана." in q
    assert "Какой факт по производительности?" in q
    assert "Производительность = 12.05 у.е." in q
    assert "А доля переводов?" in q
    # текущий вопрос подписан и не превращён в пару
    assert "Текущий вопрос: Сделай детализацию по западающим." in q
    assert q.count("Сделай детализацию по западающим.") == 1


def test_enrich_truncates_long_answers():
    long_ans = "ц" * (_DIALOG_CTX_ANSWER_CAP + 200)
    state = {
        "messages": [
            HumanMessage(content="вопрос"),
            AIMessage(content=long_ans),
            HumanMessage(content="текущий"),
        ],
    }
    q = _question_with_dialogue_context(state, "текущий")
    assert "…" in q
    assert long_ans not in q  # целиком не попал


def test_enrich_skips_step_messages():
    state = {
        "messages": [
            HumanMessage(content="вопрос"),
            AIMessage(content="📊 шаг", additional_kwargs={_STEP_KEY: True}),
            AIMessage(content="настоящий ответ"),
            HumanMessage(content="текущий"),
        ],
    }
    q = _question_with_dialogue_context(state, "текущий")
    assert "📊 шаг" not in q
    assert "настоящий ответ" in q


def test_enrich_no_context_returns_question_as_is():
    state = {"messages": [HumanMessage(content="первый вопрос")]}
    assert _question_with_dialogue_context(state, "первый вопрос") == "первый вопрос"
