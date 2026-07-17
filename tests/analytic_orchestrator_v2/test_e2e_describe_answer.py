"""E2E режима describe_answer: полный граф analytic_orchestrator_v2 in-process.

Без сети, кредов и БД: LLM/эмбеддинги подменяет mock_llm (MockGigaChat), датасет —
mock_llm._install_full_dataset (непустой, чтобы json_analyzer прошёл реальный
tool-loop: мок на стадии сбора один раз зовёт schema_overview с мотивацией).
Wiki-grounding и easyrag отключены конфигом (не требуют Postgres).

mock_llm.install() подменяет фабрики клиентов ДО импорта графов — поэтому импорт
graph.py здесь, после install(), а не в шапке.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

os.environ.setdefault("MOCK_LLM_LATENCY", "0")

import langgraph_executor.mock_llm as mock_llm

mock_llm.install(full_path=True)

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from langgraph_executor.aegra_agents.analytic_orchestrator_v2.graph import (  # noqa: E402
    build_graph,
)
from langgraph_executor.aegra_agents.analytic_orchestrator_v2.nodes import (  # noqa: E402
    _FINAL_KEY,
    _TRACE_SECTION_TITLE,
)

_GRAPH = build_graph(mock_llm.MockGigaChat(credentials="mock", verify_ssl_certs=False),
                     checkpointer=MemorySaver())


def _config(thread_id: str, describe_answer: bool | None):
    cfg = {
        "boss_tabnum": "100001",
        "employee_tabnum": "100500",
        "position": "Аналитик",
        "thread_id": thread_id,
        "easyrag_enabled": False,
        "wiki_grounding_enabled": False,
        "emit_progress_messages": False,
    }
    if describe_answer is not None:
        cfg["describe_answer"] = describe_answer
    return {"configurable": cfg}


def _final_text(result: dict) -> str:
    finals = [
        m for m in result["messages"]
        if getattr(m, "additional_kwargs", None) and m.additional_kwargs.get(_FINAL_KEY)
    ]
    assert finals, "в ходе нет итогового сообщения (orchestrator_final)"
    return finals[-1].content


def _invoke(message: str, thread_id: str, describe_answer: bool | None) -> str:
    result = asyncio.run(_GRAPH.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        config=_config(thread_id, describe_answer),
    ))
    return _final_text(result)


def test_e2e_first_turn_with_describe_answer():
    text = _invoke("Что происходит с показателями сотрудника?", "e2e-on", True)
    # Основной ответ + блок описания, сгенерированный LLM-путём (текст мока),
    # а не механическим render_trace («1. **…**»).
    assert _TRACE_SECTION_TITLE in text
    assert mock_llm.MOCK_DESCRIBE_NARRATIVE in text
    body, block = text.split(_TRACE_SECTION_TITLE, 1)
    assert body.strip(), "основной ответ пуст"
    assert "1. **" not in block


def test_e2e_first_turn_without_describe_answer():
    text = _invoke("Что происходит с показателями сотрудника?", "e2e-off", None)
    assert _TRACE_SECTION_TITLE not in text


def test_e2e_second_turn_analytics_with_describe_answer():
    thread = "e2e-two-turns"
    _invoke("Что происходит с показателями сотрудника?", thread, True)
    # Второй ход: роутер мока классифицирует как analytics → call_json_analyzer
    # (реальный tool-loop с schema_overview) → respond. Блок дописан и здесь.
    text = _invoke("Почему просела выручка?", thread, True)
    assert _TRACE_SECTION_TITLE in text
    assert mock_llm.MOCK_DESCRIBE_NARRATIVE in text
