"""Локальный прогон analytic_orchestrator без aegra-сервера.

Подкладываем MemorySaver-чекпоинтер (на проде это делает aegra) и гоняем граф
в turn-based режиме: каждый ход — отдельный вызов с входом {"messages": [...]},
состояние между ходами держит чекпоинтер по thread_id. Первый ход — сообщение-
инструкция (триггер анализа), дальше — обычные реплики руководителя. Граф ходит
через `astream`, потому что `call_easyrag` — async (как и сам aegra-runtime).
"""
from __future__ import annotations

import asyncio
import os
import sys

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph_executor.aegra_agents.analytic_orchestrator.graph import build_graph
from langgraph_executor.aegra_agents.shared.clients import create_gigachat_client


def _print_event(event: dict) -> None:
    for node, payload in event.items():
        if not isinstance(payload, dict):
            print(f"  [{node}] {payload!r}")
            continue
        for m in payload.get("messages") or []:
            who = getattr(m, "type", m.__class__.__name__)
            text = getattr(m, "content", "") or ""
            print(f"  [{node}] {who}: {text}")
        for key in (
            "intent",
            "metrics_error",
            "direction_key",
            "easyrag_error",
            "analytics_question",
            "analytics_error",
        ):
            if payload.get(key) is not None:
                print(f"  [{node}] {key}={payload[key]!r}")
        if payload.get("analytics_answer") is not None:
            answer = payload["analytics_answer"]
            print(f"  [{node}] analytics_answer: {len(answer)} симв.")
        snippets = payload.get("easyrag_snippets")
        if snippets is not None:
            print(f"  [{node}] easyrag_snippets: {len(snippets)} шт.")
            for s in snippets[:3]:
                sim = s.get("similarity")
                sim_str = f" sim={sim:.2f}" if isinstance(sim, (int, float)) else ""
                print(
                    f"      - {s.get('page_title')} / {s.get('section_title')}{sim_str}"
                )


async def main() -> int:
    llm = create_gigachat_client().get_llm()
    graph = build_graph(llm)
    graph.checkpointer = MemorySaver()

    config = {
        "configurable": {
            "thread_id": "demo-thread-1",
            "boss_tabnum": os.environ.get("DEMO_BOSS_TABNUM", "100001"),
            "employee_tabnum": os.environ.get("DEMO_EMP_TABNUM", "100500"),
            "position": os.environ.get("DEMO_POSITION", "Аналитик"),
        }
    }

    print("=== старт оркестратора (сообщение-инструкция) ===")
    async for ev in graph.astream(
        {"messages": [HumanMessage(content="Проанализируй метрики сотрудника.")]},
        config=config,
        stream_mode="updates",
    ):
        _print_event(ev)

    for user_text in (
        "Что в метриках выделяется сильнее всего?",
        "А как у нас вообще считается этот показатель?",
        "Спасибо, на этом всё.",
    ):
        print(f"\n=== пользователь говорит: {user_text!r} ===")
        async for ev in graph.astream(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
            stream_mode="updates",
        ):
            _print_event(ev)

    print("\n=== диалог завершён ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
