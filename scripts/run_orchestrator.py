"""Локальный прогон analytic_orchestrator без aegra-сервера.

Подкладываем MemorySaver-чекпоинтер (на проде это делает aegra), идём по графу
до первого `interrupt`, передаём «реплику руководителя» через `Command(resume=...)`
и повторяем, пока модель не пометит intent как `done`. Граф ходит через
`astream`, потому что `call_easyrag` — async (как и сам aegra-runtime).
"""
from __future__ import annotations

import asyncio
import os
import sys

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgraph_executor.aegra_agents.analytic_orchestrator.graph import build_graph
from langgraph_executor.aegra_agents.shared.clients import create_gigachat_client


def _print_event(event: dict) -> None:
    for node, payload in event.items():
        if node == "__interrupt__":
            for itr in payload or ():
                value = getattr(itr, "value", itr)
                print(f"  [interrupt] {value}")
            continue
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

    print("=== старт оркестратора ===")
    async for ev in graph.astream(
        {"messages": []}, config=config, stream_mode="updates"
    ):
        _print_event(ev)

    state = graph.get_state(config)
    user_lines = iter([
        "Что в метриках выделяется сильнее всего?",
        "А как у нас вообще считается этот показатель?",
        "Спасибо, на этом всё.",
    ])
    while state.next:
        try:
            user_text = next(user_lines)
        except StopIteration:
            print("\n>>> сценарий закончился, а граф ещё ждёт ввод. Прерываю.")
            return 1
        print(f"\n=== пользователь говорит: {user_text!r} ===")
        async for ev in graph.astream(
            Command(resume=user_text), config=config, stream_mode="updates"
        ):
            _print_event(ev)
        state = graph.get_state(config)

    print("\n=== оркестратор завершил работу ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
