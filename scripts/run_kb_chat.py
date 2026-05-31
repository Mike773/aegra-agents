"""Локальный прогон kb_chat без aegra-сервера.

Подкладываем MemorySaver-чекпойнтер (на проде это делает aegra) и гоняем граф в
turn-based режиме: small-talk → вопрос по базе знаний → уточнение. История диалога
между ходами держится чекпойнтером по thread_id, direction_key — через configurable.

Требуется доступная Postgres со схемой wiki_rag (POSTGRES_DSN/DATABASE_URL),
заполненная wiki_section с эмбеддингами для direction_key, и GIGACHAT_CREDENTIALS
в .env. Без наполненной базы знаний kb-вопросы получат честный ответ «не нашёл».
"""
from __future__ import annotations

import asyncio
import os
import sys

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph_executor.aegra_agents.kb_chat.graph import build_graph
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
        if payload.get("intent") is not None:
            print(f"  [{node}] intent={payload['intent']!r}")
        if payload.get("snippets") is not None:
            print(f"  [{node}] snippets: {len(payload['snippets'])} шт.")
        if payload.get("snippet_error"):
            print(f"  [{node}] snippet_error={payload['snippet_error']!r}")


async def main() -> int:
    graph = build_graph(create_gigachat_client().get_llm())
    graph.checkpointer = MemorySaver()

    config = {
        "configurable": {
            "thread_id": "kb-chat-demo-1",
            "direction_key": os.environ.get("DEMO_DIRECTION_KEY", "smoke-dir"),
        }
    }

    turns = [
        ("приветствие (small-talk)", "Привет! Чем можешь помочь?"),
        ("вопрос по базе знаний", "Сколько дней отпуска положено сотруднику?"),
        ("уточнение", "А за сколько дней подавать заявление?"),
        ("благодарность (small-talk)", "Спасибо, понятно!"),
    ]

    for label, text in turns:
        print(f"\n=== {label} ===")
        async for ev in graph.astream(
            {"messages": [HumanMessage(content=text)]},
            config=config,
            stream_mode="updates",
        ):
            _print_event(ev)

    print("\n=== готово ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
