"""Локальный прогон doc_manager без aegra-сервера.

Подкладываем MemorySaver-чекпойнтер (на проде это делает aegra) и гоняем граф в
turn-based режиме: upload → list → delete по номеру → list. Состояние между
ходами (в т.ч. last_listed для удаления по номеру) держит чекпойнтер по
thread_id. direction_key передаём через configurable.

Требуется доступная Postgres со схемой wiki_rag (POSTGRES_DSN/DATABASE_URL) и
GIGACHAT_CREDENTIALS в .env.
"""
from __future__ import annotations

import asyncio
import os
import sys

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph_executor.aegra_agents.doc_manager.graph import build_graph


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
        if payload.get("last_listed") is not None:
            print(f"  [{node}] last_listed: {len(payload['last_listed'])} шт.")


_DOC = """Политика отпусков

Сотрудник имеет право на 28 календарных дней оплачиваемого отпуска в год.
Заявление подаётся не позднее чем за 14 дней до начала отпуска.
Перенос отпуска согласуется с непосредственным руководителем.
"""


async def main() -> int:
    graph = build_graph()
    graph.checkpointer = MemorySaver()

    config = {
        "configurable": {
            "thread_id": "doc-manager-demo-1",
            "direction_key": os.environ.get("DEMO_DIRECTION_KEY", "smoke-dir"),
        }
    }

    turns = [
        ("загрузка документа", _DOC),
        ("список", "покажи загруженные документы"),
        ("удаление по номеру", "удали 1"),
        ("список после удаления", "какие документы загружены"),
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
