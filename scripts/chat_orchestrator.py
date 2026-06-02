"""Интерактивная CLI-болталка с analytic_orchestrator (без aegra-сервера).

Тот же turn-based паттерн, что у scripts/run_orchestrator.py (граф + MemorySaver
+ astream по ходам), но вместо захардкоженного сценария — живой цикл input().
Первая реплика пользователя уходит в граф как есть и работает триггером анализа
(см. need_load → load_data → initial_analysis в analytic_orchestrator/graph.py).

Запуск:
    python scripts/chat_orchestrator.py --employee 100500 --position Аналитик
    python scripts/chat_orchestrator.py --debug   # полный служебный поток
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph_executor.aegra_agents.analytic_orchestrator.graph import build_graph
from langgraph_executor.aegra_agents.shared.clients import create_gigachat_client

EXIT_WORDS = {"exit", "quit", ":q", "выход"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Интерактивный чат с analytic_orchestrator.",
    )
    p.add_argument(
        "--employee",
        default=os.environ.get("DEMO_EMP_TABNUM", "100500"),
        help="Табельный номер сотрудника (env DEMO_EMP_TABNUM).",
    )
    p.add_argument(
        "--boss",
        default=os.environ.get("DEMO_BOSS_TABNUM", "100001"),
        help="Табельный номер руководителя (env DEMO_BOSS_TABNUM).",
    )
    p.add_argument(
        "--position",
        default=os.environ.get("DEMO_POSITION", "Аналитик"),
        help="Должность сотрудника (env DEMO_POSITION).",
    )
    p.add_argument(
        "--thread",
        default="chat-thread-1",
        help="Идентификатор треда чекпоинтера (по умолчанию chat-thread-1).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Печатать полный служебный поток по всем узлам графа.",
    )
    return p.parse_args()


def _format_status(payload: dict) -> str | None:
    """Краткая строка-статус хода: intent и источники/ошибки, если есть."""
    parts: list[str] = []
    intent = payload.get("intent")
    if intent is not None:
        parts.append(f"intent={intent}")
    snippets = payload.get("easyrag_snippets")
    if snippets:
        parts.append(f"wiki={len(snippets)} сниппет(ов)")
    if payload.get("analytics_answer") is not None:
        parts.append("json_analyzer: есть ответ")
    for key in ("metrics_error", "analytics_error", "easyrag_error"):
        if payload.get(key) is not None:
            parts.append(f"{key}={payload[key]!r}")
    return " · ".join(parts) if parts else None


def _print_debug_event(event: dict) -> None:
    """Полный дамп узлов (логика из scripts/run_orchestrator.py)."""
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
            print(f"  [{node}] analytics_answer: {len(payload['analytics_answer'])} симв.")
        snippets = payload.get("easyrag_snippets")
        if snippets is not None:
            print(f"  [{node}] easyrag_snippets: {len(snippets)} шт.")
            for s in snippets[:3]:
                sim = s.get("similarity")
                sim_str = f" sim={sim:.2f}" if isinstance(sim, (int, float)) else ""
                print(f"      - {s.get('page_title')} / {s.get('section_title')}{sim_str}")


def _print_turn(event: dict, debug: bool) -> None:
    """Печать одного апдейта графа: реплики ассистента + краткий статус."""
    if debug:
        _print_debug_event(event)
        return
    for node, payload in event.items():
        if not isinstance(payload, dict):
            continue
        for m in payload.get("messages") or []:
            if getattr(m, "type", "") != "human":
                text = getattr(m, "content", "") or ""
                if text:
                    print(f"\nАссистент: {text}")
        status = _format_status(payload)
        if status:
            print(f"  ⟂ {status}")


async def main() -> int:
    args = parse_args()

    llm = create_gigachat_client().get_llm()
    graph = build_graph(llm)
    graph.checkpointer = MemorySaver()

    config = {
        "configurable": {
            "thread_id": args.thread,
            "boss_tabnum": args.boss,
            "employee_tabnum": args.employee,
            "position": args.position,
        }
    }

    print("=== Болталка с analytic_orchestrator ===")
    print(f"Сотрудник: {args.employee} · руководитель: {args.boss} · должность: {args.position}")
    print("Первое сообщение запускает анализ метрик. Выход: exit / quit / Ctrl-D.\n")

    while True:
        try:
            user_text = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n=== диалог завершён ===")
            return 0

        if not user_text or user_text.lower() in EXIT_WORDS:
            print("=== диалог завершён ===")
            return 0

        try:
            async for ev in graph.astream(
                {"messages": [HumanMessage(content=user_text)]},
                config=config,
                stream_mode="updates",
            ):
                _print_turn(ev, args.debug)
        except Exception as exc:  # noqa: BLE001 — болталка не должна падать на одном ходе
            print(f"  [ошибка хода] {exc!r}")
        print()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
