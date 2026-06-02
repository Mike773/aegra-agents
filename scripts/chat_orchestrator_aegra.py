"""Интерактивная CLI-болталка к ЗАПУЩЕННОЙ aegra (по HTTP), а не in-process.

В отличие от scripts/chat_orchestrator.py (который строит граф локально с
MemorySaver), эта болталка ходит в реальный aegra-сервер через langgraph_sdk:
создаёт thread, шлёт ходы в граф `analytic_orchestrator` и стримит апдейты.
Состояние диалога держит сам сервер по thread_id (его чекпоинтер).

Сервер должен быть поднят (docker-compose / uvicorn langgraph_executor.plugins.app:app),
по умолчанию на http://localhost:8000.

Запуск:
    python scripts/chat_orchestrator_aegra.py --employee 100500 --position Аналитик
    python scripts/chat_orchestrator_aegra.py --url http://localhost:8000 --debug
    python scripts/chat_orchestrator_aegra.py --thread <id>   # подключиться к треду
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from langgraph_sdk import get_client

ASSISTANT_ID = "analytic_orchestrator"
EXIT_WORDS = {"exit", "quit", ":q", "выход"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Интерактивный чат с analytic_orchestrator на запущенной aegra.",
    )
    p.add_argument(
        "--url",
        default=os.environ.get("AEGRA_URL", "http://localhost:8000"),
        help="Базовый URL aegra-сервера (env AEGRA_URL, дефолт http://localhost:8000).",
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
        default=None,
        help="Подключиться к существующему thread_id (по умолчанию создаётся новый).",
    )
    p.add_argument(
        "--no-easyrag",
        action="store_true",
        help="Отключить wiki-grounding (easyrag_enabled=false).",
    )
    p.add_argument(
        "--easyrag-top-k",
        type=int,
        default=5,
        help="Сколько wiki-сниппетов подмешивать (easyrag_top_k, дефолт 5).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Печатать сырые события стрима (event/data по всем узлам).",
    )
    return p.parse_args()


def _msg_type(m) -> str:
    """Тип сообщения из dict (HTTP) или объекта."""
    if isinstance(m, dict):
        return m.get("type") or m.get("role") or ""
    return getattr(m, "type", "") or ""


def _msg_content(m) -> str:
    if isinstance(m, dict):
        return m.get("content") or ""
    return getattr(m, "content", "") or ""


def _format_status(payload: dict) -> str | None:
    """Краткая строка-статус хода: intent и источники/ошибки, если есть."""
    parts: list[str] = []
    if payload.get("intent") is not None:
        parts.append(f"intent={payload['intent']}")
    snippets = payload.get("easyrag_snippets")
    if snippets:
        parts.append(f"wiki={len(snippets)} сниппет(ов)")
    if payload.get("analytics_answer") is not None:
        parts.append("json_analyzer: есть ответ")
    for key in ("metrics_error", "analytics_error", "easyrag_error"):
        if payload.get(key) is not None:
            parts.append(f"{key}={payload[key]!r}")
    return " · ".join(parts) if parts else None


def _print_update(data: dict, debug: bool) -> None:
    """Печать апдейта (stream_mode=updates): {node: payload}."""
    if not isinstance(data, dict):
        return
    for node, payload in data.items():
        if not isinstance(payload, dict):
            if debug:
                print(f"  [{node}] {payload!r}")
            continue
        for m in payload.get("messages") or []:
            if _msg_type(m) != "human":
                text = _msg_content(m)
                if text:
                    print(f"\nАссистент: {text}")
        status = _format_status(payload)
        if status:
            print(f"  ⟂ {status}")


async def _ensure_thread(client, thread_id: str | None) -> str:
    if thread_id:
        return thread_id
    thread = await client.threads.create()
    return thread["thread_id"]


async def main() -> int:
    args = parse_args()
    client = get_client(url=args.url)

    try:
        thread_id = await _ensure_thread(client, args.thread)
    except Exception as exc:  # noqa: BLE001
        print(f"Не удалось подключиться к aegra на {args.url}: {exc!r}", file=sys.stderr)
        print("Подними сервер (docker-compose / uvicorn ...plugins.app:app) и повтори.",
              file=sys.stderr)
        return 1

    configurable = {
        "boss_tabnum": args.boss,
        "employee_tabnum": args.employee,
        "position": args.position,
        "easyrag_enabled": not args.no_easyrag,
        "easyrag_top_k": args.easyrag_top_k,
    }

    print(f"=== Болталка с analytic_orchestrator @ {args.url} ===")
    print(f"thread: {thread_id}")
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
            async for ev in client.runs.stream(
                thread_id,
                ASSISTANT_ID,
                input={"messages": [{"role": "user", "content": user_text}]},
                config={"configurable": configurable},
                stream_mode="updates",
            ):
                if args.debug:
                    print(f"  [{ev.event}] {ev.data!r}")
                elif ev.event == "updates":
                    _print_update(ev.data, args.debug)
                elif ev.event == "error":
                    print(f"  [ошибка сервера] {ev.data!r}")
        except Exception as exc:  # noqa: BLE001 — болталка не должна падать на одном ходе
            print(f"  [ошибка хода] {exc!r}")
        print()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
