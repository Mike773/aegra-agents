"""Нагрузочный воспроизводитель бага «второй залп виснет».

Бьёт по ЗАПУЩЕННОЙ aegra (HTTP, langgraph_sdk) несколькими залпами по N
одновременных запросов и печатает по каждому залпу: сколько дошло до конца,
сколько упало/зависло и тайминги. Симптом из задачи: первый залп проходит,
следующий обрабатывает лишь несколько штук.

Каждый запрос — это свежий thread + один ход в граф (первая реплика запускает
анализ метрик). Зависание ловим по --timeout: не дождались финала за N сек =
TIMEOUT (именно эти и есть «висящие» раны).

Сервер поднять как в проде:
    DIAG=1 uvicorn langgraph_executor.plugins.app:app --workers 2

Запуск нагрузки (в соседнем терминале):
    python scripts/loadtest_orchestrator.py --batches 2 --concurrency 30
    python scripts/loadtest_orchestrator.py --assistant analytic_orchestrator_v2 \
        --concurrency 30 --batches 3 --timeout 180 --gap 5
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from langgraph_sdk import get_client


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=os.environ.get("AEGRA_URL", "http://localhost:8000"))
    p.add_argument("--assistant", default="analytic_orchestrator_v2",
                   help="ID графа (дефолт analytic_orchestrator_v2).")
    p.add_argument("--concurrency", type=int, default=30,
                   help="Сколько запросов в одном залпе одновременно.")
    p.add_argument("--batches", type=int, default=2,
                   help="Сколько залпов подряд (баг проявляется со 2-го).")
    p.add_argument("--gap", type=float, default=2.0,
                   help="Пауза между залпами, сек.")
    p.add_argument("--timeout", type=float, default=180.0,
                   help="Потолок ожидания финала одного рана, сек (превышение = завис).")
    p.add_argument("--message", default="Что происходит с показателями сотрудника?")
    p.add_argument("--employee", default=os.environ.get("DEMO_EMP_TABNUM", "100500"))
    p.add_argument("--boss", default=os.environ.get("DEMO_BOSS_TABNUM", "100001"))
    p.add_argument("--position", default=os.environ.get("DEMO_POSITION", "Аналитик"))
    return p.parse_args()


async def _one_request(client, args, idx: int) -> tuple[str, float]:
    """Один полный ход. Возврат: (исход, длительность). Исход: ok/timeout/error."""
    cfg = {
        "boss_tabnum": args.boss,
        "employee_tabnum": args.employee,
        "position": args.position,
    }
    t0 = time.monotonic()
    try:
        thread = await client.threads.create()
        tid = thread["thread_id"]

        async def _drain() -> None:
            async for ev in client.runs.stream(
                tid, args.assistant,
                input={"messages": [{"role": "user", "content": args.message}]},
                config={"configurable": cfg},
                stream_mode="updates",
            ):
                if ev.event == "error":
                    raise RuntimeError(f"server error: {ev.data!r}")

        await asyncio.wait_for(_drain(), timeout=args.timeout)
        return "ok", time.monotonic() - t0
    except asyncio.TimeoutError:
        return "timeout", time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001 — фиксируем любой исход рана
        return f"error:{type(exc).__name__}", time.monotonic() - t0


async def _run_batch(client, args, batch_no: int) -> None:
    print(f"\n=== Залп #{batch_no}: {args.concurrency} одновременных запросов ===")
    t0 = time.monotonic()
    results = await asyncio.gather(
        *[_one_request(client, args, i) for i in range(args.concurrency)]
    )
    elapsed = time.monotonic() - t0

    ok = [d for s, d in results if s == "ok"]
    timeouts = [d for s, d in results if s == "timeout"]
    errors = [s for s, _ in results if s.startswith("error")]
    print(f"  завершено за {elapsed:.1f}с | ok={len(ok)} timeout={len(timeouts)} "
          f"error={len(errors)}")
    if ok:
        print(f"  ok-тайминги: min={min(ok):.1f}с max={max(ok):.1f}с "
              f"avg={sum(ok)/len(ok):.1f}с")
    if errors:
        from collections import Counter
        print(f"  ошибки: {dict(Counter(errors))}")
    if timeouts:
        print(f"  !! {len(timeouts)} ранов не уложились в {args.timeout:.0f}с — это и есть «висящие»")


async def main() -> int:
    args = parse_args()
    client = get_client(url=args.url)
    try:
        await client.threads.create()  # проба связи
    except Exception as exc:  # noqa: BLE001
        print(f"Нет связи с aegra на {args.url}: {exc!r}", file=sys.stderr)
        print("Подними сервер: DIAG=1 uvicorn langgraph_executor.plugins.app:app --workers 2",
              file=sys.stderr)
        return 1

    for b in range(1, args.batches + 1):
        await _run_batch(client, args, b)
        if b < args.batches:
            await asyncio.sleep(args.gap)
    print("\n=== нагрузка завершена ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
