"""In-process воспроизведение бага без HTTP-сервера aegra.

Дёргает скомпилированный граф analytic_orchestrator_v2 напрямую N параллельными
``ainvoke`` (несколько залпов), пока работает DIAG-сэмплер. Прод-сервер aegra
в этом окружении не поднимается (нет plugins.app/uvicorn), но путь
route(sync)→initial_analysis→json_analyzer(tool-loop в to_thread)→ground_wiki(DB)
→form_insights→respond здесь тот же — а значит видны thread-pool и БД-пул.

ВАЖНО: запускать с DIAG=1, чтобы сэмплер писал счётчики:
    DIAG=1 DIAG_INTERVAL=0.5 .venv/bin/python scripts/loadtest_inprocess.py \
        --concurrency 6 --batches 2 --timeout 180
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # корень репо для import link

def _on(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


# Мок LLM (обход квоты 429) ДО импорта графов — фабрики подменяются на сборке графа.
if _on("MOCK_LLM"):
    import langgraph_executor.mock_llm as _mock  # noqa: E402

    _mock.install()
    print(f"[mock] LLM подменён на MockGigaChat (без сети/квоты); pid={os.getpid()}"
          f" full_path={os.environ.get('MOCK_FULL_PATH')}", flush=True)
elif _on("MOCK_DATASET"):
    # Реальный LLM + реальные эмбеддинги, но непустой датасет — чтобы РЕАЛЬНАЯ
    # сеть прошла tool-loop json_analyzer (ловим сетевой вис локально).
    import langgraph_executor.mock_llm as _mock  # noqa: E402

    _mock._install_full_dataset()
    print(f"[dataset] непустой датасет, LLM/эмбеддинги РЕАЛЬНЫЕ; pid={os.getpid()}", flush=True)

# импорт link стартует DIAG-сэмплер и компилирует графы
import link  # noqa: F401,E402
from langgraph_executor.aegra_agents.analytic_orchestrator_v2.graph import graph


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--batches", type=int, default=2)
    p.add_argument("--gap", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--message", default="Что происходит с показателями сотрудника?")
    p.add_argument("--employee", default=os.environ.get("DEMO_EMP_TABNUM", "100500"))
    p.add_argument("--boss", default=os.environ.get("DEMO_BOSS_TABNUM", "100001"))
    p.add_argument("--position", default=os.environ.get("DEMO_POSITION", "Аналитик"))
    return p.parse_args()


async def _one(args, idx: int) -> tuple[str, float]:
    cfg = {"configurable": {
        "boss_tabnum": args.boss,
        "employee_tabnum": args.employee,
        "position": args.position,
        "easyrag_enabled": True,
        "easyrag_top_k": 5,
    }}
    payload = {"messages": [{"role": "user", "content": args.message}]}
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(graph.ainvoke(payload, config=cfg), timeout=args.timeout)
        return "ok", time.monotonic() - t0
    except asyncio.TimeoutError:
        return "timeout", time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001
        return f"error:{type(exc).__name__}:{str(exc)[:80]}", time.monotonic() - t0


async def _batch(args, n: int) -> None:
    print(f"\n=== Залп #{n}: {args.concurrency} параллельных ainvoke ===", flush=True)
    t0 = time.monotonic()
    res = await asyncio.gather(*[_one(args, i) for i in range(args.concurrency)])
    el = time.monotonic() - t0
    ok = [d for s, d in res if s == "ok"]
    to = [d for s, d in res if s == "timeout"]
    err = [s for s, _ in res if s.startswith("error")]
    print(f"  за {el:.1f}с | ok={len(ok)} timeout={len(to)} error={len(err)}", flush=True)
    if ok:
        print(f"  ok: min={min(ok):.1f} max={max(ok):.1f} avg={sum(ok)/len(ok):.1f}с", flush=True)
    for s in err[:5]:
        print(f"  err: {s}", flush=True)
    if to:
        print(f"  !! {len(to)} зависли (> {args.timeout:.0f}с)", flush=True)


async def main() -> int:
    args = parse_args()
    for b in range(1, args.batches + 1):
        await _batch(args, b)
        if b < args.batches:
            await asyncio.sleep(args.gap)
    print("\n=== готово ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
