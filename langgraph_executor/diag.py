"""Диагностический сэмплер пулов — ТОЛЬКО под флагом ``DIAG=1``.

Назначение: под нагрузкой (2 залпа по 30 запросов) показать, какой именно
разделяемый ресурс выедается и не освобождается, когда второй залп виснет:

  * thread-pool: число живых тредов и backlog в ThreadPoolExecutor'ах
    (дефолтный executor loop'а обслуживает И sync-узлы LangGraph, И
    ``asyncio.to_thread`` — размер ``min(32, cpu+4)``);
  * БД-пул: ``checkedout/checkedin/overflow`` общего AsyncEngine
    (его делят чекпойнтер aegra и ``session_scope`` easyrag);
  * длительность каждого ``llm.invoke``/``ainvoke`` (см. ``wrap_llm_timing``)
    — ловим «вечные» вызовы GigaChat без таймаута.

Ничего не чинит и не меняет поведение при ``DIAG`` не выставленном — сэмплер
просто не стартует. Всё гонится в отдельном daemon-потоке, loop не трогает.

Включение:
    DIAG=1 uvicorn langgraph_executor.plugins.app:app --workers 2
По желанию: DIAG_INTERVAL=0.5 (период сэмплинга, сек, дефолт 1.0).
"""
from __future__ import annotations

import faulthandler
import gc
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("diag")

_started = False
_lock = threading.Lock()


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    return _truthy(os.environ.get("DIAG"))


def _executor_lines() -> list[str]:
    """Backlog по всем живым ThreadPoolExecutor'ам процесса.

    Дефолтный executor event loop'а (его использует и LangGraph для sync-узлов,
    и asyncio.to_thread) тут виден как ThreadPoolExecutor без prefix. ``queued``
    > 0 при ``threads == max`` = пул потоков исчерпан, задачи стоят в очереди.
    """
    lines: list[str] = []
    for obj in gc.get_objects():
        if not isinstance(obj, ThreadPoolExecutor):
            continue
        try:
            queued = obj._work_queue.qsize()  # type: ignore[attr-defined]
            nthreads = len(obj._threads)  # type: ignore[attr-defined]
            mx = obj._max_workers  # type: ignore[attr-defined]
            prefix = obj._thread_name_prefix or "<default>"  # type: ignore[attr-defined]
            lines.append(f"exec[{prefix}] threads={nthreads}/{mx} queued={queued}")
        except Exception:  # noqa: BLE001 — приватные поля могут поменяться между версиями
            continue
    return lines or ["exec: <нет ThreadPoolExecutor>"]


def _db_status() -> str:
    """Статус общего пула SQLAlchemy: checkedout — занятые коннекты."""
    try:
        from langgraph_executor.aegra_agents.easyrag.db import get_engine

        return get_engine().pool.status()
    except Exception as exc:  # noqa: BLE001 — пул может быть ещё не создан / недоступен
        return f"<db pool n/a: {exc!r}>"


def _thread_breakdown() -> str:
    threads = threading.enumerate()
    tpe = sum(1 for t in threads if t.name.startswith("ThreadPoolExecutor"))
    asy = sum(1 for t in threads if t.name.startswith("asyncio"))
    return f"threads total={len(threads)} tpe={tpe} asyncio={asy}"


def _run(interval: float) -> None:
    log.warning("DIAG sampler started pid=%s interval=%.2fs", os.getpid(), interval)
    while True:
        try:
            parts = [_thread_breakdown(), *_executor_lines(), f"db_pool: {_db_status()}"]
            log.warning("DIAG | %s", " | ".join(parts))
        except Exception as exc:  # noqa: BLE001 — сэмплер не должен ронять процесс
            log.warning("DIAG sampler error: %r", exc)
        time.sleep(interval)


def _install_faulthandler() -> None:
    """Дамп стеков ВСЕХ потоков — чтобы поймать дедлок (на чём стоят потоки).

    * по сигналу ``kill -USR1 <pid>`` — мгновенный дамп в stderr;
    * если задан ``DIAG_DUMP_EVERY`` (сек) — авто-дамп с этим периодом
      (ловит вис без ручного сигнала).
    Печатает PID, чтобы знать кого слать сигналом.
    """
    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
        log.warning("DIAG faulthandler: kill -USR1 %s → дамп стеков всех потоков", os.getpid())
    every = os.environ.get("DIAG_DUMP_EVERY")
    if every:
        faulthandler.dump_traceback_later(float(every), repeat=True, file=sys.stderr)
        log.warning("DIAG faulthandler: авто-дамп стеков каждые %sс", every)


def start_diag() -> None:
    """Стартует фоновый сэмплер (идемпотентно). No-op без ``DIAG=1``."""
    global _started
    if not enabled():
        return
    with _lock:
        if _started:
            return
        _started = True
    _install_faulthandler()
    interval = float(os.environ.get("DIAG_INTERVAL") or "1.0")
    thread = threading.Thread(
        target=_run, args=(interval,), name="diag-sampler", daemon=True
    )
    thread.start()


def wrap_llm_timing(llm):
    """Оборачивает ``invoke``/``ainvoke`` LLM для лога длительности каждого вызова.

    No-op без ``DIAG=1``. Логирует thread id и время — «вечные» вызовы (нет
    таймаута у GigaChat) видно как строки без парного завершения / с большим dt.
    """
    if not enabled():
        return llm

    orig_invoke = llm.invoke
    orig_ainvoke = getattr(llm, "ainvoke", None)

    def invoke(*args, **kwargs):
        tid = threading.get_ident()
        t0 = time.monotonic()
        log.warning("LLM invoke start tid=%s", tid)
        try:
            return orig_invoke(*args, **kwargs)
        finally:
            log.warning("LLM invoke end   tid=%s dt=%.2fs", tid, time.monotonic() - t0)

    # GigaChat — pydantic-модель: обычное присваивание ловит валидатор полей,
    # поэтому затеняем метод на инстансе через object.__setattr__.
    object.__setattr__(llm, "invoke", invoke)

    if orig_ainvoke is not None:
        async def ainvoke(*args, **kwargs):
            t0 = time.monotonic()
            log.warning("LLM ainvoke start")
            try:
                return await orig_ainvoke(*args, **kwargs)
            finally:
                log.warning("LLM ainvoke end   dt=%.2fs", time.monotonic() - t0)

        object.__setattr__(llm, "ainvoke", ainvoke)

    return llm


__all__ = ["start_diag", "wrap_llm_timing", "enabled"]
