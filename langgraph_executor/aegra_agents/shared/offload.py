"""Безопасный вызов блокирующих внешних клиентов из async-узлов.

Клиенты внешних сервисов данных (`agent_dataset`, `orgstructure`,
`assignments_service`) — СИНХРОННЫЕ (в проде ходят по HTTP). Если звать их
прямо в async-узле, они блокируют event loop; если просто увести в общий
``asyncio.to_thread`` — они делят дефолтный пул потоков с LLM-вызовами и при
зависании внешнего сервиса выедают его, роняя весь сервис.

Поэтому уводим такие вызовы в ОТДЕЛЬНЫЙ ограниченный пул и оборачиваем
эффективным таймаутом:

* выделенный ``ThreadPoolExecutor`` (``DATA_IO_MAX_WORKERS``, дефолт 16) —
  изолирован от дефолтного пула, поэтому зависший сервис данных не блокирует
  LLM-вызовы; число одновременно зависших потоков ограничено сверху;
* ``asyncio.wait_for`` (``DATA_IO_TIMEOUT``, дефолт 30с) — даёт оркестратору
  ВЫЙТИ из ожидания, даже если у самого клиента таймаута нет. Узел получит
  ``TimeoutError`` и деградирует, запрос продолжится, а не «висит навсегда».

Важно: ``wait_for`` отменяет ожидание корутины, но НЕ убивает уже бегущий
поток (его нельзя прервать). Поэтому при действительно мёртвом сервисе потоки
копятся до ``DATA_IO_MAX_WORKERS`` — настоящий socket-таймаут в самом клиенте
всё равно нужен для полного здоровья. Здесь мы лишь не даём оркестратору
зависнуть и не позволяем внешнему I/O утопить пул LLM.
"""
from __future__ import annotations

import asyncio
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_MAX_WORKERS = int(os.environ.get("DATA_IO_MAX_WORKERS") or "16")
_DEFAULT_TIMEOUT = float(os.environ.get("DATA_IO_TIMEOUT") or "30")

# Отдельный пул только под внешний data-I/O — НЕ дефолтный (тот занят LLM/to_thread).
_executor = ThreadPoolExecutor(
    max_workers=_MAX_WORKERS, thread_name_prefix="data-io"
)


async def run_blocking(
    func: Callable[..., T],
    *args: Any,
    timeout: float | None = _DEFAULT_TIMEOUT,
    **kwargs: Any,
) -> T:
    """Выполнить блокирующий ``func`` в выделенном пуле, не блокируя event loop.

    timeout: верхняя граница ожидания (сек). ``None`` — без таймаута.
    Бросает ``asyncio.TimeoutError`` при превышении (вызывающий узел ловит и
    деградирует).
    """
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(_executor, functools.partial(func, *args, **kwargs))
    if timeout and timeout > 0:
        return await asyncio.wait_for(fut, timeout)
    return await fut


__all__ = ["run_blocking"]
