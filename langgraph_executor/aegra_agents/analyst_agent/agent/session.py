"""Сессия GigaChat на один ход: заголовок ``X-Session-ID``.

Каждый вызов модели внутри цикла инструментов шлёт весь контекст заново, и
GigaChat умеет отдавать общий префикс из кэша (``precached_prompt_tokens``),
но только если запросы приходят в одной сессии. SDK читает идентификатор из
контекстной переменной ``session_id_cvar``, а ``asyncio.to_thread`` копирует
контекст в рабочий поток, так что выставить его достаточно один раз вокруг
цикла. Идентификатором служит ``thread_id`` диалога; без него — случайный
на ход, чтобы кэш работал хотя бы внутри одного хода.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from gigachat.context import session_id_cvar


@contextmanager
def llm_session(session_id: str | None = None) -> Iterator[str]:
    """Держит X-Session-ID на время блока и возвращает переменную обратно."""
    sid = (session_id or "").strip() or uuid.uuid4().hex
    token = session_id_cvar.set(sid)
    try:
        yield sid
    finally:
        session_id_cvar.reset(token)


__all__ = ["llm_session"]
