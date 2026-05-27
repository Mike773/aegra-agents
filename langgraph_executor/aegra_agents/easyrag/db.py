"""Подключение к БД для подагента easyrag.

В проде графы поднимает aegra-api, у неё уже есть готовый ``AsyncEngine``
(``aegra_api.core.database.db_manager.engine``) — переиспользуем его, чтобы
не плодить второй пул на тот же DSN. Для standalone-запусков (smoke-скрипты,
юнит-тесты) собираем свой engine из ``POSTGRES_DSN``/``DATABASE_URL``.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

_ENV_LOADED = False
_FALLBACK_ENGINE: AsyncEngine | None = None


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break
    _ENV_LOADED = True


def _async_dsn(raw: str) -> str:
    if raw.startswith("postgresql+asyncpg://"):
        return raw
    if raw.startswith("postgresql://"):
        return "postgresql+asyncpg://" + raw[len("postgresql://") :]
    return raw


def get_engine() -> AsyncEngine:
    # Под aegra-api используем её engine — общий пул соединений.
    try:
        from aegra_api.core.database import db_manager  # type: ignore[import-not-found]
    except ImportError:
        db_manager = None
    if db_manager is not None and getattr(db_manager, "engine", None) is not None:
        return db_manager.engine

    # Standalone-фолбэк: собираем свой engine.
    global _FALLBACK_ENGINE
    if _FALLBACK_ENGINE is None:
        _ensure_env_loaded()
        dsn = (
            os.environ.get("POSTGRES_DSN")
            or os.environ.get("DATABASE_URL")
            or ""
        ).strip()
        if not dsn:
            raise RuntimeError(
                "easyrag.db: нет ни aegra db_manager.engine, "
                "ни POSTGRES_DSN/DATABASE_URL в окружении."
            )
        _FALLBACK_ENGINE = create_async_engine(
            _async_dsn(dsn),
            pool_pre_ping=True,
            connect_args={"server_settings": {"search_path": "wiki_rag,public"}},
        )
    return _FALLBACK_ENGINE


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    async with AsyncSession(get_engine(), expire_on_commit=False) as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


__all__ = ["get_engine", "session_scope"]
