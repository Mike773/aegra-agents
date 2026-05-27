"""DDL и автомиграции для таблицы ``metric_embeddings``.

Запускаются при инициализации :class:`pg_cache.PgCache`. Это не версионированная
система с историей ревизий — для единственной таблицы кэша достаточно
idempotent-стратегии: при каждом подключении проверяем форму таблицы и
пересоздаём её, если форма не соответствует ожидаемой. Кэш самовосстанавливается
синхронизацией эмбеддингов после пересоздания, поэтому пересоздание — не потеря
данных, а вынужденная мера.

Поводы для пересоздания:

* Размерность вектора в существующей таблице не равна ожидаемой ``dim`` —
  она закодирована в типе колонки ``vector(N)`` и не меняется на лету.
* В таблице нет колонки ``direction_key`` — это наследие standalone-проекта,
  в котором кэш был общий для всех направлений.
"""
from __future__ import annotations

from typing import Any

from psycopg import Connection, sql


def ensure_schema(conn: Connection, dim: int, *, schema: str = "") -> None:
    """Готовит подключение и таблицу ``metric_embeddings`` к работе.

    Idempotent: повторный вызов с теми же ``dim``/``schema`` не делает ничего
    нового. При изменении размерности или legacy-форме без ``direction_key``
    таблица пересоздаётся.
    """
    with conn.cursor() as cur:
        if schema:
            _set_search_path(cur, schema)
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if _table_exists(cur) and _needs_recreate(cur, dim):
            cur.execute("DROP TABLE metric_embeddings")
        _create_table(cur, dim)
        _create_indexes(cur)


def _set_search_path(cur: Any, schema: str) -> None:
    cur.execute(
        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
    )
    cur.execute(
        sql.SQL("SET search_path TO {}, public, ext").format(sql.Identifier(schema))
    )


def _table_exists(cur: Any) -> bool:
    cur.execute("SELECT to_regclass('metric_embeddings')")
    row = cur.fetchone()
    return row is not None and row[0] is not None


def _needs_recreate(cur: Any, dim: int) -> bool:
    existing_dim = _existing_dim(cur)
    if existing_dim is not None and existing_dim != dim:
        return True
    if not _has_direction_key(cur):
        return True
    return False


def _existing_dim(cur: Any) -> int | None:
    """Размерность вектора в существующей таблице или ``None``, если её нет."""
    cur.execute(
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = to_regclass('metric_embeddings') "
        "AND attname = 'embedding'"
    )
    row = cur.fetchone()
    if row is None or row[0] is None or row[0] <= 0:
        return None
    return int(row[0])


def _has_direction_key(cur: Any) -> bool:
    cur.execute(
        "SELECT 1 FROM pg_attribute "
        "WHERE attrelid = to_regclass('metric_embeddings') "
        "AND attname = 'direction_key'"
    )
    return cur.fetchone() is not None


def _create_table(cur: Any, dim: int) -> None:
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS metric_embeddings (
            id            BIGSERIAL PRIMARY KEY,
            direction_key TEXT NOT NULL,
            kind          TEXT NOT NULL,
            content       TEXT NOT NULL,
            content_hash  TEXT NOT NULL,
            canonical     TEXT NOT NULL,
            embedding     vector({dim}) NOT NULL,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_metric_embeddings_dir_hash
                UNIQUE (direction_key, content_hash)
        )
        """
    )


def _create_indexes(cur: Any) -> None:
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_metric_embeddings_dir_kind "
        "ON metric_embeddings(direction_key, kind)"
    )


__all__ = ["ensure_schema"]
