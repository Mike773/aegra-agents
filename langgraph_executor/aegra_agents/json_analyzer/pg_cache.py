"""Кэш эмбеддингов в PostgreSQL + pgvector — runtime-операции.

Хранит вектора названий/описаний метрик и значений поля element. Ключ кэша —
sha256 от (kind, текст), но изоляция между направлениями обеспечивается отдельной
колонкой direction_key и составным UNIQUE (direction_key, content_hash):
один и тот же текст в двух направлениях занимает две строки, поиск возвращает
только записи своего direction_key.

DDL и автомиграция таблицы — в :mod:`.migrations`; здесь только подключение
и CRUD/поиск. DSN читается из POSTGRES_DSN/DATABASE_URL (как у easyrag/db.py);
psycopg — синхронный, узлы графа тоже sync. Размерность вектора либо
передаётся явно (gather_node делает runtime-probe через embed_query), либо
читается из EMBEDDING_DIM с грубым fallback 2560 — этот fallback используется
только при ручных вызовах PgCache без явного dim.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

from .migrations import ensure_schema

_ENV_LOADED = False


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


def _sync_dsn() -> str:
    """DSN в формате, который понимает sync psycopg.

    aegra-api/easyrag используют asyncpg-DSN (`postgresql+asyncpg://`); тут нужен
    обычный `postgresql://`. Если в env лежит asyncpg-форма — конвертируем.
    """
    _ensure_env_loaded()
    dsn = (
        os.environ.get("POSTGRES_DSN")
        or os.environ.get("DATABASE_URL")
        or ""
    ).strip()
    if not dsn:
        raise RuntimeError(
            "json_analyzer/pg_cache: не задан POSTGRES_DSN/DATABASE_URL в окружении."
        )
    if dsn.startswith("postgresql+asyncpg://"):
        dsn = "postgresql://" + dsn[len("postgresql+asyncpg://"):]
    return dsn


def _embedding_dim() -> int:
    raw = os.environ.get("EMBEDDING_DIM", "2560").strip()
    try:
        return int(raw)
    except ValueError:
        return 2560


def make_hash(kind: str, content: str) -> str:
    return hashlib.sha256(f"{kind}\x00{content}".encode("utf-8")).hexdigest()


class PgCache:
    """Доступ к таблице metric_embeddings с изоляцией по direction_key."""

    def __init__(self, dsn: str | None = None, dim: int | None = None) -> None:
        self.dsn = dsn or _sync_dsn()
        self.dim = dim or _embedding_dim()
        self.schema = (os.environ.get("POSTGRES_SCHEMA") or "").strip()
        self.conn = psycopg.connect(self.dsn, autocommit=True)
        ensure_schema(self.conn, self.dim, schema=self.schema)
        register_vector(self.conn)

    def existing_hashes(self, hashes: list[str], direction_key: str) -> set[str]:
        if not hashes:
            return set()
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT content_hash FROM metric_embeddings "
                "WHERE direction_key = %s AND content_hash = ANY(%s)",
                (direction_key, list(hashes)),
            )
            return {row[0] for row in cur.fetchall()}

    def upsert(
        self,
        items: list[tuple[str, str, str, list[float]]],
        direction_key: str,
    ) -> int:
        """Вставляет новые вектора. items: (kind, content, canonical, embedding)."""
        if not items:
            return 0
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO metric_embeddings "
                "(direction_key, kind, content, content_hash, canonical, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (direction_key, content_hash) DO NOTHING",
                [
                    (
                        direction_key,
                        kind,
                        content,
                        make_hash(kind, content),
                        canonical,
                        Vector(embedding),
                    )
                    for kind, content, canonical, embedding in items
                ],
            )
        return len(items)

    def search(
        self,
        query_embedding: list[float],
        direction_key: str,
        kinds: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Поиск ближайших по косинусу записей кэша для конкретного направления."""
        query_vector = Vector(query_embedding)
        params: list[Any] = [query_vector, direction_key]
        kind_clause = ""
        if kinds:
            kind_clause = "AND kind = ANY(%s)"
            params.append(list(kinds))
        params.append(query_vector)
        params.append(top_k)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT kind, canonical, content,
                       1 - (embedding <=> %s) AS similarity
                FROM metric_embeddings
                WHERE direction_key = %s
                {kind_clause}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                params,
            )
            return [
                {
                    "kind": kind,
                    "canonical": canonical,
                    "content": content,
                    "similarity": round(float(similarity), 4),
                }
                for kind, canonical, content, similarity in cur.fetchall()
            ]

    def row_count(self, direction_key: str | None = None) -> int:
        with self.conn.cursor() as cur:
            if direction_key is None:
                cur.execute("SELECT COUNT(*) FROM metric_embeddings")
            else:
                cur.execute(
                    "SELECT COUNT(*) FROM metric_embeddings WHERE direction_key = %s",
                    (direction_key,),
                )
            return cur.fetchone()[0]

    def close(self) -> None:
        self.conn.close()


def sync_embeddings(
    store: Any,
    pg: PgCache,
    direction_key: str,
    embed_documents: Any,
) -> dict[str, int]:
    """Досчитывает в кэш эмбеддинги для текстов загруженного датасета.

    Эмбеддятся только тексты, отсутствующие в кэше СВОЕГО direction_key
    (проверка по content_hash + direction_key). embed_documents — callable
    `(list[str]) -> list[list[float]]` (см. shared.clients.create_gigachat_embeddings).
    """
    raw: list[tuple[str, str, str]] = []
    for name in store.distinct_metric_names():
        raw.append(("metric_name", name, name))
    for name, description in store.distinct_descriptions():
        raw.append(("metric_description", description, name))
    for element in store.distinct_elements():
        raw.append(("element", element, element))

    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str, str]] = []
    for kind, content, canonical in raw:
        if not content:
            continue
        key = (kind, content)
        if key in seen:
            continue
        seen.add(key)
        unique.append((kind, content, canonical))

    by_hash = {make_hash(k, c): (k, c, ca) for k, c, ca in unique}
    existing = pg.existing_hashes(list(by_hash.keys()), direction_key=direction_key)
    missing = [meta for h, meta in by_hash.items() if h not in existing]

    added = 0
    if missing:
        vectors = embed_documents([content for _, content, _ in missing])
        if vectors and len(vectors[0]) != pg.dim:
            raise RuntimeError(
                f"Размерность эмбеддинга {len(vectors[0])} не совпадает с "
                f"таблицей ({pg.dim}). Задайте EMBEDDING_DIM={len(vectors[0])} "
                f"и пересоздайте таблицу metric_embeddings."
            )
        added = pg.upsert(
            [
                (kind, content, canonical, vector)
                for (kind, content, canonical), vector in zip(missing, vectors)
            ],
            direction_key=direction_key,
        )

    return {"total": len(unique), "added": added, "cached": len(unique) - added}
