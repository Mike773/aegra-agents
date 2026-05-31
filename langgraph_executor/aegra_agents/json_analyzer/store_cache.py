"""Кэш эмбеддингов поверх LangGraph Store + поиск в памяти.

Раньше json_analyzer держал свой Postgres-коннект (psycopg) и таблицу
``json_analyzer.metric_embeddings`` с pgvector. Это требовало дублировать
конфигурацию подключения aegra (включая нетривиальный SSL). Теперь вектора
лежат в LangGraph Store — том самом, что рантайм (aegra) прокидывает в граф:
запись идёт в его стандартные таблицы его же подключением, отдельный коннект и
SSL не нужны.

Store у aegra поднят БЕЗ векторного индекса (нет таблицы ``store_vectors``),
поэтому встроенный семантический поиск Store недоступен — он и не нужен:
эмбеддинги считаем сами (GigaChat) и храним как обычные KV-значения, а
косинусный поиск делаем в памяти (корпус направления маленький). Изоляция
направлений — через неймспейс Store (вместо прежней колонки direction_key).
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Awaitable, Callable

import numpy as np
from langgraph.store.base import BaseStore

_NAMESPACE_ROOT = ("json_analyzer", "metric_embeddings")
_PAGE = 200


def make_hash(kind: str, content: str) -> str:
    return hashlib.sha256(f"{kind}\x00{content}".encode("utf-8")).hexdigest()


def _ns(direction_key: str) -> tuple[str, ...]:
    return (*_NAMESPACE_ROOT, direction_key)


class EmbeddingIndex:
    """In-memory косинусный поиск по кэшу эмбеддингов одного направления.

    Замена прежнему pgvector-поиску: на вход — записи
    ``{kind, content, canonical, embedding}``; на выход (``search``) — тот же
    формат, что отдавал ``PgCache.search``.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._kinds: list[str] = []
        self._canonical: list[str] = []
        self._content: list[str] = []
        vectors: list[list[float]] = []
        dim: int | None = None
        for rec in records:
            embedding = rec.get("embedding")
            if not embedding:
                continue
            if dim is None:
                dim = len(embedding)
            elif len(embedding) != dim:
                raise ValueError(
                    "Несогласованная размерность эмбеддингов в кэше: "
                    f"{dim} vs {len(embedding)}. Похоже, кэш направления собран "
                    "разными моделями — очисти неймспейс и пересчитай."
                )
            self._kinds.append(rec.get("kind", ""))
            self._canonical.append(rec.get("canonical", ""))
            self._content.append(rec.get("content", ""))
            vectors.append(embedding)
        if vectors:
            matrix = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = matrix / norms
        else:
            self._matrix = np.empty((0, 0), dtype=np.float32)

    def search(
        self,
        query_embedding: list[float],
        kinds: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if self._matrix.shape[0] == 0:
            return []
        query = np.asarray(query_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        sims = self._matrix @ (query / norm)
        kind_set = set(kinds) if kinds else None
        candidates = [
            i
            for i in range(len(self._kinds))
            if kind_set is None or self._kinds[i] in kind_set
        ]
        candidates.sort(key=lambda i: sims[i], reverse=True)
        return [
            {
                "kind": self._kinds[i],
                "canonical": self._canonical[i],
                "content": self._content[i],
                "similarity": round(float(sims[i]), 4),
            }
            for i in candidates[:top_k]
        ]


async def load_cache(store: BaseStore, direction_key: str) -> dict[str, dict[str, Any]]:
    """Читает все записи неймспейса направления: ``{content_hash: value}``.

    ``asearch`` без ``query`` — это плоский листинг неймспейса (векторный индекс
    Store не требуется). Тянем постранично, пока страница полная.
    """
    namespace = _ns(direction_key)
    loaded: dict[str, dict[str, Any]] = {}
    offset = 0
    while True:
        page = await store.asearch(namespace, limit=_PAGE, offset=offset)
        for item in page:
            loaded[item.key] = dict(item.value)
        if len(page) < _PAGE:
            break
        offset += _PAGE
    return loaded


async def sync_embeddings(
    store: BaseStore,
    sqlite_store: Any,
    direction_key: str,
    embed_documents: Callable[[list[str]], list[list[float]]],
) -> EmbeddingIndex:
    """Досчитывает недостающие эмбеддинги корпуса направления и строит индекс.

    Корпус — distinct названия/описания/значения element загруженного датасета.
    Эмбеддятся только тексты, которых ещё нет в Store для этого направления;
    результат (старое из кэша ∪ новое) собирается в ``EmbeddingIndex``.
    ``embed_documents`` — синхронный GigaChat-callable, поэтому зовём его в
    отдельном потоке, чтобы не блокировать event loop.
    """
    raw: list[tuple[str, str, str]] = []
    for name in sqlite_store.distinct_metric_names():
        raw.append(("metric_name", name, name))
    for name, description in sqlite_store.distinct_descriptions():
        raw.append(("metric_description", description, name))
    for element in sqlite_store.distinct_elements():
        raw.append(("element", element, element))

    seen: set[tuple[str, str]] = set()
    by_hash: dict[str, tuple[str, str, str]] = {}
    for kind, content, canonical in raw:
        if not content:
            continue
        key = (kind, content)
        if key in seen:
            continue
        seen.add(key)
        by_hash[make_hash(kind, content)] = (kind, content, canonical)

    loaded = await load_cache(store, direction_key)
    missing = {h: meta for h, meta in by_hash.items() if h not in loaded}

    if missing:
        hashes = list(missing.keys())
        texts = [missing[h][1] for h in hashes]
        vectors = await asyncio.to_thread(embed_documents, texts)
        namespace = _ns(direction_key)
        for h, vector in zip(hashes, vectors):
            kind, content, canonical = missing[h]
            value = {
                "kind": kind,
                "content": content,
                "canonical": canonical,
                "embedding": list(vector),
            }
            await store.aput(namespace, h, value, index=False)
            loaded[h] = value

    return EmbeddingIndex(list(loaded.values()))


__all__ = ["EmbeddingIndex", "load_cache", "sync_embeddings", "make_hash"]
