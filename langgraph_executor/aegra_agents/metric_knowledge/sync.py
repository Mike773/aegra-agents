"""Ленивое дозаполнение трактовок во время хода агента.

На первом ходе добираем ТОЛЬКО недостающие показатели и не больше заданного
числа, всё под общим таймаутом: руководитель ждёт ответ, а не полный обход
каталога. Остальное дособерёт фоновый граф ``metric_enricher``.

Недоступность Postgres здесь не фатальна: агент отработает без справки.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from ..easyrag.db import session_scope
from ..easyrag.retrieval import retrieve_sections
from ..wiki_ingest.embeddings import get_embeddings
from ..wiki_ingest.llm import get_llm
from . import lookup, repository
from .types import CatalogEntry, KnowledgeRow

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW = 10
DEFAULT_TIMEOUT_S = 25.0
DEFAULT_CONCURRENCY = 3


@dataclass
class KnowledgeSync:
    rows: dict[str, KnowledgeRow] = field(default_factory=dict)
    fetched: int = 0
    missing_left: int = 0
    error: str | None = None


async def fetch_snippets(
    entry: CatalogEntry, *, direction_key: str, embedder: Any, top_k: int = 6
) -> list[dict[str, Any]]:
    """Фрагменты базы знаний по нескольким запросам о показателе."""
    queries = lookup.build_queries(entry)
    if not queries:
        return []
    vectors = await embedder.embed_many(queries)
    found: dict[str, dict[str, Any]] = {}
    async with session_scope() as session:
        for vector in vectors:
            if not vector:
                continue
            sections = await retrieve_sections(
                session, direction_key=direction_key, query_vec=vector, top_k=top_k
            )
            for section in sections:
                data = section.to_dict()
                key = str(data.get("section_id"))
                if key not in found or data["similarity"] > found[key]["similarity"]:
                    found[key] = data
    return sorted(found.values(), key=lambda s: -s["similarity"])[: lookup.MAX_SNIPPETS]


async def enrich_one(
    entry: CatalogEntry,
    *,
    direction_key: str,
    embedder: Any,
    llm: Any,
    model_name: str | None = None,
    top_k: int = 6,
) -> KnowledgeRow | None:
    """Одна трактовка: поиск во фрагментах + сведение моделью."""
    try:
        snippets = await fetch_snippets(
            entry, direction_key=direction_key, embedder=embedder, top_k=top_k
        )
    except Exception:  # noqa: BLE001 — БД/эмбеддинги: кэш не отравляем
        logger.warning(
            "knowledge: поиск по «%s» не удался", entry.metric_name, exc_info=True
        )
        return None
    return await lookup.build_row(entry, snippets=snippets, llm=llm, model_name=model_name)


async def sync_knowledge(
    catalog: list[CatalogEntry],
    *,
    direction_key: str,
    max_new: int = DEFAULT_MAX_NEW,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    concurrency: int = DEFAULT_CONCURRENCY,
    embedder: Any = None,
    llm: Any = None,
) -> KnowledgeSync:
    """Известные трактовки + ограниченное дозаполнение недостающих."""
    result = KnowledgeSync()
    if not catalog or not direction_key:
        return result

    try:
        async with session_scope() as session:
            await repository.upsert_catalog(
                session, direction_key=direction_key, entries=catalog
            )
            result.rows = await repository.get_knowledge(
                session, direction_key=direction_key, entries=catalog
            )
    except Exception as exc:  # noqa: BLE001 — Postgres недоступен
        logger.warning("knowledge: кэш недоступен", exc_info=True)
        result.error = f"{type(exc).__name__}: {exc}"[:200]
        return result

    missing = repository.missing_entries(catalog, result.rows)
    result.missing_left = max(0, len(missing) - max_new)
    todo = missing[:max_new]
    if not todo:
        return result

    embedder = embedder or get_embeddings()
    llm = llm or get_llm()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _one(entry: CatalogEntry) -> KnowledgeRow | None:
        async with semaphore:
            return await enrich_one(
                entry, direction_key=direction_key, embedder=embedder, llm=llm
            )

    try:
        async with asyncio.timeout(timeout_s):
            found = await asyncio.gather(
                *(_one(e) for e in todo), return_exceptions=True
            )
    except TimeoutError:
        logger.info("knowledge: дозаполнение прервано по таймауту")
        return result

    fresh = [r for r in found if isinstance(r, KnowledgeRow)]
    if not fresh:
        return result
    try:
        async with session_scope() as session:
            await repository.upsert_knowledge(
                session, direction_key=direction_key, rows=fresh
            )
    except Exception:  # noqa: BLE001 — запись в кэш не критична для хода
        logger.warning("knowledge: не удалось сохранить трактовки", exc_info=True)
    for row in fresh:
        result.rows[row.metric_key] = row
    result.fetched = len(fresh)
    return result


__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MAX_NEW",
    "DEFAULT_TIMEOUT_S",
    "KnowledgeSync",
    "enrich_one",
    "fetch_snippets",
    "sync_knowledge",
]
