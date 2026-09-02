"""Узлы фонового обогатителя: взять недостающие трактовки и дозаполнить их.

Работает по таблице ``wiki_rag.metric_catalog`` — тому, что видел агент, — а не
по последнему загруженному датасету: так обогащение не зависит от того, чей
диалог был последним.

Каждый показатель обрабатывается в СВОЕЙ транзакции: сбой на одном не отменяет
уже найденное по остальным (тот же приём, что в gap_resolver).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..easyrag.db import session_scope
from ..metric_knowledge import repository
from ..metric_knowledge.sync import enrich_one
from ..metric_knowledge.types import STATUS_UNKNOWN, CatalogEntry
from ..wiki_ingest.embeddings import get_embeddings
from ..wiki_ingest.llm import get_llm

logger = logging.getLogger(__name__)

DEFAULT_MAX_METRICS = 50
DEFAULT_TOP_K = 6
DEFAULT_CONCURRENCY = 3


def _cfg(config: RunnableConfig | None, state: Any, key: str, default: Any) -> Any:
    """Порядок как в gap_resolver: configurable → state → дефолт."""
    conf = (config or {}).get("configurable") or {}
    for source in (conf, state or {}):
        value = source.get(key)
        if value is not None and value != "":
            return value
    return default


async def load_catalog(state: Any, config: RunnableConfig) -> dict:
    direction_key = str(_cfg(config, state, "direction_key", "") or "").strip()
    if not direction_key:
        return {"pending": [], "errors": ["не задан direction_key"]}
    limit = int(_cfg(config, state, "max_metrics", DEFAULT_MAX_METRICS))
    try:
        async with session_scope() as session:
            entries = await repository.load_catalog_pending(
                session, direction_key=direction_key, limit=limit
            )
    except Exception as exc:  # noqa: BLE001 — БД недоступна
        logger.warning("metric_enricher: каталог не прочитан", exc_info=True)
        return {"pending": [], "errors": [f"каталог недоступен: {exc}"[:200]]}
    return {
        "direction_key": direction_key,
        "pending": [e.__dict__ for e in entries],
    }


def after_load(state: Any) -> str:
    return "enrich" if (state or {}).get("pending") else "finalize"


async def enrich(state: Any, config: RunnableConfig) -> dict:
    direction_key = str(state.get("direction_key") or "")
    entries = [CatalogEntry(**e) for e in state.get("pending") or []]
    top_k = int(_cfg(config, state, "top_k", DEFAULT_TOP_K))
    concurrency = int(_cfg(config, state, "concurrency", DEFAULT_CONCURRENCY))

    embedder = get_embeddings()
    llm = get_llm()
    semaphore = asyncio.Semaphore(max(1, concurrency))
    found = unknown = 0
    errors: list[str] = []

    async def _one(entry: CatalogEntry) -> None:
        nonlocal found, unknown
        async with semaphore:
            row = await enrich_one(
                entry,
                direction_key=direction_key,
                embedder=embedder,
                llm=llm,
                top_k=top_k,
            )
        if row is None:
            # Транзиентный сбой: ничего не пишем, показатель попадёт в следующий прогон.
            errors.append(f"{entry.metric_name}: трактовка не получена")
            return
        try:
            async with session_scope() as session:
                await repository.upsert_knowledge(
                    session, direction_key=direction_key, rows=[row]
                )
        except Exception as exc:  # noqa: BLE001 — запись одного показателя
            logger.warning(
                "metric_enricher: не записал «%s»", entry.metric_name, exc_info=True
            )
            errors.append(f"{entry.metric_name}: {exc}"[:200])
            return
        if row.status == STATUS_UNKNOWN:
            unknown += 1
        else:
            found += 1

    await asyncio.gather(*(_one(e) for e in entries), return_exceptions=True)
    return {"found": found, "unknown": unknown, "errors": errors}


async def finalize(state: Any, config: RunnableConfig) -> dict:
    pending = state.get("pending") or []
    found = int(state.get("found") or 0)
    unknown = int(state.get("unknown") or 0)
    errors = state.get("errors") or []
    if not pending and not errors:
        report = "Все показатели направления уже с трактовками — обогащать нечего."
    else:
        parts = [f"Взято в работу: {len(pending)}", f"найдено трактовок: {found}"]
        if unknown:
            parts.append(f"в базе знаний нет: {unknown}")
        if errors:
            parts.append(f"с ошибками: {len(errors)}")
        report = "; ".join(parts) + "."
    return {"report": report, "messages": [AIMessage(content=report)]}


__all__ = ["after_load", "enrich", "finalize", "load_catalog"]
