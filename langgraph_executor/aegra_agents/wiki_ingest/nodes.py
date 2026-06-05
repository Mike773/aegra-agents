"""Узлы графа wiki_ingest.

``load_pending`` выбирает необработанные документы направления; ``process``
прогоняет каждый через :func:`ingest_one_document` в отдельной транзакции
(один битый документ не валит весь прогон); ``finalize`` собирает сводку.
"""
from __future__ import annotations

import logging
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from sqlalchemy import select

from ..easyrag.db import session_scope
from ..easyrag.models import SourceDoc
from .config import get_settings
from .embeddings import EmbeddingClient
from .llm import LLMClient
from .pipeline import ingest_one_document
from .state import WikiIngestState

logger = logging.getLogger(__name__)


def _resolve_direction_key(
    state: WikiIngestState, config: RunnableConfig | None
) -> str:
    """direction_key из стейта, а при отсутствии — из config.configurable."""
    direction_key = (state.get("direction_key") or "").strip()
    if direction_key:
        return direction_key
    configurable = (config or {}).get("configurable") or {}
    return (configurable.get("direction_key") or "").strip()


async def load_pending(state: WikiIngestState, config: RunnableConfig) -> dict:
    direction_key = _resolve_direction_key(state, config)
    if not direction_key:
        return {
            "pending_doc_ids": [],
            "errors": [{"error": "direction_key required"}],
        }
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(SourceDoc.id)
                .where(
                    SourceDoc.direction_key == direction_key,
                    SourceDoc.processed_at.is_(None),
                )
                .order_by(SourceDoc.ingested_at)
            )
        ).scalars().all()
    # Возвращаем разрешённый ключ в стейт, чтобы process его увидел,
    # даже если он пришёл только через configurable.
    return {
        "direction_key": direction_key,
        "pending_doc_ids": [str(r) for r in rows],
    }


async def process(state: WikiIngestState, config: RunnableConfig) -> dict:
    direction_key = _resolve_direction_key(state, config)
    pending = state.get("pending_doc_ids") or []

    # Клиенты строим один раз на прогон.
    llm = LLMClient()
    embedder = EmbeddingClient()
    settings = get_settings()

    processed: list[str] = []
    created: list[str] = []
    merged: list[str] = []
    relinked: list[str] = []
    stubs: list[str] = []
    errors: list[dict] = list(state.get("errors") or [])
    ambiguous_total = 0

    for doc_id in pending:
        try:
            async with session_scope() as session:
                doc = await session.get(SourceDoc, UUID(doc_id))
                if doc is None or doc.processed_at is not None:
                    continue
                if doc.direction_key != direction_key:
                    continue
                result = await ingest_one_document(
                    session,
                    doc,
                    llm=llm,
                    embeddings=embedder,
                    settings=settings,
                )
            processed.append(doc_id)
            created.extend(result.created_pages)
            merged.extend(result.merged_pages)
            relinked.extend(result.relinked_pages)
            stubs.extend(result.created_stub_pages)
            ambiguous_total += result.ambiguous_candidate_count
        except Exception as exc:  # один битый документ не валит весь прогон
            logger.exception("wiki_ingest: ошибка обработки doc_id=%s", doc_id)
            errors.append({"doc_id": doc_id, "error": str(exc)})

    return {
        "processed_doc_ids": processed,
        "created_pages": _dedup(created),
        "merged_pages": _dedup(merged),
        "relinked_pages": _dedup(relinked),
        "created_stub_pages": _dedup(stubs),
        "ambiguous_count": ambiguous_total,
        "errors": errors,
    }


async def finalize(state: WikiIngestState) -> dict:
    processed = state.get("processed_doc_ids") or []
    created = state.get("created_pages") or []
    merged = state.get("merged_pages") or []
    relinked = state.get("relinked_pages") or []
    stubs = state.get("created_stub_pages") or []
    errors = state.get("errors") or []
    report = (
        f"Обработано документов: {len(processed)}; "
        f"создано страниц: {len(created)}; "
        f"дополнено: {len(merged)}; "
        f"заглушек создано: {len(stubs)}; "
        f"перелинковано: {len(relinked)}; "
        f"ambiguous-кандидатов: {state.get('ambiguous_count', 0)}; "
        f"ошибок: {len(errors)}."
    )
    return {"report": report}


def after_load(state: WikiIngestState) -> str:
    return "process" if (state.get("pending_doc_ids") or []) else "finalize"


def _dedup(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


__all__ = ["load_pending", "process", "finalize", "after_load"]
