"""Операции над ``wiki_rag.source_doc`` для doc_manager.

Каждая функция открывает свою сессию через ``easyrag.db.session_scope`` (auto
commit/rollback). Загрузка пишет документ как pending (``processed_at IS NULL``) —
индексацию делает подагент ``wiki_ingest``. Удаление каскадит на
``source_chunk``/``entity_candidate`` (FK ``ON DELETE CASCADE``).
"""
from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID

from sqlalchemy import func, select

from ..easyrag.db import session_scope
from ..easyrag.models import SourceChunk, SourceDoc


def _row_to_dict(row: Any) -> dict:
    return {
        "id": str(row.id),
        "uri": row.uri,
        "mime": row.mime,
        "processed_at": row.processed_at.isoformat() if row.processed_at else None,
        "ingested_at": row.ingested_at.isoformat() if row.ingested_at else None,
        "chunks": int(row.chunks or 0),
    }


async def insert_pending_doc(
    *,
    direction_key: str,
    uri: str,
    content: str,
    mime: str = "text/markdown",
    dedup: bool = True,
) -> dict:
    """Сохранить документ как pending. Опциональная мягкая дедуп по sha256.

    Возвращает {"status": "created"|"duplicate", "id", "uri", ...}.
    """
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    async with session_scope() as session:
        if dedup:
            existing = (
                await session.execute(
                    select(
                        SourceDoc.id, SourceDoc.uri, SourceDoc.processed_at
                    ).where(
                        SourceDoc.direction_key == direction_key,
                        SourceDoc.sha256 == sha,
                    )
                )
            ).first()
            if existing is not None:
                return {
                    "status": "duplicate",
                    "id": str(existing.id),
                    "uri": existing.uri,
                    "processed": existing.processed_at is not None,
                }
        doc = SourceDoc(
            direction_key=direction_key,
            uri=uri,
            mime=mime,
            content=content,
            sha256=sha,
            # processed_at оставляем NULL — триггер обработки для wiki_ingest.
        )
        session.add(doc)
        await session.flush()  # получить doc.id до commit
        return {"status": "created", "id": str(doc.id), "uri": uri}


async def list_docs(direction_key: str) -> list[dict]:
    """Документы направления с числом чанков, новые первыми."""
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(
                    SourceDoc.id,
                    SourceDoc.uri,
                    SourceDoc.mime,
                    SourceDoc.processed_at,
                    SourceDoc.ingested_at,
                    func.count(SourceChunk.id).label("chunks"),
                )
                .outerjoin(SourceChunk, SourceChunk.doc_id == SourceDoc.id)
                .where(SourceDoc.direction_key == direction_key)
                .group_by(SourceDoc.id)
                .order_by(SourceDoc.ingested_at.desc())
            )
        ).all()
    return [_row_to_dict(r) for r in rows]


async def delete_doc(*, direction_key: str, doc_id: str) -> dict:
    """Удалить документ с проверкой изоляции по direction_key.

    Возвращает {"status": "deleted"|"not_found", ...}. На deleted каскад убирает
    source_chunk/entity_candidate; wiki_page/wiki_section НЕ трогаются.
    """
    try:
        pk = UUID(doc_id)
    except (ValueError, AttributeError):
        return {"status": "not_found"}
    async with session_scope() as session:
        doc = await session.get(SourceDoc, pk)
        if doc is None or doc.direction_key != direction_key:
            return {"status": "not_found"}
        result = {
            "status": "deleted",
            "id": doc_id,
            "uri": doc.uri,
            "was_processed": doc.processed_at is not None,
        }
        await session.delete(doc)
        return result


__all__ = ["insert_pending_doc", "list_docs", "delete_doc"]
