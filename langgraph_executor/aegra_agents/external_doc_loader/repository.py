"""Доступ к ``wiki_rag.source_doc`` для external_doc_loader.

Функции принимают готовую ``AsyncSession`` (её открывает узел графа через общий
``easyrag.db.session_scope``) — отдельный engine/psql не заводим. Все выборки
жёстко ограничены ``direction_key``: документ никогда не сравнивается и не
конфликтует с документами чужого направления.
"""
from __future__ import annotations

from typing import NamedTuple
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..easyrag.models import SourceDoc


class ExistingDoc(NamedTuple):
    id: str
    uri: str
    sha256: str | None
    content: str


async def load_direction_docs(
    session: AsyncSession, *, direction_key: str
) -> list[ExistingDoc]:
    """Все документы направления — база для сравнения новых поступлений."""
    rows = (
        await session.execute(
            select(SourceDoc.id, SourceDoc.uri, SourceDoc.sha256, SourceDoc.content)
            .where(SourceDoc.direction_key == direction_key)
            .order_by(SourceDoc.ingested_at)
        )
    ).all()
    return [ExistingDoc(str(r.id), r.uri, r.sha256, r.content) for r in rows]


async def insert_pending_doc(
    session: AsyncSession, *, direction_key: str, uri: str, content: str, sha256: str
) -> str:
    """Записать документ как pending (``processed_at IS NULL``) — индексацию
    сделает wiki_ingest. Возвращает id нового документа."""
    doc = SourceDoc(
        direction_key=direction_key,
        uri=uri,
        mime="text/plain",
        content=content,
        sha256=sha256,
    )
    session.add(doc)
    await session.flush()  # получить doc.id до commit
    return str(doc.id)


async def delete_doc(session: AsyncSession, *, doc_id: str) -> None:
    """Удалить документ; каскад (FK ON DELETE CASCADE) снесёт его
    source_chunk/entity_candidate. wiki_page/wiki_section не трогаются."""
    doc = await session.get(SourceDoc, UUID(doc_id))
    if doc is not None:
        await session.delete(doc)


__all__ = ["ExistingDoc", "delete_doc", "insert_pending_doc", "load_direction_docs"]
