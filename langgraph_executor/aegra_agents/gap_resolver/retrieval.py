"""Поиск по исходным документам направления (``source_chunk``) под gap-вопрос.

Зеркалит ``easyrag.retrieval._vector_top_k``, но идёт по ``source_chunk`` (сырой
текст загруженных документов), а не по ``wiki_section``. Нужен, чтобы понять,
есть ли ответ на вопрос в уже загруженных источниках, даже если в wiki его ещё
нет. Фильтр по ``direction_key`` — выборка только в пределах направления.
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..easyrag.models import SourceChunk, SourceDoc


@dataclass(frozen=True)
class SourceMatch:
    chunk_id: UUID
    doc_id: UUID
    uri: str
    ord: int
    text: str
    similarity: float


async def search_source_chunks(
    session: AsyncSession,
    *,
    direction_key: str,
    query_vec: list[float],
    top_k: int = 5,
) -> list[SourceMatch]:
    """Топ-K чанков исходных документов направления по косинусной близости."""
    if top_k <= 0 or not query_vec:
        return []
    distance = SourceChunk.embedding.cosine_distance(query_vec).label("distance")
    stmt = (
        select(SourceChunk, SourceDoc.uri, distance)
        .join(SourceDoc, SourceDoc.id == SourceChunk.doc_id)
        .where(
            SourceChunk.direction_key == direction_key,
            SourceChunk.embedding.is_not(None),
        )
        .order_by(distance.asc())
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).all()
    return [
        SourceMatch(
            chunk_id=ch.id,
            doc_id=ch.doc_id,
            uri=uri,
            ord=ch.ord,
            text=ch.text,
            similarity=1.0 - float(dist),
        )
        for ch, uri, dist in rows
    ]


__all__ = ["SourceMatch", "search_source_chunks"]
