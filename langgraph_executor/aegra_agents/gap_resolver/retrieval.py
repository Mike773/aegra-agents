"""Чтение исходных документов направления (``source_chunk``) для разбора пробелов.

Векторного отбора здесь больше нет: у чанков нет эмбеддинга (миграция 0004), и
отбирать «похожие» нечем и незачем. Отдаём чанки направления как есть, а решает,
есть ли в них ответ, LLM-судья — по каждому чанку отдельно.

Порядок детерминированный (документ, затем позиция в нём): при равных условиях
прогон должен вести себя одинаково от запуска к запуску.
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


async def load_source_chunks(
    session: AsyncSession,
    *,
    direction_key: str,
    limit: int = 400,
) -> list[SourceMatch]:
    """Чанки исходных документов направления, до ``limit`` штук.

    ``limit`` — не отбор по релевантности, а предохранитель: судья вызывается на
    каждый чанк, поэтому размер выборки прямо задаёт стоимость разбора одного
    пробела.
    """
    if limit <= 0 or not direction_key:
        return []
    stmt = (
        select(SourceChunk, SourceDoc.uri)
        .join(SourceDoc, SourceDoc.id == SourceChunk.doc_id)
        .where(SourceChunk.direction_key == direction_key)
        .order_by(SourceDoc.uri, SourceChunk.ord)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [
        SourceMatch(
            chunk_id=ch.id,
            doc_id=ch.doc_id,
            uri=uri,
            ord=ch.ord,
            text=ch.text,
        )
        for ch, uri in rows
    ]


__all__ = ["SourceMatch", "load_source_chunks"]
