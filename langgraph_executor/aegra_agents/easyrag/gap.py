"""Запись gap'а — факта обращения к wiki за вопросом, на который ответа нет.

Пустой ``resolved_section_ids`` + ``resolved_at = NULL`` означают «вопрос задан,
ответа в wiki по данному направлению нет» — это и есть сигнал для enrichment-loop'а.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from .models import QueryGap


async def record_gap(
    session: AsyncSession,
    *,
    direction_key: str,
    question: str,
    embedding: list[float] | None,
    resolved_section_ids: tuple[UUID, ...] = (),
) -> None:
    gap = QueryGap(
        query=question,
        embedding=embedding,
        direction_key=direction_key,
        resolved_section_ids=list(resolved_section_ids),
    )
    if resolved_section_ids:
        gap.resolved_at = datetime.now(timezone.utc)
    session.add(gap)
    await session.flush()


__all__ = ["record_gap"]
