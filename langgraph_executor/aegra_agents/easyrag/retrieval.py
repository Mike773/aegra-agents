"""Retrieval поверх wiki_section: vector top-K + graph expansion.

Порт из easyRag/query/retrieval.py с фильтром по ``direction_key`` —
выборка идёт только по секциям того же направления, по которому пришёл вопрос.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from .models import WikiLink, WikiPage, WikiSection

DEFAULT_GRAPH_EXPAND_THRESH = 0.55


@dataclass(frozen=True)
class RetrievedSection:
    section_id: UUID
    page_id: UUID
    slug: str
    anchor: str
    page_title: str
    section_title: str
    body_md: str
    similarity: float
    source: str  # "vector" | "graph"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["section_id"] = str(self.section_id)
        d["page_id"] = str(self.page_id)
        return d


async def retrieve_sections(
    session: AsyncSession,
    *,
    direction_key: str,
    query_vec: list[float],
    top_k: int = 8,
    graph_expand: bool = True,
    graph_expand_thresh: float = DEFAULT_GRAPH_EXPAND_THRESH,
) -> list[RetrievedSection]:
    if top_k <= 0:
        return []

    base = await _vector_top_k(session, direction_key, query_vec, top_k)
    if not base:
        return []
    if not graph_expand:
        return base

    seen_section_ids = {r.section_id for r in base}
    seen_page_ids = {r.page_id for r in base}
    extra = await _graph_expand(
        session,
        direction_key=direction_key,
        query_vec=query_vec,
        seed_page_ids=seen_page_ids,
        skip_section_ids=seen_section_ids,
        min_similarity=graph_expand_thresh,
    )
    return base + extra


async def _vector_top_k(
    session: AsyncSession,
    direction_key: str,
    query_vec: list[float],
    top_k: int,
) -> list[RetrievedSection]:
    distance = WikiSection.embedding.cosine_distance(query_vec).label("distance")
    stmt = (
        select(WikiSection, distance)
        .options(joinedload(WikiSection.page))
        .where(
            WikiSection.direction_key == direction_key,
            WikiSection.embedding.is_not(None),
        )
        .order_by(distance.asc())
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).unique().all()
    return [_row_to_retrieved(sec, float(dist), "vector") for sec, dist in rows]


async def _graph_expand(
    session: AsyncSession,
    *,
    direction_key: str,
    query_vec: list[float],
    seed_page_ids: set[UUID],
    skip_section_ids: set[UUID],
    min_similarity: float,
) -> list[RetrievedSection]:
    if not seed_page_ids:
        return []

    link_stmt = select(WikiLink.to_page_id).where(
        WikiLink.from_page_id.in_(seed_page_ids),
        WikiLink.to_page_id.is_not(None),
    )
    target_ids_raw = (await session.execute(link_stmt)).scalars().all()
    target_ids = {t for t in target_ids_raw if t is not None} - seed_page_ids
    if not target_ids:
        return []

    distance = WikiSection.embedding.cosine_distance(query_vec).label("distance")
    stmt = (
        select(WikiSection, distance)
        .options(joinedload(WikiSection.page))
        .where(
            WikiSection.direction_key == direction_key,
            WikiSection.page_id.in_(target_ids),
            WikiSection.embedding.is_not(None),
        )
        .order_by(distance.asc())
    )
    rows = (await session.execute(stmt)).unique().all()

    out: list[RetrievedSection] = []
    for sec, dist in rows:
        if sec.id in skip_section_ids:
            continue
        sim = 1.0 - float(dist)
        if sim < min_similarity:
            continue
        out.append(_row_to_retrieved(sec, float(dist), "graph"))
    return out


def _row_to_retrieved(sec: WikiSection, distance: float, source: str) -> RetrievedSection:
    page: WikiPage = sec.page
    return RetrievedSection(
        section_id=sec.id,
        page_id=sec.page_id,
        slug=page.slug,
        anchor=sec.anchor,
        page_title=page.title,
        section_title=sec.title,
        body_md=sec.body_md,
        similarity=1.0 - distance,
        source=source,
    )


__all__ = ["RetrievedSection", "retrieve_sections", "DEFAULT_GRAPH_EXPAND_THRESH"]
