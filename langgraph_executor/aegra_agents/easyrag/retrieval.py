"""Retrieval по базе знаний: алиасы + вектор по секциям + расширение по графу.

Порт из easyRag/query/retrieval.py с фильтром по ``direction_key`` — выборка
идёт только по секциям того же направления, по которому пришёл вопрос.

Первым работает поиск по алиасам страниц (``wiki_alias``): аббревиатуры вроде
«AHT» в векторе секции растворяются, а модель эмбеддингов с ними и так
справляется плохо. Точное совпадение слова из запроса с алиасом находит страницу
гарантированно; вектор алиаса добирает близкие формы. Попадание в алиас
разворачивается во ВСЕ секции страницы — алиас указывает на страницу целиком.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from .aliases import alias_norm, query_alias_candidates
from .models import WikiAlias, WikiLink, WikiPage, WikiSection

DEFAULT_GRAPH_EXPAND_THRESH = 0.55
# Порог для вектора алиаса. Он высокий не «на всякий случай»: у используемой
# модели эмбеддингов косинус сжат в узкую полосу, и посторонний запрос к любому
# алиасу даёт около 0.78. Настоящее же совпадение даёт 0.96-1.00 (замер на живой
# базе: «что такое КЛБ» → 0.96, «расписание поездов» → 0.78). Порог лежит между.
# Сменится модель — значение меняется здесь или через параметр вызова.
DEFAULT_ALIAS_THRESH = 0.88
# Сколько страниц может дать векторный поиск по алиасам за раз.
_ALIAS_PAGES = 3
# Точное совпадение — это уверенность, а не близость.
_EXACT_SIMILARITY = 1.0


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


def alias_vector_accepted(
    similarity: float, *, best_section: float | None, threshold: float
) -> bool:
    """Пускать ли страницу, найденную ВЕКТОРОМ алиаса (не точным совпадением).

    Два условия. Порог отсекает шум модели — без него один алиас притягивает
    любой вопрос. Сравнение с лучшей секцией защищает от обратного перекоса:
    если по смыслу ближе обычная секция, отвечать должна она, а не страница,
    чей алиас случайно оказался чуть похож.
    """
    if similarity < threshold:
        return False
    if best_section is None:
        return True
    return similarity > best_section


def merge_hits(
    *,
    alias_hits: list[RetrievedSection],
    vector_hits: list[RetrievedSection],
    limit: int,
) -> list[RetrievedSection]:
    """Алиасы впереди, затем векторные попадания; дубликаты секций убираются.

    Алиас — это точное указание на нужную страницу, поэтому его секции идут
    первыми: если руководитель спросил «что такое AHT», определение важнее
    тематически похожего текста.
    """
    out: list[RetrievedSection] = []
    seen: set[Any] = set()
    for item in [*alias_hits, *vector_hits]:
        if len(out) >= limit:
            break
        if item.section_id in seen:
            continue
        seen.add(item.section_id)
        out.append(item)
    return out


async def retrieve_sections(
    session: AsyncSession,
    *,
    direction_key: str,
    query_vec: list[float],
    query: str = "",
    top_k: int = 8,
    graph_expand: bool = True,
    graph_expand_thresh: float = DEFAULT_GRAPH_EXPAND_THRESH,
    alias_thresh: float = DEFAULT_ALIAS_THRESH,
) -> list[RetrievedSection]:
    if top_k <= 0:
        return []

    vector_hits = await _vector_top_k(session, direction_key, query_vec, top_k)
    alias_hits = await _alias_hits(
        session,
        direction_key=direction_key,
        query=query,
        query_vec=query_vec,
        min_similarity=alias_thresh,
        best_section=vector_hits[0].similarity if vector_hits else None,
    )
    base = merge_hits(alias_hits=alias_hits, vector_hits=vector_hits, limit=top_k)
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


async def _alias_hits(
    session: AsyncSession,
    *,
    direction_key: str,
    query: str,
    query_vec: list[float],
    min_similarity: float,
    best_section: float | None = None,
) -> list[RetrievedSection]:
    """Страницы, найденные по алиасам, развёрнутые в свои секции.

    Два пути: точное совпадение нормализованного алиаса со словом (или парой
    слов) из запроса и косинус к вектору алиаса. Точные идут первыми.
    """
    pages: dict[UUID, float] = {}

    candidates = query_alias_candidates(query)
    if candidates:
        rows = (
            await session.execute(
                select(WikiAlias.page_id).where(
                    WikiAlias.direction_key == direction_key,
                    WikiAlias.alias_norm.in_(candidates),
                )
            )
        ).all()
        for (page_id,) in rows:
            pages[page_id] = _EXACT_SIMILARITY

    if query_vec:
        distance = WikiAlias.embedding.cosine_distance(query_vec).label("distance")
        rows = (
            await session.execute(
                select(WikiAlias.page_id, distance)
                .where(
                    WikiAlias.direction_key == direction_key,
                    WikiAlias.embedding.is_not(None),
                )
                .order_by(distance.asc())
                .limit(_ALIAS_PAGES)
            )
        ).all()
        for page_id, dist in rows:
            similarity = 1.0 - float(dist)
            if page_id in pages:
                continue
            if alias_vector_accepted(
                similarity, best_section=best_section, threshold=min_similarity
            ):
                pages[page_id] = similarity

    if not pages:
        return []
    return await _sections_of_pages(session, pages)


async def _sections_of_pages(
    session: AsyncSession, pages: dict[UUID, float]
) -> list[RetrievedSection]:
    """Все секции найденных страниц: алиас указывает на страницу целиком."""
    rows = (
        await session.execute(
            select(WikiSection)
            .options(joinedload(WikiSection.page))
            .where(WikiSection.page_id.in_(list(pages)))
            .order_by(WikiSection.page_id, WikiSection.ord)
        )
    ).unique().scalars().all()
    out = [
        _row_to_retrieved(sec, 1.0 - pages.get(sec.page_id, 0.0), "alias")
        for sec in rows
    ]
    # Сильнее совпал алиас — выше секции его страницы.
    out.sort(key=lambda r: (-r.similarity, r.slug, r.anchor))
    return out


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


__all__ = [
    "DEFAULT_ALIAS_THRESH",
    "DEFAULT_GRAPH_EXPAND_THRESH",
    "RetrievedSection",
    "alias_vector_accepted",
    "merge_hits",
    "retrieve_sections",
]
