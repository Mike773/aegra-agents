"""Общие хелперы для merge-подобных операций над wiki-страницами.

Используется и резолвером (:mod:`wiki_ingest.resolver`), и back-linker'ом
(:mod:`wiki_ingest.backlinker`). Порт из easyRag с фильтрацией каталога по
``direction_key`` — каждая страница видит только соседей своего направления.
"""
from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..easyrag.models import SectionProvenance, WikiPage, WikiSection
from .embeddings import EmbeddingClient

EMBED_BATCH_SIZE = 128


async def load_existing_catalog(
    session: AsyncSession,
    *,
    direction_key: str,
    exclude_slug: str | None,
    limit: int,
) -> list[tuple[str, list[str]]]:
    """Подгрузить каталог существующих страниц направления для подсказки LLM.

    Возвращает ``[(title, aliases), ...]``. Фильтрует по ``direction_key``; если
    задан ``exclude_slug`` — исключает страницу с этим slug'ом (обычно «саму
    себя» при merge'е). При ``limit <= 0`` или пустой wiki вернёт пустой список.
    """
    if limit <= 0:
        return []
    stmt = select(WikiPage.title, WikiPage.aliases).where(
        WikiPage.direction_key == direction_key
    )
    if exclude_slug is not None:
        stmt = stmt.where(WikiPage.slug != exclude_slug)
    stmt = stmt.order_by(WikiPage.updated_at.desc()).limit(limit)
    rows = (await session.execute(stmt)).all()
    return [(t, list(a or [])) for t, a in rows]


def section_embed_text(page: WikiPage, section: WikiSection) -> str:
    """Текст для эмбеддинга секции: ``page_title. section_title. body_md``."""
    parts: list[str] = [page.title]
    if section.title and section.title != page.title:
        parts.append(section.title)
    body = (section.body_md or "").strip()
    if body:
        parts.append(body)
    return ". ".join(parts)


async def embed_batched(
    embedder: EmbeddingClient, texts: list[str]
) -> list[list[float]]:
    """Эмбеддим длинные списки порциями, чтобы не упереться в лимит провайдера."""
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        out.extend(await embedder.embed_many(texts[i : i + EMBED_BATCH_SIZE]))
    return out


async def reembed_sections(
    session: AsyncSession, page: WikiPage, embedder: EmbeddingClient
) -> None:
    """Пересчитать ``wiki_section.embedding`` для всех секций страницы.

    ``upsert_page`` стирает старые секции и вставляет новые без эмбеддингов —
    здесь мы их добиваем, иначе pgvector-поиск перестанет видеть страницу.
    """
    sections = (
        await session.execute(
            select(WikiSection).where(WikiSection.page_id == page.id)
        )
    ).scalars().all()
    if not sections:
        return

    texts = [section_embed_text(page, s) for s in sections]
    vectors = await embed_batched(embedder, texts)
    for sec, vec in zip(sections, vectors):
        sec.embedding = vec
    await session.flush()


async def snapshot_provenance(
    session: AsyncSession, page_id: UUID
) -> list[tuple[str, UUID]]:
    """Снять ``(anchor, source_chunk_id)`` для всех секций страницы.

    Используется, чтобы пережить ``upsert_page`` без потери провенанса.
    """
    rows = (
        await session.execute(
            select(WikiSection.anchor, SectionProvenance.source_chunk_id).join(
                SectionProvenance, SectionProvenance.section_id == WikiSection.id
            ).where(WikiSection.page_id == page_id)
        )
    ).all()
    return [(anchor, chunk_id) for anchor, chunk_id in rows]


async def restore_provenance_by_anchor(
    session: AsyncSession,
    page_id: UUID,
    snapshot: Sequence[tuple[str, UUID]],
) -> int:
    """Восстановить ``section_provenance`` по совпадающим anchor'ам.

    Якори, которых в новой структуре страницы нет, проваливаются — это
    осознанная потеря провенанса (LLM мог переименовать секцию). Возвращает
    количество восстановленных пар.
    """
    if not snapshot:
        return 0
    sections = (
        await session.execute(
            select(WikiSection.id, WikiSection.anchor).where(
                WikiSection.page_id == page_id
            )
        )
    ).all()
    anchor_to_id = {anchor: sid for sid, anchor in sections}

    wanted = {
        (anchor_to_id[anchor], chunk_id)
        for anchor, chunk_id in snapshot
        if anchor in anchor_to_id
    }
    if not wanted:
        return 0
    section_ids = {sid for sid, _ in wanted}
    chunk_ids = {cid for _, cid in wanted}
    existing = (
        await session.execute(
            select(
                SectionProvenance.section_id, SectionProvenance.source_chunk_id
            ).where(
                SectionProvenance.section_id.in_(section_ids),
                SectionProvenance.source_chunk_id.in_(chunk_ids),
            )
        )
    ).all()
    existing_pairs = {(sid, cid) for sid, cid in existing}
    restored = 0
    for sid, cid in wanted - existing_pairs:
        session.add(SectionProvenance(section_id=sid, source_chunk_id=cid))
        restored += 1
    return restored


__all__ = [
    "EMBED_BATCH_SIZE",
    "embed_batched",
    "load_existing_catalog",
    "reembed_sections",
    "restore_provenance_by_anchor",
    "section_embed_text",
    "snapshot_provenance",
]
