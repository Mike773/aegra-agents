"""Оркестратор ingest-пайплайна для одного документа.

Адаптация easyRag ``ingest_text`` под aegra: на вход — уже-персистентный
``SourceDoc`` (его сырой текст в ``doc.content``). За один проход:

1. ``analyze_document`` по началу документа → ``domain_brief`` (контекст для extraction).
2. ``chunk_text`` → эмбеддинг чанков → запись ``source_chunk``.
3. По каждому чанку ``extract_entities`` → эмбеддинг кандидата по чистому
   ``name`` → запись ``entity_candidate``.
4. ``resolve_candidates`` свежих кандидатов → wiki-страницы + ``section_provenance``.
5. ``backfill_links`` (если включён) — back-link на свежие сущности.
6. Простановка ``doc.processed_at`` — идемпотентность.

Отличия от easyRag: убраны sha-дедуп и создание ``SourceDoc`` (документ уже в
БД); ``direction_key`` берётся из самого документа и протягивается во все записи.
Транзакцию (commit/rollback) ведёт вызывающий код (узел графа).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ..easyrag.models import EntityCandidate, SourceChunk, SourceDoc
from .backlinker import backfill_links
from .chunker import Chunk, chunk_text
from .config import Settings, get_settings
from .embeddings import EmbeddingClient, get_embeddings
from .extractor import (
    DocumentBrief,
    ExtractedEntity,
    analyze_document,
    extract_entities,
)
from .llm import LLMClient, get_llm
from .markdown import make_slug
from .merge_utils import embed_batched
from .repository import ensure_stub_page
from .resolver import resolve_candidates


@dataclass(frozen=True)
class IngestResult:
    doc_id: UUID
    chunk_count: int
    entity_count: int
    created_pages: tuple[str, ...] = field(default_factory=tuple)
    merged_pages: tuple[str, ...] = field(default_factory=tuple)
    ambiguous_candidate_count: int = 0
    resolved_candidate_count: int = 0
    relinked_pages: tuple[str, ...] = field(default_factory=tuple)
    created_stub_pages: tuple[str, ...] = field(default_factory=tuple)
    domain_brief: DocumentBrief | None = None


async def ingest_one_document(
    session: AsyncSession,
    doc: SourceDoc,
    *,
    llm: LLMClient | None = None,
    embeddings: EmbeddingClient | None = None,
    settings: Settings | None = None,
) -> IngestResult:
    """Прогнать один уже-персистентный ``SourceDoc`` через пайплайн.

    Читает ``doc.content`` / ``doc.uri`` / ``doc.direction_key``; в конце ставит
    ``doc.processed_at``. Если текст пустой — просто помечает документ обработанным.
    """
    llm_client = llm or get_llm()
    embedder = embeddings or get_embeddings()
    cfg = settings or get_settings()

    direction_key = doc.direction_key
    uri = doc.uri
    text = doc.content or ""

    if not text.strip():
        doc.processed_at = datetime.now(timezone.utc)
        await session.flush()
        return IngestResult(doc_id=doc.id, chunk_count=0, entity_count=0)

    # Шаг 1: domain brief.
    domain_brief: DocumentBrief | None = None
    if cfg.doc_brief_window > 0:
        domain_brief = await analyze_document(
            text[: cfg.doc_brief_window], source_hint=uri, llm=llm_client
        )
        if domain_brief is not None:
            doc.domain_brief = _serialize_brief(domain_brief)

    # Шаг 2: чанки + эмбеддинги.
    chunks = chunk_text(
        text,
        target_size=cfg.chunk_target_size,
        max_size=cfg.chunk_max_size,
        overlap=cfg.chunk_overlap,
    )
    chunk_rows = await _persist_chunks(session, doc, chunks, embedder)

    # Шаг 3: извлечение сущностей (+ упоминаний для заглушек) по чанкам.
    entity_total = 0
    candidate_ids: list[UUID] = []
    mention_names: list[str] = []
    for chunk_row, parsed in zip(chunk_rows, chunks):
        extraction = await extract_entities(
            parsed.text,
            source_hint=uri,
            domain_brief=domain_brief,
            llm=llm_client,
        )
        mention_names.extend(extraction.mentions)
        if not extraction.entities:
            continue
        new_ids = await _persist_entities(
            session,
            doc=doc,
            chunk_id=chunk_row.id,
            extracted=list(extraction.entities),
            embedder=embedder,
        )
        candidate_ids.extend(new_ids)
        entity_total += len(new_ids)

    await session.flush()

    # Шаг 4: резолв полноценных сущностей в wiki.
    created_pages: tuple[str, ...] = ()
    merged_pages: tuple[str, ...] = ()
    ambiguous_count = 0
    resolved_count = 0
    if candidate_ids:
        summary = await resolve_candidates(
            session,
            candidate_ids,
            direction_key=direction_key,
            llm=llm_client,
            embeddings=embedder,
            settings=cfg,
        )
        created_pages = summary.created_pages
        merged_pages = summary.merged_pages
        ambiguous_count = len(summary.ambiguous_candidate_ids)
        resolved_count = summary.resolved_candidate_count

    # Шаг 5: страницы-заглушки для упомянутых, но не описанных сущностей.
    # После резолва (знаем created/merged → не задваиваем) и до back-link
    # (чтобы свежие заглушки попали в exclude и не гонялись relink впустую).
    created_stub_pages: tuple[str, ...] = ()
    if mention_names:
        taken = set(created_pages) | set(merged_pages)
        seen_stub_slugs: set[str] = set()
        stubs: list[str] = []
        for name in mention_names:
            slug = make_slug(name)
            if not slug or slug in taken or slug in seen_stub_slugs:
                continue
            seen_stub_slugs.add(slug)
            page = await ensure_stub_page(
                session, direction_key=direction_key, name=name
            )
            if page is not None:  # None = страница уже существовала в БД
                stubs.append(page.slug)
        created_stub_pages = tuple(stubs)

    # Шаг 6: back-link (триггерится только свежими полноценными страницами).
    relinked_pages: tuple[str, ...] = ()
    fresh = set(created_pages) | set(merged_pages)
    if cfg.backlink_enabled and fresh:
        # 6a. Долинковать СТАРЫЕ страницы направления на свежие сущности/заглушки.
        backfill = await backfill_links(
            session,
            direction_key=direction_key,
            exclude_slugs=fresh | set(created_stub_pages),
            llm=llm_client,
            embeddings=embedder,
            settings=cfg,
        )
        # 6b. Проставить ссылки на самих свежих страницах — в т.ч. на заглушки
        #     этого раунда (иначе они остаются orphan). prefilter=off: substring
        #     не ловит склонённые формы («бабушка» vs «бабушки»), полагаемся на LLM.
        relink_fresh = await backfill_links(
            session,
            direction_key=direction_key,
            only_slugs=fresh,
            prefilter=False,
            llm=llm_client,
            embeddings=embedder,
            settings=cfg,
        )
        relinked_pages = tuple(
            dict.fromkeys((*backfill.relinked, *relink_fresh.relinked))
        )

    doc.processed_at = datetime.now(timezone.utc)
    await session.flush()

    return IngestResult(
        doc_id=doc.id,
        chunk_count=len(chunk_rows),
        entity_count=entity_total,
        created_pages=created_pages,
        merged_pages=merged_pages,
        ambiguous_candidate_count=ambiguous_count,
        resolved_candidate_count=resolved_count,
        relinked_pages=relinked_pages,
        created_stub_pages=created_stub_pages,
        domain_brief=domain_brief,
    )


async def _persist_chunks(
    session: AsyncSession,
    doc: SourceDoc,
    chunks: list[Chunk],
    embedder: EmbeddingClient,
) -> list[SourceChunk]:
    if not chunks:
        return []
    vectors = await embed_batched(embedder, [c.text for c in chunks])
    rows: list[SourceChunk] = []
    for chunk, vec in zip(chunks, vectors):
        row = SourceChunk(
            doc_id=doc.id,
            direction_key=doc.direction_key,
            ord=chunk.ord,
            text=chunk.text,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            embedding=vec,
        )
        session.add(row)
        rows.append(row)
    await session.flush()
    return rows


async def _persist_entities(
    session: AsyncSession,
    *,
    doc: SourceDoc,
    chunk_id: UUID,
    extracted: list[ExtractedEntity],
    embedder: EmbeddingClient,
) -> list[UUID]:
    texts = [_embed_text(e) for e in extracted]
    vectors = await embed_batched(embedder, texts)
    rows: list[EntityCandidate] = []
    for ent, vec in zip(extracted, vectors):
        row = EntityCandidate(
            doc_id=doc.id,
            chunk_id=chunk_id,
            direction_key=doc.direction_key,
            name=ent.name,
            descriptor=ent.descriptor,
            statements=list(ent.statements),
            embedding=vec,
        )
        session.add(row)
        rows.append(row)
    if rows:
        # Нужны id для последующего resolve_candidates(...) — flush заранее.
        await session.flush()
    return [r.id for r in rows]


def _serialize_brief(brief: DocumentBrief) -> str:
    """Сохранить brief в source_doc.domain_brief как JSON-строку."""
    return json.dumps(
        {
            "summary": brief.summary,
            "entity_types": list(brief.entity_types),
        },
        ensure_ascii=False,
    )


def _embed_text(entity: ExtractedEntity) -> str:
    """Текст для эмбеддинга кандидата — ТОЛЬКО имя.

    descriptor шумит и тянет сходство к «похожим по тематике, но разным»
    сущностям; имя сопоставляется с ``wiki_section.embedding``, где уже сидят и
    название страницы, и контент — этого достаточно как сигнала.
    """
    return entity.name


__all__ = ["IngestResult", "ingest_one_document"]
