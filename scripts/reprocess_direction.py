"""Переобработка направления в wiki_ingest «с нуля».

Чистит производные данные направления (wiki_page/section/link, entity_candidate,
source_chunk), сбрасывает source_doc.processed_at/domain_brief и заново гоняет
граф wiki_ingest. Нужен, когда поменялся промпт/логика извлечения и существующие
документы надо переиндексировать (sam ingest идемпотентен и повторно их не берёт).

Подключение к БД — через aegra ``easyrag.db.session_scope`` (под aegra-сервером
общий engine, иначе fallback на POSTGRES_DSN из .env). Отдельный engine/psql не
заводим.

Запуск:
    .venv/bin/python scripts/reprocess_direction.py "аналитик"   # одно направление
    .venv/bin/python scripts/reprocess_direction.py --all         # все направления
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, func, select, update  # noqa: E402

from langgraph_executor.aegra_agents.easyrag.db import session_scope  # noqa: E402
from langgraph_executor.aegra_agents.easyrag.models import (  # noqa: E402
    EntityCandidate,
    SourceChunk,
    SourceDoc,
    WikiPage,
)
from langgraph_executor.aegra_agents.wiki_ingest.graph import graph  # noqa: E402


async def _counts(direction_key: str) -> dict:
    async with session_scope() as session:
        async def cnt(model):
            return (
                await session.execute(
                    select(func.count()).select_from(model).where(
                        model.direction_key == direction_key
                    )
                )
            ).scalar_one()

        docs = await cnt(SourceDoc)
        chunks = await cnt(SourceChunk)
        cands = await cnt(EntityCandidate)
        pages = (
            await session.execute(
                select(func.count())
                .select_from(WikiPage)
                .where(WikiPage.direction_key == direction_key)
            )
        ).scalar_one()
        stubs = (
            await session.execute(
                select(func.count())
                .select_from(WikiPage)
                .where(
                    WikiPage.direction_key == direction_key,
                    WikiPage.type == "stub",
                )
            )
        ).scalar_one()
    return {"docs": docs, "chunks": chunks, "cands": cands,
            "pages": pages, "stubs": stubs}


async def _reset(direction_key: str) -> None:
    """Снести производные данные направления и сбросить processed_at."""
    async with session_scope() as session:
        # wiki_page → каскад на wiki_section/wiki_link; entity_candidate.resolved_page_id → SET NULL.
        await session.execute(
            delete(WikiPage).where(WikiPage.direction_key == direction_key)
        )
        # Кандидаты и чанки пересоберутся; source_chunk имеет UNIQUE(doc_id, ord),
        # поэтому старые надо снести, иначе повторная вставка ord=0 конфликтует.
        await session.execute(
            delete(EntityCandidate).where(EntityCandidate.direction_key == direction_key)
        )
        await session.execute(
            delete(SourceChunk).where(SourceChunk.direction_key == direction_key)
        )
        await session.execute(
            update(SourceDoc)
            .where(SourceDoc.direction_key == direction_key)
            .values(processed_at=None, domain_brief=None)
        )


async def _all_directions() -> list[str]:
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(SourceDoc.direction_key).distinct().order_by(SourceDoc.direction_key)
            )
        ).scalars().all()
    return list(rows)


async def _reprocess_one(direction_key: str) -> None:
    print(f"\n=== direction_key = {direction_key!r} ===")
    print("before:", await _counts(direction_key))

    await _reset(direction_key)
    print("after reset:", await _counts(direction_key))

    res = await graph.ainvoke({"direction_key": direction_key})
    print("report:", res.get("report"))
    print("created_pages:", res.get("created_pages"))
    print("created_stub_pages:", res.get("created_stub_pages"))
    print("errors:", res.get("errors"))

    print("after ingest:", await _counts(direction_key))


async def main(target: str) -> int:
    if target == "--all":
        directions = await _all_directions()
        print(f"directions to reprocess: {directions}")
    else:
        directions = [target]
    for direction_key in directions:
        await _reprocess_one(direction_key)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: reprocess_direction.py <direction_key>|--all", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1])))
