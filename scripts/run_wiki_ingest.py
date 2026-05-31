"""Локальный smoke-прогон подагента wiki_ingest.

Всё идёт через ПОДКЛЮЧЕНИЕ К БД ИЗ aegra: ``easyrag.db.get_engine()`` /
``session_scope()`` (под aegra-сервером — общий ``db_manager.engine``, иначе
fallback на ``POSTGRES_DSN`` из ``.env``). Отдельный ``psql``/DSN не используется.

Шаги:
1. применить миграцию 0002 (идемпотентно, IF NOT EXISTS) через aegra-engine;
2. почистить и вставить тестовый ``source_doc`` для direction_key='test-dir';
3. прогнать граф ``wiki_ingest``;
4. распечатать созданные wiki_page/section/link и проставленный processed_at;
5. повторный прогон — проверка идемпотентности (нечего обрабатывать).

Требует живой Postgres и GIGACHAT_CREDENTIALS в .env.
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

from sqlalchemy import delete, func, select

from langgraph_executor.aegra_agents.easyrag.db import get_engine, session_scope
from langgraph_executor.aegra_agents.easyrag.models import (
    SourceDoc,
    WikiLink,
    WikiPage,
    WikiSection,
)
from langgraph_executor.aegra_agents.wiki_ingest.graph import graph

_ROOT = Path(__file__).resolve().parents[1]
_MIGRATION = _ROOT / "migrations" / "wiki_rag" / "0002_ingest.sql"
_DIRECTION = "test-dir"
_DOC_TEXT = (
    "Колобок укатился от бабушки и дедушки. По дороге Колобок встретил Зайца, "
    "но укатился и от него.\n\n"
    "Потом Колобок повстречал Волка и спел ему песенку. Волк не смог поймать "
    "Колобка.\n\n"
    "Дальше на пути был Медведь, но и Медведю Колобок не дался. "
    "А Лиса перехитрила Колобка и съела его."
)


def _split_sql(text: str) -> list[str]:
    no_comments = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("--")
    )
    return [s.strip() for s in re.split(r";\s*(?:\n|$)", no_comments) if s.strip()]


async def _apply_migration() -> None:
    statements = _split_sql(_MIGRATION.read_text(encoding="utf-8"))
    async with get_engine().begin() as conn:
        for stmt in statements:
            await conn.exec_driver_sql(stmt)
    print(f"migration applied: {_MIGRATION.name} ({len(statements)} stmts)")


async def _reset_and_seed() -> None:
    async with session_scope() as session:
        await session.execute(
            delete(WikiPage).where(WikiPage.direction_key == _DIRECTION)
        )
        await session.execute(
            delete(SourceDoc).where(SourceDoc.direction_key == _DIRECTION)
        )
        session.add(
            SourceDoc(direction_key=_DIRECTION, uri="kolobok.txt", content=_DOC_TEXT)
        )
    print(f"seeded 1 source_doc for direction_key={_DIRECTION!r}")


async def _dump_state(label: str) -> None:
    async with session_scope() as session:
        pages = (
            await session.execute(
                select(WikiPage.slug, WikiPage.type).where(
                    WikiPage.direction_key == _DIRECTION
                )
            )
        ).all()
        full = [s for s, t in pages if t != "stub"]
        stubs = [s for s, t in pages if t == "stub"]
        sec_count = (
            await session.execute(
                select(func.count())
                .select_from(WikiSection)
                .where(WikiSection.direction_key == _DIRECTION)
            )
        ).scalar_one()
        link_count = (
            await session.execute(
                select(func.count())
                .select_from(WikiLink)
                .join(WikiPage, WikiPage.id == WikiLink.from_page_id)
                .where(WikiPage.direction_key == _DIRECTION)
            )
        ).scalar_one()
        unprocessed = (
            await session.execute(
                select(func.count())
                .select_from(SourceDoc)
                .where(
                    SourceDoc.direction_key == _DIRECTION,
                    SourceDoc.processed_at.is_(None),
                )
            )
        ).scalar_one()
    print(f"\n[{label}]")
    print(f"  wiki_page: {len(pages)} (full={len(full)}, stub={len(stubs)})")
    print(f"  full pages: {full}")
    print(f"  stub pages: {stubs}")
    print(f"  wiki_section: {sec_count}; wiki_link: {link_count}")
    print(f"  source_doc unprocessed: {unprocessed}")


async def main() -> int:
    await _apply_migration()
    await _reset_and_seed()

    print("\n=== first run ===")
    res = await graph.ainvoke({"direction_key": _DIRECTION})
    print("  report:", res.get("report"))
    print("  processed_doc_ids:", res.get("processed_doc_ids"))
    print("  created_pages:", res.get("created_pages"))
    print("  errors:", res.get("errors"))
    await _dump_state("after first run")

    print("\n=== second run (idempotency) ===")
    res2 = await graph.ainvoke({"direction_key": _DIRECTION})
    print("  pending_doc_ids:", res2.get("pending_doc_ids"))
    print("  processed_doc_ids:", res2.get("processed_doc_ids"))
    print("  report:", res2.get("report"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
