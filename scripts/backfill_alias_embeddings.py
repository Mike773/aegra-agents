#!/usr/bin/env python3
"""Дозаливка векторов алиасов для уже существующих страниц вики.

Зачем: алиасы (`wiki_page.aliases`) раньше в поиске не участвовали, поэтому у
страниц, созданных до появления таблицы ``wiki_alias``, векторов нет. Скрипт
проходит по страницам направления и эмбеддит ТОЛЬКО недостающие алиасы —
страницы не перестраиваются, LLM не вызывается, тексты не меняются.

Применить сначала миграцию ``migrations/wiki_rag/0005_wiki_alias.sql``.

Примеры::

    python scripts/backfill_alias_embeddings.py --direction оператор
    python scripts/backfill_alias_embeddings.py --direction оператор --dry-run
    python scripts/backfill_alias_embeddings.py --all
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Без этого пакет берётся из site-packages: editable-установка venv может
# указывать на другой каталог, и скрипт молча прогонит чужой код.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Креды НЕ подменяем: скрипт реально ходит за эмбеддингами, ему нужен .env.

from sqlalchemy import distinct, select  # noqa: E402

from langgraph_executor.aegra_agents.easyrag.aliases import alias_norm  # noqa: E402
from langgraph_executor.aegra_agents.easyrag.db import session_scope  # noqa: E402
from langgraph_executor.aegra_agents.easyrag.models import (  # noqa: E402
    WikiAlias,
    WikiPage,
)
from langgraph_executor.aegra_agents.wiki_ingest.embeddings import (  # noqa: E402
    get_embeddings,
)
from langgraph_executor.aegra_agents.wiki_ingest.merge_utils import (  # noqa: E402
    sync_alias_embeddings,
)


async def _directions() -> list[str]:
    async with session_scope() as session:
        rows = (await session.execute(select(distinct(WikiPage.direction_key)))).all()
    return sorted(r[0] for r in rows if r[0])


async def _report(direction: str) -> tuple[int, int, int]:
    """(страниц с алиасами, всего алиасов, уже с вектором)."""
    async with session_scope() as session:
        pages = (
            await session.execute(
                select(WikiPage).where(WikiPage.direction_key == direction)
            )
        ).scalars().all()
        have = (
            await session.execute(
                select(WikiAlias.page_id, WikiAlias.alias_norm).where(
                    WikiAlias.direction_key == direction
                )
            )
        ).all()
    known = {(pid, norm) for pid, norm in have}
    with_aliases = 0
    total = 0
    embedded = 0
    for page in pages:
        norms = {alias_norm(a) for a in (page.aliases or []) if alias_norm(a)}
        if not norms:
            continue
        with_aliases += 1
        total += len(norms)
        embedded += sum(1 for n in norms if (page.id, n) in known)
    return with_aliases, total, embedded


async def backfill(direction: str, *, dry_run: bool) -> int:
    with_aliases, total, embedded = await _report(direction)
    missing = total - embedded
    print(
        f"[{direction}] страниц с алиасами: {with_aliases}; алиасов: {total}; "
        f"уже с вектором: {embedded}; не хватает: {missing}"
    )
    if dry_run or missing <= 0:
        return 0

    embedder = get_embeddings()
    added = 0
    async with session_scope() as session:
        pages = (
            await session.execute(
                select(WikiPage).where(WikiPage.direction_key == direction)
            )
        ).scalars().all()
        for page in pages:
            if not (page.aliases or []):
                continue
            try:
                added += await sync_alias_embeddings(session, page, embedder)
            except Exception as exc:  # одна страница не валит прогон
                print(f"  ! {page.slug}: {type(exc).__name__}: {exc}")
    print(f"[{direction}] доэмбеддено алиасов: {added}")
    return added


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--direction", help="ключ направления")
    group.add_argument("--all", action="store_true", help="все направления в базе")
    parser.add_argument("--dry-run", action="store_true",
                        help="только показать, сколько не хватает")
    args = parser.parse_args()

    directions = await _directions() if args.all else [args.direction]
    if not directions:
        print("в базе нет страниц вики")
        return 0
    total = 0
    for direction in directions:
        total += await backfill(direction, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"итого доэмбеддено: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
