"""Смоук gap_resolver на throwaway-направлении test-dir (БД из aegra).

Сеет 2 нерешённых gap'а (один отвечается из исходника «Колобок», один — нет),
гоняет граф, печатает отчёт и проверяет, что найденный помечен resolved_at, а
ненайденный — нет. В конце подчищает за собой (свои gap'ы + созданные заглушки).

Требует живой Postgres (POSTGRES_DSN) и GIGACHAT_CREDENTIALS в .env. Предполагает,
что в test-dir уже есть проиндексированный «Колобок» (scripts/run_wiki_ingest.py).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, select  # noqa: E402

from langgraph_executor.aegra_agents.easyrag.db import session_scope  # noqa: E402
from langgraph_executor.aegra_agents.easyrag.models import (  # noqa: E402
    QueryGap,
    WikiPage,
)
from langgraph_executor.aegra_agents.gap_resolver.graph import graph  # noqa: E402
from langgraph_executor.aegra_agents.wiki_ingest.embeddings import (  # noqa: E402
    EmbeddingClient,
)

_DIRECTION = "test-dir"
_ANSWERABLE = "Кто съел Колобка?"
_MISSING = "На каком автомобиле ездил Колобок?"


async def _embed_retry(embedder: EmbeddingClient, text: str) -> list[float]:
    """Эмбеддинг с ретраем на 429 GigaChat (PERS-тариф любит троттлить)."""
    last: Exception | None = None
    for attempt in range(6):
        try:
            return await embedder.embed_one(text)
        except Exception as exc:  # noqa: BLE001
            last = exc
            await asyncio.sleep(5 * (attempt + 1))
    raise last  # type: ignore[misc]


async def _seed() -> list[str]:
    embedder = EmbeddingClient()
    ids: list[str] = []
    async with session_scope() as session:
        for q in (_ANSWERABLE, _MISSING):
            vec = await _embed_retry(embedder, q)
            gap = QueryGap(query=q, embedding=vec, direction_key=_DIRECTION)
            session.add(gap)
            await session.flush()
            ids.append(str(gap.id))
    print(f"seeded gaps: {ids}")
    return ids


async def _resolved_map(ids: list[str]) -> dict[str, bool]:
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(QueryGap.id, QueryGap.query, QueryGap.resolved_at).where(
                    QueryGap.id.in_([UUID(i) for i in ids])
                )
            )
        ).all()
    return {str(i): (q, ra is not None) for i, q, ra in rows}


async def _cleanup(ids: list[str], stub_slugs: list[str]) -> None:
    async with session_scope() as session:
        await session.execute(
            delete(QueryGap).where(QueryGap.id.in_([UUID(i) for i in ids]))
        )
        if stub_slugs:
            await session.execute(
                delete(WikiPage).where(
                    WikiPage.direction_key == _DIRECTION,
                    WikiPage.slug.in_(stub_slugs),
                    WikiPage.type == "stub",
                )
            )
    print(f"cleaned up gaps={len(ids)} stubs={stub_slugs}")


async def main() -> int:
    ids = await _seed()
    res: dict = {}
    try:
        res = await graph.ainvoke({"direction_key": _DIRECTION})
        print("\n=== REPORT ===")
        print(res.get("report"))
        print("\nresolved:", [r["query"] for r in (res.get("resolved") or [])])
        print("unresolved:", [u["query"] for u in (res.get("unresolved") or [])])
        stubs = res.get("created_stub_pages") or []
        print("created_stub_pages:", stubs)

        print("\n=== resolved_at in DB ===")
        rmap = await _resolved_map(ids)
        for gid, (q, is_res) in rmap.items():
            print(f"  {'RESOLVED' if is_res else 'open    '}  {q!r}")

        ok = True
        # Найденный (answerable) должен быть resolved, missing — нет.
        for gid in ids:
            q, is_res = rmap.get(gid, ("?", False))
            if q == _ANSWERABLE and not is_res:
                ok = False
                print("FAIL: answerable gap not marked resolved")
            if q == _MISSING and is_res:
                ok = False
                print("FAIL: missing gap wrongly marked resolved")
        print("\nSMOKE:", "OK" if ok else "FAILED")
        return 0 if ok else 1
    finally:
        await _cleanup(ids, res.get("created_stub_pages") or [])


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
