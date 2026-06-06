"""Доступ к ``query_gap`` для gap_resolver: загрузка нерешённых и пометка решёнными.

Функции принимают готовую ``AsyncSession`` (её открывает узел графа через общий
``easyrag.db.session_scope``) — отдельный engine/psql не заводим.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..easyrag.models import QueryGap


async def load_unresolved_groups(
    session: AsyncSession, *, direction_key: str, limit: int
) -> list[dict]:
    """Нерешённые gap'ы направления (``resolved_at IS NULL``), сгруппированные по
    нормализованному вопросу. До ``limit`` групп, старые первыми.

    Элемент: ``{"id": <repr id>, "query": <текст>, "ids": [<все id дублей вопроса>]}``.
    Группировка нужна, чтобы один и тот же вопрос не искать дважды, но при
    резолве пометить решёнными сразу все его дубли.
    """
    rows = (
        await session.execute(
            select(QueryGap.id, QueryGap.query)
            .where(
                QueryGap.direction_key == direction_key,
                QueryGap.resolved_at.is_(None),
            )
            .order_by(QueryGap.asked_at)
        )
    ).all()
    groups: dict[str, dict] = {}
    for gid, query in rows:
        q = (query or "").strip()
        if not q:
            continue
        key = q.casefold()
        group = groups.get(key)
        if group is None:
            if len(groups) >= limit:
                continue
            groups[key] = {"id": str(gid), "query": q, "ids": [str(gid)]}
        else:
            group["ids"].append(str(gid))
    return list(groups.values())


async def mark_groups_resolved(session: AsyncSession, ids: list[str]) -> None:
    """Проставить ``resolved_at = now`` всем gap'ам с переданными id."""
    if not ids:
        return
    await session.execute(
        update(QueryGap)
        .where(QueryGap.id.in_([UUID(i) for i in ids]))
        .values(resolved_at=datetime.now(timezone.utc))
    )


__all__ = ["load_unresolved_groups", "mark_groups_resolved"]
