"""Доступ к кэшу трактовок показателей в схеме ``wiki_rag``.

Функции принимают готовую ``AsyncSession`` (её открывает вызывающий через общий
``easyrag.db.session_scope``) — второй engine не заводим.

Здесь же — применение трактовок к базе запуска: они ложатся колонками ``kn_*``
таблицы ``metric``, поэтому доступны и промпту, и SQL-запросам модели.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..easyrag.models import MetricCatalog, MetricKnowledge
from .types import (
    DEFAULT_KIND,
    KNOWLEDGE_VERSION,
    STATUS_UNKNOWN,
    CatalogEntry,
    KnowledgeRow,
)

logger = logging.getLogger(__name__)

# Насколько снижаем доверие, когда трактовка найдена по имени, а не по точному
# ключу: описание показателя изменилось, значит трактовка могла устареть.
NAME_FALLBACK_FACTOR = 0.7
# Как часто перепроверяем показатели, про которые в базе знаний ничего не нашли.
RETRY_UNKNOWN_AFTER = timedelta(days=7)
# Сколько трактовок печатаем в промпт и сколько символов на каждую.
_BLOCK_MAX_ENTRIES = 40
_BLOCK_ENTRY_CAP = 160


def catalog_entries(db: Any, *, limit: int | None = None) -> list[CatalogEntry]:
    """Каталог показателей базы запуска — вход для поиска трактовок."""
    sql = (
        "SELECT metric_key, name_key, name, description, ext_id, direction, unit, "
        "parent_name, depth FROM metric ORDER BY depth, name"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [
        CatalogEntry(
            metric_key=r["metric_key"],
            name_key=r["name_key"],
            metric_name=r["name"],
            metric_description=r["description"] or "",
            metric_ext_id=r["ext_id"],
            direction=r["direction"],
            unit=r["unit"],
            parent_name=r["parent_name"],
            depth=r["depth"],
        )
        for r in db.conn.execute(sql)
    ]


async def upsert_catalog(
    session: AsyncSession, *, direction_key: str, entries: Iterable[CatalogEntry]
) -> int:
    """Отмечает, что показатели видели: их дозаполнит фоновый обогатитель."""
    count = 0
    for entry in entries:
        stmt = insert(MetricCatalog).values(
            direction_key=direction_key,
            metric_key=entry.metric_key,
            name_key=entry.name_key,
            metric_name=entry.metric_name,
            metric_description=entry.metric_description,
            metric_ext_id=entry.metric_ext_id,
            direction=entry.direction,
            unit=entry.unit,
            parent_name=entry.parent_name,
            depth=entry.depth,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[MetricCatalog.direction_key, MetricCatalog.metric_key],
            set_={
                "last_seen_at": datetime.now(timezone.utc),
                "seen_count": MetricCatalog.seen_count + 1,
                "metric_description": entry.metric_description,
            },
        )
        await session.execute(stmt)
        count += 1
    await session.flush()
    return count


def _row_from_model(model: Any) -> KnowledgeRow:
    return KnowledgeRow(
        metric_key=model.metric_key,
        name_key=model.name_key,
        metric_name=model.metric_name,
        metric_description=model.metric_description or "",
        aliases=list(model.aliases or []),
        summary=model.summary or "",
        rules=dict(model.rules or {}),
        status=model.status,
        confidence=float(model.confidence or 0.0),
        source_section_ids=[str(x) for x in (model.source_section_ids or [])],
        model=model.model,
        version=int(model.version or 0),
        updated_at=model.updated_at,
    )


def merge_rows(
    *,
    exact_rows: list[KnowledgeRow],
    name_rows: list[KnowledgeRow],
    metric_keys: list[str],
    name_keys: list[str],
) -> dict[str, KnowledgeRow]:
    """Точное совпадение по ключу побеждает; фолбэк по имени идёт с понижением.

    Ключ содержит описание показателя: правка описания в источнике данных не
    должна обнулять уже найденную трактовку, но и доверять ей полностью нельзя.
    """
    by_key: dict[str, KnowledgeRow] = {r.metric_key: r for r in exact_rows}
    by_name: dict[str, KnowledgeRow] = {}
    for row in name_rows:
        by_name.setdefault(row.name_key, row)

    out: dict[str, KnowledgeRow] = {}
    for metric_key, name_key in zip(metric_keys, name_keys):
        exact = by_key.get(metric_key)
        if exact is not None:
            out[metric_key] = exact
            continue
        fallback = by_name.get(name_key)
        if fallback is not None:
            out[metric_key] = KnowledgeRow(
                **{
                    **fallback.__dict__,
                    "confidence": round(fallback.confidence * NAME_FALLBACK_FACTOR, 4),
                }
            )
    return out


async def get_knowledge(
    session: AsyncSession, *, direction_key: str, entries: list[CatalogEntry]
) -> dict[str, KnowledgeRow]:
    """Известные трактовки по ключу показателя (с фолбэком по имени)."""
    if not entries:
        return {}
    metric_keys = [e.metric_key for e in entries]
    name_keys = [e.name_key for e in entries]
    rows = (
        await session.execute(
            select(MetricKnowledge).where(
                MetricKnowledge.direction_key == direction_key,
                MetricKnowledge.name_key.in_(name_keys),
            )
        )
    ).scalars().all()
    models = [_row_from_model(m) for m in rows]
    exact = [m for m in models if m.metric_key in set(metric_keys)]
    return merge_rows(
        exact_rows=exact, name_rows=models, metric_keys=metric_keys, name_keys=name_keys
    )


def missing_entries(
    catalog: list[CatalogEntry],
    known: dict[str, KnowledgeRow],
    *,
    version: int = KNOWLEDGE_VERSION,
    retry_unknown_after: timedelta = RETRY_UNKNOWN_AFTER,
) -> list[CatalogEntry]:
    """Что стоит поискать: нет трактовки, устарела версия или пора перепроверить.

    Показатели верхнего уровня идут первыми — на них смотрят чаще всего.
    """
    now = datetime.now(timezone.utc)
    out: list[CatalogEntry] = []
    for entry in catalog:
        row = known.get(entry.metric_key)
        if row is None:
            out.append(entry)
            continue
        if int(row.version or 0) < version:
            out.append(entry)
            continue
        if row.status == STATUS_UNKNOWN:
            updated = row.updated_at
            if updated is None or (now - _aware(updated)) >= retry_unknown_after:
                out.append(entry)
    out.sort(key=lambda e: (e.depth or 99, e.metric_name))
    return out


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


async def upsert_knowledge(
    session: AsyncSession, *, direction_key: str, rows: Iterable[KnowledgeRow]
) -> int:
    count = 0
    for row in rows:
        stmt = insert(MetricKnowledge).values(
            direction_key=direction_key,
            metric_key=row.metric_key,
            name_key=row.name_key,
            metric_name=row.metric_name,
            metric_description=row.metric_description,
            aliases=row.aliases,
            summary=row.summary,
            rules=row.rules,
            status=row.status,
            confidence=row.confidence,
            source_section_ids=row.source_section_ids,
            model=row.model,
            version=row.version,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_metric_knowledge",
            set_={
                "aliases": row.aliases,
                "summary": row.summary,
                "rules": row.rules,
                "status": row.status,
                "confidence": row.confidence,
                "source_section_ids": row.source_section_ids,
                "model": row.model,
                "version": row.version,
                "metric_description": row.metric_description,
                "updated_at": datetime.now(timezone.utc),
            },
        )
        await session.execute(stmt)
        count += 1
    await session.flush()
    return count


async def load_catalog_pending(
    session: AsyncSession, *, direction_key: str, limit: int, version: int = KNOWLEDGE_VERSION
) -> list[CatalogEntry]:
    """Показатели направления, которым не хватает трактовки, — для фонового графа."""
    rows = (
        await session.execute(
            select(MetricCatalog)
            .where(MetricCatalog.direction_key == direction_key)
            .order_by(MetricCatalog.depth, MetricCatalog.last_seen_at.desc())
        )
    ).scalars().all()
    catalog = [
        CatalogEntry(
            metric_key=r.metric_key,
            name_key=r.name_key,
            metric_name=r.metric_name,
            metric_description=r.metric_description or "",
            metric_ext_id=r.metric_ext_id,
            direction=r.direction,
            unit=r.unit,
            parent_name=r.parent_name,
            depth=r.depth,
        )
        for r in rows
    ]
    known = await get_knowledge(session, direction_key=direction_key, entries=catalog)
    return missing_entries(catalog, known, version=version)[:limit]


def apply_knowledge(db: Any, rows: dict[str, KnowledgeRow]) -> int:
    """Кладёт трактовки в колонки kn_* таблицы metric базы запуска."""
    if not rows:
        return 0
    applied = 0
    writable = getattr(db, "writable", None)
    ctx = writable() if writable else None
    if ctx is not None:
        ctx.__enter__()
    try:
        for metric_key, row in rows.items():
            rules = row.rules or {}
            kind = rules.get("kind")
            db.conn.execute(
                "UPDATE metric SET kn_status = ?, kn_summary = ?, kn_aliases = ?, "
                "kn_cumulative = ?, kn_plan_semantics = ?, kn_fact_semantics = ?, "
                "kn_processes = ?, kn_notes = ?, kn_confidence = ?, "
                "kind = COALESCE(?, kind) WHERE metric_key = ?",
                (
                    row.status,
                    row.summary,
                    ", ".join(row.aliases) if row.aliases else None,
                    1 if rules.get("cumulative") else (0 if "cumulative" in rules else None),
                    rules.get("plan_semantics"),
                    rules.get("fact_semantics"),
                    ", ".join(rules.get("processes") or []) or None,
                    rules.get("notes"),
                    row.confidence,
                    kind if kind in (DEFAULT_KIND, "вклад", "индекс") else None,
                    metric_key,
                ),
            )
            applied += 1
        db.conn.commit()
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return applied


def knowledge_block(db: Any, *, max_entries: int = _BLOCK_MAX_ENTRIES) -> str:
    """Блок справки по показателям для системного промпта."""
    rows = db.conn.execute(
        "SELECT name, kn_summary, kn_aliases, kn_cumulative, kn_plan_semantics "
        "FROM metric WHERE kn_summary IS NOT NULL AND kn_summary <> '' "
        "ORDER BY depth, name LIMIT ?",
        (max_entries,),
    ).fetchall()
    if not rows:
        return ""
    lines = ["СПРАВКА ПО ПОКАЗАТЕЛЯМ (из базы знаний; числа всё равно из данных)"]
    for r in rows:
        bits = [r["kn_summary"]]
        if r["kn_aliases"]:
            bits.append(f"также: {r['kn_aliases']}")
        if r["kn_cumulative"]:
            bits.append("показатель накопительный")
        if r["kn_plan_semantics"]:
            bits.append(f"план: {r['kn_plan_semantics']}")
        text = "; ".join(bits)[:_BLOCK_ENTRY_CAP]
        lines.append(f"- {r['name']} — {text}")
    return "\n".join(lines)


def rules_json(row: KnowledgeRow) -> str:
    return json.dumps(row.rules or {}, ensure_ascii=False)


__all__ = [
    "NAME_FALLBACK_FACTOR",
    "RETRY_UNKNOWN_AFTER",
    "apply_knowledge",
    "catalog_entries",
    "get_knowledge",
    "knowledge_block",
    "load_catalog_pending",
    "merge_rows",
    "missing_entries",
    "rules_json",
    "upsert_catalog",
    "upsert_knowledge",
]
