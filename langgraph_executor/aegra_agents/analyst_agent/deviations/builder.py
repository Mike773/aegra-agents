"""Построение карты отклонений: детерминированно, без LLM.

Карта живёт весь диалог (в стейте через чекпойнтер) и каждый ход
материализуется в таблицу ``deviation``, чтобы модель могла ходить по ней
обычным SQL. Записи, отмеченные самой моделью (``source='agent'``), переживают
пересборку; авто-записи, которых больше нет в данных, помечаются resolved.

Идентификатор записи стабилен между ходами: sha1 от вида, человека, показателя
и разреза. Поэтому заметка модели остаётся привязанной к тому же отклонению,
даже когда цифры обновились.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

from ..db.analytics import direction_better
from . import rules

# Сколько ходов держим отклонение, которого больше нет в данных.
_RESOLVED_TTL_TURNS = 2

_CANDIDATE_SQL = """
SELECT v.person_key, v.metric, v.ext_id, v.element, v.date, v.depth, v.root_name,
       v.direction, v.kind, v.is_star, v.is_star_metric, v.star_of,
       v.fact, v.plan, v.star_received,
       v.plan_dev_pct, v.plan_status, v.pop_change_pct, v.pop_status,
       v.trend_status, v.zscore, v.is_anomaly, v.peer_status, v.peer_percentile,
       v.rel_status, v.group_change_pct, v.benchmark_dev_pct, v.benchmark_status,
       v.influent_percent,
       (SELECT COUNT(*) FROM fact f
         JOIN metric m ON m.metric_id = f.metric_id
        WHERE m.name = v.metric AND f.plan IS NOT NULL AND f.plan <> 0) > 0 AS has_plan,
       (SELECT t.share_pct FROM v_tree t WHERE t.child = v.metric LIMIT 1) AS share_pct
FROM v_fact_latest v
WHERE v.person_key = :person_key
"""


def _dev_id(kind: str, person_key: str, metric: str, element: str | None) -> str:
    raw = f"{kind}|{person_key}|{metric}|{element or ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _persistent(conn, person_key: str, metric: str, element: str | None) -> bool:
    """Ухудшение подтверждено минимум тремя периодами подряд."""
    rows = conn.execute(
        "SELECT pop_status FROM v_fact WHERE person_key = ? AND metric = ? "
        "AND COALESCE(element, '') = COALESCE(?, '') ORDER BY period_idx DESC LIMIT 3",
        (person_key, metric, element),
    ).fetchall()
    statuses = [r["pop_status"] for r in rows]
    return len(statuses) >= 3 and all(s == "ухудшение" for s in statuses)


def _group_worsening(row: dict[str, Any]) -> bool:
    change = row.get("group_change_pct")
    if change is None:
        return False
    return not direction_better(change > 0, row.get("direction"))


def _make(
    row: dict[str, Any],
    kind: str,
    polarity: str,
    *,
    share_pct: float | None,
    turn: int,
) -> dict[str, Any]:
    return {
        "id": _dev_id(kind, row["person_key"], row["metric"], row["element"]),
        "kind": kind,
        "polarity": polarity,
        "metric_name": row["metric"],
        "metric_ext_id": row["ext_id"],
        "person_key": row["person_key"],
        "element": row["element"],
        "date": row["date"],
        "depth": row["depth"],
        "root_name": row["root_name"],
        "signal": rules.signal(row),
        "priority": rules.priority(row, share_pct),
        "fact": row["fact"],
        "plan": row["plan"],
        "plan_dev_pct": row["plan_dev_pct"],
        "pop_change_pct": row["pop_change_pct"],
        "zscore": row["zscore"],
        "percentile": row["peer_percentile"],
        "benchmark_dev_pct": row["benchmark_dev_pct"],
        "influent_share": share_pct,
        "source": "auto",
        "status": "open",
        "task_id": None,
        "task_target": None,
        "task_dev_pct": None,
        "note": None,
        "turn_created": turn,
        "turn_updated": turn,
    }


def _kinds_for(row: dict[str, Any]) -> list[tuple[str, str]]:
    """Виды отклонений строки: (kind, polarity)."""
    out: list[tuple[str, str]] = []
    is_index = row.get("kind") == "индекс"

    if row.get("star_received") is not None:
        out.append(
            ("star_received", "positive") if row["star_received"] else ("star_missed", "negative")
        )
        return out

    if not is_index:
        if row.get("plan_status") == "хуже_плана":
            out.append(("below_plan", "negative"))
            if row.get("is_star_metric"):
                out.append(("star_at_risk", "negative"))
        elif row.get("plan_status") == "лучше_плана":
            out.append(("above_plan", "positive"))

        if row.get("pop_status") == "ухудшение" or row.get("trend_status") == "ухудшение":
            out.append(("declining", "negative"))
        elif row.get("pop_status") == "улучшение":
            out.append(("improving", "positive"))

        if row.get("benchmark_status") == "хуже_бенчмарка":
            out.append(("benchmark_gap", "negative"))
        if row.get("rel_status") == "хуже_группы":
            out.append(("group_worse", "negative"))

    if row.get("is_anomaly"):
        out.append(("anomaly", "negative"))
    if row.get("peer_status") == "хуже_коллег":
        out.append(("peer_worse", "negative"))
    elif row.get("peer_status") == "лучше_коллег" and not is_index:
        out.append(("peer_better", "positive"))
    return out


def _aggregate_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["metric"]: r for r in rows if r["element"] is None}


def build_deviations(
    db: Any,
    *,
    focus_person_key: str,
    tasks: Iterable[dict[str, Any]] | None = None,
    previous: list[dict[str, Any]] | None = None,
    turn: int = 1,
    max_auto: int = 40,
) -> list[dict[str, Any]]:
    """Пересобирает карту отклонений фокус-персоны и сливает её с предыдущей."""
    conn = db.conn
    rows = [
        dict(r) for r in conn.execute(_CANDIDATE_SQL, {"person_key": focus_person_key})
    ]
    aggregates = _aggregate_index(rows)

    task_metrics: set[str] = set()
    task_by_metric: dict[str, dict[str, Any]] = {}
    for task in tasks or []:
        for name in task.get("metric_names") or []:
            task_metrics.add(name)
            task_by_metric[name] = task

    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        row["persistent"] = _persistent(
            conn, row["person_key"], row["metric"], row["element"]
        )
        row["group_worsening"] = _group_worsening(row)

        if row["element"] is not None:
            agg = aggregates.get(row["metric"])
            agg_flagged = bool(agg and agg.get("plan_status") == "хуже_плана")
            if not rules.element_is_material(
                row.get("plan_dev_pct"),
                agg.get("plan_dev_pct") if agg else None,
                agg_flagged,
            ):
                continue

        share_pct = row.get("share_pct")
        for kind, polarity in _kinds_for(row):
            dev = _make(row, kind, polarity, share_pct=share_pct, turn=turn)
            if row["element"] is not None and kind == "below_plan":
                dev["kind"] = "segment_worst"
                dev["id"] = _dev_id(
                    "segment_worst", row["person_key"], row["metric"], row["element"]
                )
            _apply_task_focus(dev, row, task_metrics, task_by_metric, bool(task_metrics))
            found[dev["id"]] = dev

    merged = _merge(found, previous or [], turn)
    auto = [d for d in merged if d["source"] == "auto" and d["status"] == "open"]
    other = [d for d in merged if d not in auto]
    auto.sort(key=lambda d: (-d["priority"], -(d.get("signal") or 0), d["metric_name"]))
    kept = auto[:max_auto]
    result = kept + other
    result.sort(key=lambda d: (-d["priority"], d["metric_name"]))
    return result


def _apply_task_focus(
    dev: dict[str, Any],
    row: dict[str, Any],
    task_metrics: set[str],
    task_by_metric: dict[str, dict[str, Any]],
    has_tasks: bool,
) -> None:
    """Задачи руководителя поднимают свои показатели и приглушают остальные."""
    if not has_tasks:
        return
    name = row["metric"]
    if name in task_metrics:
        task = task_by_metric[name]
        dev["priority"] = round(dev["priority"] * rules.TASK_BOOST, 4)
        dev["task_id"] = task.get("id")
        target = task.get("target_value")
        if target is not None and row.get("fact") is not None and target != 0:
            dev["task_target"] = target
            dev["task_dev_pct"] = round((row["fact"] - target) / abs(target) * 100.0, 4)
            if not direction_better(row["fact"] > target, row.get("direction")):
                # Цель задачи не достигнута — это отдельный повод, даже если план выполнен.
                dev["kind"] = "task_related"
                dev["polarity"] = "negative"
    else:
        dev["priority"] = round(dev["priority"] * rules.NON_TASK_DAMPEN, 4)


def _merge(
    found: dict[str, dict[str, Any]], previous: list[dict[str, Any]], turn: int
) -> list[dict[str, Any]]:
    """Свежая карта + память прошлых ходов."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    prev_by_id = {d["id"]: d for d in previous}

    for dev_id, dev in found.items():
        old = prev_by_id.get(dev_id)
        if old is not None:
            dev["turn_created"] = old.get("turn_created", turn)
            dev["source"] = old.get("source", dev["source"])
            dev["note"] = old.get("note")
            if old.get("status") == "dismissed":
                dev["status"] = "dismissed"
        dev["turn_updated"] = turn
        out.append(dev)
        seen.add(dev_id)

    for old in previous:
        if old["id"] in seen:
            continue
        if old.get("source") == "agent":
            # Заметки модели живут, пока их не снимут явно.
            out.append({**old, "turn_updated": turn})
            continue
        age = turn - int(old.get("turn_updated") or turn)
        if age <= _RESOLVED_TTL_TURNS:
            out.append({**old, "status": "resolved"})
    return out


def materialize(db: Any, deviations: list[dict[str, Any]]) -> int:
    """Кладёт карту в таблицу ``deviation``, чтобы её было видно из SQL."""
    conn = db.conn
    metric_ids = {
        r["name"]: r["metric_id"] for r in conn.execute("SELECT metric_id, name FROM metric")
    }
    person_ids = {
        r["person_key"]: r["person_id"]
        for r in conn.execute("SELECT person_id, person_key FROM person")
    }
    conn.execute("DELETE FROM deviation")
    for d in deviations:
        conn.execute(
            "INSERT OR REPLACE INTO deviation (id, kind, polarity, metric_id, metric_name, "
            "metric_ext_id, person_id, person_key, element, date, depth, root_name, signal, "
            "priority, fact, plan, plan_dev_pct, pop_change_pct, zscore, percentile, "
            "benchmark_dev_pct, influent_share, source, status, task_id, task_target, "
            "task_dev_pct, note, turn_created, turn_updated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?)",
            (
                d["id"], d["kind"], d["polarity"], metric_ids.get(d["metric_name"]),
                d["metric_name"], d.get("metric_ext_id"),
                person_ids.get(d["person_key"]), d["person_key"], d.get("element"),
                d.get("date"), d.get("depth"), d.get("root_name"), d.get("signal"),
                d["priority"], d.get("fact"), d.get("plan"), d.get("plan_dev_pct"),
                d.get("pop_change_pct"), d.get("zscore"), d.get("percentile"),
                d.get("benchmark_dev_pct"), d.get("influent_share"), d["source"],
                d.get("status", "open"), d.get("task_id"), d.get("task_target"),
                d.get("task_dev_pct"), d.get("note"), d.get("turn_created"),
                d.get("turn_updated"),
            ),
        )
    conn.commit()
    return len(deviations)


__all__ = ["build_deviations", "materialize"]
