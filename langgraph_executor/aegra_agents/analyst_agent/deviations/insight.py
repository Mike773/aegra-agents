"""Главный вывод для сервиса инсайтов — детерминированно, без LLM.

Берём вершину карты отклонений: она уже отранжирована по влиянию, масштабу и
управляемости, поэтому отдельный вызов модели-классификатора не нужен. Текст
собирается шаблоном из тех же чисел, что видит руководитель в ответе.
"""
from __future__ import annotations

from typing import Any

from . import rules

# Типы, которые понимает сервис инсайтов (контракт v4). Помимо type/metric_id/
# metric_name/text инсайт несёт fact и plan метрики — последний срез, по
# которому и сформулирован вывод.
TYPE_MAIN_PROBLEM = "main_problem"
TYPE_ACHIEVEMENT = "achievement"
TYPE_NORM = "norm"


def _num(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{round(value, 2):g}"
    return str(value)


def _describe(dev: dict[str, Any]) -> str:
    label = rules.KIND_LABELS.get(dev["kind"], dev["kind"])
    parts = [f"{dev['metric_name']}"]
    if dev.get("element"):
        parts[0] += f" ({dev['element']})"
    if dev.get("fact") is not None:
        head = f"факт {_num(dev['fact'])}"
        if dev.get("plan") is not None:
            head += f" при плане {_num(dev['plan'])}"
        parts.append(head)
    if dev.get("plan_dev_pct") is not None:
        parts.append(f"{label} на {abs(dev['plan_dev_pct']):.1f} %")
    else:
        parts.append(label)
    if dev.get("pop_change_pct") is not None:
        parts.append(f"к прошлому периоду {dev['pop_change_pct']:+.1f} %")
    if dev.get("task_target") is not None:
        parts.append(f"цель задачи {_num(dev['task_target'])}")
    return ": ".join([parts[0], ", ".join(parts[1:])]) if len(parts) > 1 else parts[0]


def _fallback_norm(db: Any) -> dict[str, Any] | None:
    """Отклонений нет — берём корневой показатель с наибольшим покрытием."""
    if db is None:
        return None
    row = db.conn.execute(
        "SELECT v.metric, v.ext_id, v.fact, v.plan, v.date "
        "FROM v_fact_latest v WHERE v.depth = 1 AND v.element IS NULL "
        "ORDER BY (v.plan IS NOT NULL) DESC, v.metric LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    text = f"{row['metric']}: факт {_num(row['fact'])}"
    if row["plan"] is not None:
        text += f" при плане {_num(row['plan'])}"
    text += " — показатели в норме, существенных отклонений нет."
    return {
        "type": TYPE_NORM,
        "metric_id": row["ext_id"],
        "metric_name": row["metric"],
        "fact": row["fact"],
        "plan": row["plan"],
        "text": text,
    }


def pick_main_insight(
    deviations: list[dict[str, Any]], *, db: Any = None
) -> dict[str, Any] | None:
    """Один главный вывод: проблема → достижение → «всё в норме»."""
    open_devs = [
        d for d in deviations if d.get("status", "open") == "open"
    ]
    material = [
        d for d in open_devs if d.get("priority", 0) >= rules.MIN_MATERIAL_PRIORITY
    ]

    negatives = sorted(
        (d for d in material if d["polarity"] == "negative"),
        key=lambda d: -d["priority"],
    )
    if negatives:
        top = negatives[0]
        return {
            "type": TYPE_MAIN_PROBLEM,
            "metric_id": top.get("metric_ext_id"),
            "metric_name": top["metric_name"],
            "fact": top.get("fact"),
            "plan": top.get("plan"),
            "text": _describe(top),
        }

    positives = sorted(
        (d for d in material if d["polarity"] == "positive"),
        key=lambda d: -d["priority"],
    )
    if positives:
        top = positives[0]
        return {
            "type": TYPE_ACHIEVEMENT,
            "metric_id": top.get("metric_ext_id"),
            "metric_name": top["metric_name"],
            "fact": top.get("fact"),
            "plan": top.get("plan"),
            "text": _describe(top),
        }

    return _fallback_norm(db)


__all__ = ["TYPE_ACHIEVEMENT", "TYPE_MAIN_PROBLEM", "TYPE_NORM", "pick_main_insight"]
