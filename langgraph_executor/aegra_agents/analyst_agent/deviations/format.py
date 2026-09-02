"""Карта отклонений в системный промпт — верхушка с числами.

Полная карта доступна модели инструментом list_deviations; в промпт идёт только
самое важное, чтобы не съесть контекст.
"""
from __future__ import annotations

from typing import Any

from . import rules

_MAX_NEGATIVE = 12
_MAX_POSITIVE = 3
_BUDGET = 3000


def _num(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{round(value, 2):g}"
    return str(value)


def _line(dev: dict[str, Any]) -> str:
    label = rules.KIND_LABELS.get(dev["kind"], dev["kind"])
    head = dev["metric_name"] + (f" ({dev['element']})" if dev.get("element") else "")
    bits = [f"{head} — {label}"]
    if dev.get("fact") is not None:
        piece = f"факт {_num(dev['fact'])}"
        if dev.get("plan") is not None:
            piece += f", план {_num(dev['plan'])}"
        bits.append(piece)
    if dev.get("plan_dev_pct") is not None:
        bits.append(f"к плану {dev['plan_dev_pct']:+.1f} %")
    if dev.get("pop_change_pct") is not None:
        bits.append(f"к прошлому периоду {dev['pop_change_pct']:+.1f} %")
    if dev.get("task_id"):
        bits.append("по задаче руководителя")
    if dev.get("date"):
        bits.append(f"на {dev['date']}")
    if dev.get("note"):
        bits.append(f"твоя заметка: {dev['note']}")
    return "- " + "; ".join(bits) + "."


def deviations_block(deviations: list[dict[str, Any]], *, budget: int = _BUDGET) -> str:
    """Текстовый блок карты для промпта; пустая карта → пустая строка."""
    items = [d for d in deviations or [] if d.get("status", "open") == "open"]
    if not items:
        return ""
    negative = [d for d in items if d["polarity"] == "negative"][:_MAX_NEGATIVE]
    positive = [d for d in items if d["polarity"] == "positive"][:_MAX_POSITIVE]
    lines = [
        "КАРТА ОТКЛОНЕНИЙ (посчитана по данным, отсортирована по важности; "
        "полная — инструментом list_deviations)"
    ]
    if negative:
        lines.append("Области внимания:")
        lines.extend(_line(d) for d in negative)
    if positive:
        lines.append("Достижения:")
        lines.extend(_line(d) for d in positive)
    text = "\n".join(lines)
    return text if len(text) <= budget else text[:budget].rsplit("\n", 1)[0]


__all__ = ["deviations_block"]
