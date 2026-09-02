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
# Сколько строк-разрезов пускаем в промпт: всего и на один показатель.
# Разрезов бывают сотни, и без капа они вытесняли из блока сами показатели.
_MAX_ELEMENT_LINES = 4
_MAX_ELEMENT_LINES_PER_METRIC = 1


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
    # Нулевое изменение печатать незачем — это шум в промпте.
    if dev.get("pop_change_pct") is not None and round(dev["pop_change_pct"], 1) != 0:
        bits.append(f"к прошлому периоду {dev['pop_change_pct']:+.1f} %")
    if dev.get("task_id"):
        bits.append("по задаче руководителя")
    if dev.get("date"):
        bits.append(f"на {dev['date']}")
    if dev.get("note"):
        bits.append(f"твоя заметка: {dev['note']}")
    return "- " + "; ".join(bits) + "."


def _limit_elements(items: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    """Оставляет в блоке немного строк-разрезов: они объясняют показатель, но
    не должны его вытеснять. Имена самих разрезов сохраняем — по ним модель
    называет проблемную зону в ответе."""
    # Сначала показатели, следом разрезы: разрез объясняет показатель, поэтому
    # читается после него, а не вместо него.
    items = sorted(items, key=lambda d: (1 if d.get("element") else 0,))
    out: list[dict[str, Any]] = []
    per_metric: dict[str, int] = {}
    elements = 0
    for dev in items:
        if len(out) >= cap:
            break
        if not dev.get("element"):
            out.append(dev)
            continue
        name = dev["metric_name"]
        if (
            elements >= _MAX_ELEMENT_LINES
            or per_metric.get(name, 0) >= _MAX_ELEMENT_LINES_PER_METRIC
        ):
            continue
        per_metric[name] = per_metric.get(name, 0) + 1
        elements += 1
        out.append(dev)
    return out


def deviations_block(deviations: list[dict[str, Any]], *, budget: int = _BUDGET) -> str:
    """Текстовый блок карты для промпта; пустая карта → пустая строка."""
    items = [d for d in deviations or [] if d.get("status", "open") == "open"]
    if not items:
        return ""
    negative = _limit_elements(
        [d for d in items if d["polarity"] == "negative"], _MAX_NEGATIVE
    )
    positive = _limit_elements(
        [d for d in items if d["polarity"] == "positive"], _MAX_POSITIVE
    )
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
