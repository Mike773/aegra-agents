"""Журнал отклонений: чтение картой и дозапись находок моделью.

Под ним живут два инструмента агента — ``list_deviations`` (полная карта с
числами; в промпт попадает только верхушка) и ``note_deviation`` (модель
фиксирует свою находку, и она переживает ходы и подзадачи дашборда).
"""
from __future__ import annotations

import hashlib
from typing import Any

from . import rules
from .builder import materialize

# Наборы видов отклонений под фильтры инструмента.
SCOPES: dict[str, tuple[str, ...]] = {
    "plan": ("below_plan", "above_plan", "segment_worst", "segment_best", "task_related"),
    "dynamics": ("declining", "improving", "anomaly"),
    "peers": ("peer_worse", "peer_better", "group_worse", "benchmark_gap"),
    "stars": ("star_missed", "star_at_risk", "star_received"),
}
_LIST_LIMIT = 20
_NOTE_CAP = 400


def _num(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{round(value, 2):g}"
    return str(value)


class DeviationLedger:
    """Карта отклонений одного запуска: живёт в замыкании узла и инструментов."""

    def __init__(self, db: Any, deviations: list[dict[str, Any]], *, turn: int = 1) -> None:
        self.db = db
        self.turn = turn
        self._items: list[dict[str, Any]] = [dict(d) for d in deviations]
        materialize(db, self._items)

    # --- чтение -----------------------------------------------------------

    def select(
        self,
        *,
        scope: str = "all",
        metric: str | None = None,
        include_resolved: bool = False,
        limit: int = _LIST_LIMIT,
    ) -> list[dict[str, Any]]:
        items = self._items
        if not include_resolved:
            items = [d for d in items if d.get("status", "open") == "open"]
        if scope.startswith("task:"):
            task_id = scope.split(":", 1)[1]
            items = [d for d in items if d.get("task_id") == task_id]
        elif scope == "achievements":
            items = [d for d in items if d["polarity"] == "positive"]
        elif scope == "model":
            items = [d for d in items if d.get("source") == "agent"]
        elif scope in SCOPES:
            items = [d for d in items if d["kind"] in SCOPES[scope]]
        if metric:
            ref = self.db.resolve_metric(metric)
            name = ref.name if ref else metric
            items = [d for d in items if d["metric_name"] == name]
        return sorted(items, key=lambda d: -d.get("priority", 0))[:limit]

    def render(self, *, scope: str = "all", metric: str | None = None, limit: int = _LIST_LIMIT) -> str:
        items = self.select(scope=scope, metric=metric, limit=limit)
        if not items:
            return "Достижений в карте нет." if scope == "achievements" else "Карта отклонений пуста."
        lines = []
        for d in items:
            label = rules.KIND_LABELS.get(d["kind"], d["kind"])
            head = d["metric_name"] + (f" ({d['element']})" if d.get("element") else "")
            bits = [f"{head} — {label}"]
            if d.get("fact") is not None:
                piece = f"факт {_num(d['fact'])}"
                if d.get("plan") is not None:
                    piece += f", план {_num(d['plan'])}"
                bits.append(piece)
            if d.get("plan_dev_pct") is not None:
                bits.append(f"к плану {d['plan_dev_pct']:+.1f} %")
            if d.get("pop_change_pct") is not None:
                bits.append(f"к прошлому периоду {d['pop_change_pct']:+.1f} %")
            if d.get("date"):
                bits.append(f"на {d['date']}")
            if d.get("note"):
                bits.append(f"заметка: {d['note']}")
            lines.append("- " + "; ".join(bits) + ".")
        return "\n".join(lines)

    # --- дозапись моделью -------------------------------------------------

    def note(
        self,
        *,
        metric: str,
        kind: str,
        text: str,
        element: str | None = None,
        polarity: str = "negative",
    ) -> str:
        """Фиксирует находку модели. Показатель резолвится по каталогу."""
        ref = self.db.resolve_metric(metric)
        if ref is None:
            names = [
                r["name"]
                for r in self.db.conn.execute(
                    "SELECT name FROM metric ORDER BY depth, name LIMIT 8"
                )
            ]
            return (
                f"Показатель «{metric}» не найден в данных. "
                f"Есть, например: {', '.join(names)}."
            )
        note_text = (text or "").strip()[:_NOTE_CAP]
        dev_id = "a" + hashlib.sha1(
            f"{kind}|{ref.name}|{element or ''}|{note_text}".encode("utf-8")
        ).hexdigest()[:11]
        row = self.db.conn.execute(
            "SELECT person_key FROM person ORDER BY is_me, person_id LIMIT 1"
        ).fetchone()
        entry = {
            "id": dev_id,
            "kind": kind or "hypothesis",
            "polarity": "positive" if polarity == "positive" else "negative",
            "metric_name": ref.name,
            "metric_ext_id": None,
            "person_key": row["person_key"] if row else "",
            "element": element,
            "date": None,
            "depth": None,
            "root_name": None,
            "signal": None,
            # Заметка модели весит наравне с заметным авто-отклонением, но не
            # перебивает верхушку карты: приоритет ставят данные, а не текст.
            "priority": rules.MIN_MATERIAL_PRIORITY,
            "source": "agent",
            "status": "open",
            "note": note_text,
            "turn_created": self.turn,
            "turn_updated": self.turn,
        }
        self._items = [d for d in self._items if d["id"] != dev_id] + [entry]
        materialize(self.db, self._items)
        return f"Отметил: {ref.name} — {note_text}"

    def to_state(self) -> list[dict[str, Any]]:
        """Карта для чекпойнтера: переживает ход и подзадачи дашборда."""
        return [dict(d) for d in self._items]


__all__ = ["SCOPES", "DeviationLedger"]
