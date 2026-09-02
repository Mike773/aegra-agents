"""Задачи руководителя из брифинга.

Брифинг — первое сообщение диалога, свободный текст. Если в нём перечислены
конкретные задачи («1. посмотри продажи, 2. проверь качество»), карта отклонений
строится ОТ НИХ: названные показатели поднимаются в приоритете, остальные
приглушаются, а невыполненная числовая цель становится отдельным поводом.

Разбор эвристический и без LLM — нумерованные и маркированные строки. Тот же
формат ``Task`` использует планировщик дашборда, только заполняет его LLM.
"""
from __future__ import annotations

import re
from typing import Any, TypedDict

# Строка-задача: «1. …», «1) …», «- …», «• …».
_ITEM_RE = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•—])\s+(?P<text>\S.*)$")
# Числовая цель: «до 120», «не менее 95», «= 3.5».
_TARGET_RE = re.compile(
    r"(?:до|не\s+менее|не\s+более|минимум|максимум|=)\s+(?P<value>-?\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
_MAX_TASKS = 8
_TITLE_CAP = 200


class Task(TypedDict, total=False):
    id: str
    title: str
    question: str
    metric_names: list[str]
    target_value: float | None
    target_kind: str | None
    depends_on: list[str]


def _target(text: str) -> float | None:
    m = _TARGET_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group("value").replace(",", "."))
    except ValueError:
        return None


def parse_tasks(briefing: str | None) -> list[Task]:
    """Брифинг → список задач. Нет перечисления — пустой список (обычный режим)."""
    if not briefing or not briefing.strip():
        return []
    out: list[Task] = []
    for line in briefing.splitlines():
        m = _ITEM_RE.match(line)
        if not m:
            continue
        title = m.group("text").strip().rstrip(";.")[:_TITLE_CAP]
        if not title:
            continue
        out.append(
            Task(
                id=f"t{len(out) + 1}",
                title=title,
                question=title,
                metric_names=[],
                target_value=_target(title),
                target_kind="value" if _target(title) is not None else None,
                depends_on=[],
            )
        )
        if len(out) >= _MAX_TASKS:
            break
    return out


def resolve_task_metrics(db: Any, tasks: list[Task]) -> list[Task]:
    """Дополняет задачи именами показателей из каталога.

    Лестница резолва: точное вхождение имени показателя в текст задачи, затем
    общий резолвер базы (нормализация, подстрока, нечёткое сходство, алиасы из
    справки wiki). Ничего не нашли — задача остаётся без показателей и просто
    не влияет на приоритеты.
    """
    catalog = [
        (r["name"], r["name_norm"])
        for r in db.conn.execute("SELECT name, name_norm FROM metric ORDER BY depth, name")
    ]
    out: list[Task] = []
    for task in tasks:
        text_norm = " ".join((task.get("title") or "").split()).casefold()
        names: list[str] = []
        for name, name_norm in catalog:
            if name_norm and name_norm in text_norm and name not in names:
                names.append(name)
        if not names:
            ref = db.resolve_metric(task.get("title"))
            if ref is not None:
                names.append(ref.name)
        out.append({**task, "metric_names": names})
    return out


__all__ = ["Task", "parse_tasks", "resolve_task_metrics"]
