"""Общие фикстуры тестов analyst_agent.

Датасеты строятся программно (см. make_person/make_metric) либо берутся из
samples_v2/*.json. Синтетический генератор make_dataset — для проверок бюджета
промпта на «проде» (100 L1-метрик × 5 уровней).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "samples_v2"


def load_sample(name: str) -> dict[str, Any]:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def make_metric(
    name: str,
    *,
    date: str = "2026-04-06",
    fact: Any = 10.0,
    plan: Any = None,
    metric_type: str = "прямая",
    element: str | None = None,
    children: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "id": extra.pop("id", f"id-{name}"),
        "metric_name": name,
        "metric_description": extra.pop("description", f"Описание {name}"),
        "metric_type": metric_type,
        "measure_type": extra.pop("measure_type", "у.е."),
        "date": date,
        "calc_period": extra.pop("calc_period", "неделя"),
        "fact": fact,
        "plan": plan,
        "element": element,
        "child_metrics": children or [],
    }
    node.update(extra)
    return node


def make_person(
    metrics: list[dict[str, Any]],
    *,
    tabnum: Any = 100500,
    fio: str = "Иванов Иван Иванович",
    is_me: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "tabnum": tabnum,
        "fio": fio,
        "post": extra.pop("post", "Аналитик"),
        "depart": extra.pop("depart", "Отдел"),
        "metrics": metrics,
        **extra,
    }


def make_dataset_obj(
    employee_metrics: list[dict[str, Any]],
    *,
    me: dict[str, Any] | None = None,
    employees_extra: list[dict[str, Any]] | None = None,
    **person_kwargs: Any,
) -> dict[str, Any]:
    return {
        "me": me,
        "employees": [make_person(employee_metrics, **person_kwargs)]
        + (employees_extra or []),
    }


def make_synthetic_dataset(
    *,
    n_level1: int = 100,
    depth: int = 5,
    periods: int = 6,
    elements: int = 5,
    people: int = 1,
    stars: int = 2,
) -> dict[str, Any]:
    """Синтетика «прод-масштаба»: n_level1 корней, цепочка детей до depth,
    история periods дат сиблингами, у первого ребёнка каждого корня разрезы."""

    def dates(n: int) -> list[str]:
        start = date(2026, 4, 6)
        return [(start + timedelta(days=7 * i)).isoformat() for i in range(n)]

    def chain(l1_idx: int, level: int) -> list[dict[str, Any]]:
        if level > depth:
            return []
        name = f"М{l1_idx}.{level}"
        nodes = [
            make_metric(name, date=d, fact=100.0 + level + i, plan=100.0 + level)
            for i, d in enumerate(dates(periods))
        ]
        if level == 2:
            for e in range(elements):
                nodes.append(
                    make_metric(
                        name,
                        date=dates(periods)[-1],
                        fact=50.0 + e,
                        element=f"Разрез {e}",
                    )
                )
        nodes[-1]["child_metrics"] = chain(l1_idx, level + 1)
        return nodes

    roots: list[dict[str, Any]] = []
    for i in range(n_level1):
        root_dates = dates(periods)
        for j, d in enumerate(root_dates):
            node = make_metric(f"М{i}.1", date=d, fact=200.0 + i + j, plan=200.0 + i)
            if j == len(root_dates) - 1:
                node["child_metrics"] = chain(i, 2)
            roots.append(node)
    for s in range(stars):
        star = make_metric(
            f"Звезда {s}", fact=None, plan=None, star_received=s % 2 == 0
        )
        star["child_metrics"] = [
            make_metric(
                f"Влияющий {s}", fact=10.0, plan=12.0, is_star_metric=True
            )
        ]
        roots.append(star)

    persons = [
        make_person(roots, tabnum=100500 + i, fio=f"Сотрудник {i}")
        for i in range(people)
    ]
    return {"me": None, "employees": persons}
