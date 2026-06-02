"""Загрузка входного JSON и разворачивание дерева метрик в плоские строки.

Структура входа фиксирована для всех доменов (колл-центр, разработчики и т.д.):
    {"me": <person>, "employees": [<person>, ...]}
    person  = {tabnum, fio, post, depart, metrics: [<metric>, ...]}
    metric  = {id, metric_name, metric_description, metric_type, measure_type,
               date, calc_period, fact, plan, benchmark, [influent_percent],
               element, child_metrics: [<metric>, ...]}

Названия конкретных метрик НЕ хардкодятся — обходим то, что есть в JSON.
"""
from __future__ import annotations

import json
from typing import Any

ROW_FIELDS: tuple[str, ...] = (
    "metric_uid",
    "parent_uid",
    "depth",
    "person_tabnum",
    "person_fio",
    "person_post",
    "person_depart",
    "person_is_me",
    "metric_id",
    "metric_name",
    "metric_description",
    "metric_type",
    "measure_type",
    "date",
    "calc_period",
    "fact",
    "plan",
    "benchmark",
    "influent_percent",
    "element",
)


def _normalize_metric_type(value: Any) -> Any:
    """Приводит направление метрики к каноничным 'прямая'/'обратная'.

    Источники присылают разные формы и регистр ('Обратный', 'ОБРАТНАЯ',
    'прямой'), а весь расчёт вердиктов и подсказки LLM сравнивают строго с
    'обратная' (analytics.py: `metric_type != "обратная"`). Без нормализации
    'Обратный' не матчится и обратная метрика считается как прямая. Неизвестные
    значения отдаём как есть (трактуются как 'прямая', прежнее поведение)."""
    if not isinstance(value, str):
        return value
    norm = value.strip().casefold()
    if norm.startswith("обратн"):
        return "обратная"
    if norm.startswith("прям"):
        return "прямая"
    return value


def _walk(
    metrics: list[dict[str, Any]],
    person: dict[str, Any],
    rows: list[dict[str, Any]],
    counter: list[int],
    parent_uid: int | None,
    depth: int,
) -> None:
    for node in metrics:
        uid = counter[0]
        counter[0] += 1
        rows.append(
            {
                "metric_uid": uid,
                "parent_uid": parent_uid,
                "depth": depth,
                "person_tabnum": person.get("tabnum"),
                "person_fio": person.get("fio"),
                "person_post": person.get("post"),
                "person_depart": person.get("depart"),
                "person_is_me": 1 if person.get("_is_me") else 0,
                "metric_id": node.get("id"),
                "metric_name": node.get("metric_name"),
                "metric_description": node.get("metric_description"),
                "metric_type": _normalize_metric_type(node.get("metric_type")),
                "measure_type": node.get("measure_type"),
                "date": node.get("date"),
                "calc_period": node.get("calc_period"),
                "fact": node.get("fact"),
                "plan": node.get("plan"),
                "benchmark": node.get("benchmark"),
                "influent_percent": node.get("influent_percent"),
                "element": node.get("element"),
            }
        )
        children = node.get("child_metrics") or []
        if children:
            _walk(children, person, rows, counter, uid, depth + 1)


def load_dataset_obj(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Разворачивает уже распарсенный JSON-объект в плоский список строк."""
    people: list[dict[str, Any]] = []
    me = data.get("me")
    if me is not None:
        me = {**me, "_is_me": True}
        people.append(me)
    for emp in data.get("employees", []) or []:
        people.append({**emp, "_is_me": False})

    rows: list[dict[str, Any]] = []
    counter = [1]
    for person in people:
        _walk(person.get("metrics", []) or [], person, rows, counter, None, 1)
    return rows


def load_dataset(path: str) -> list[dict[str, Any]]:
    """Читает JSON-файл и возвращает плоский список строк метрик."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return load_dataset_obj(data)
