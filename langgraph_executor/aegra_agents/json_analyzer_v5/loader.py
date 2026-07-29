"""Загрузка входного JSON и разворачивание дерева метрик в плоские строки.

Структура входа фиксирована для всех доменов (колл-центр, разработчики и т.д.):
    {"me": <person>, "employees": [<person>, ...]}
    person  = {tabnum, fio, post, depart, metrics: [<metric>, ...]}
    metric  = {id, metric_name, metric_description, metric_type, measure_type,
               date, calc_period, fact, plan, benchmark, [ex], [rr],
               [influent_percent], element, [rankings],
               child_metrics: [<metric>, ...]}
    rankings = [{rank: "458 из 500", level: "ORG"|"TERR"|"OFFICE",
                 percentile}, ...] — место сотрудника в peer-группе уровня.

Отдельный вход — batch-агрегаты peer-групп (``load_aggregates_obj``):
    [{"dataset": {"level": "ORG"|"TERR"|"OFFICE", "metrics": [<agg_metric>]}}]
    agg_metric = {metric_id, metric_name, aggregates: {dt, calc_period,
                  mean_fact, mean_plan, mean_ex, median, hit_rate,
                  top20_mean_fact, iqr, cv, total_objects,
                  history: [<те же поля за прошлые периоды>]},
                  children_metrics: [<agg_metric>, ...]}

Названия конкретных метрик НЕ хардкодятся — обходим то, что есть в JSON.
"""
from __future__ import annotations

import re
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
    "person_key",
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
    "ex",
    "rr",
    "influent_percent",
    "element",
)


def _is_empty_fact(value: Any) -> bool:
    """Факт «пустой» = None или пустая строка. Ноль (0/0.0) — ВАЛИДНЫЙ факт и
    пустым не считается (иначе потеряли бы законные нулевые значения)."""
    if value is None:
        return True
    return isinstance(value, str) and value.strip() == ""


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
        children = node.get("child_metrics") or []
        # Строку с пустым фактом в базу не грузим: в ряду по датам пустая точка
        # затеняет последнюю реальную (analytics берёт prev = непосредственно
        # предыдущий период), и сравнение период-к-периоду не считается. Отбрасываем
        # только ЛИСТ без факта — узел-агрегат с детьми сохраняем, чтобы не потерять
        # реальные разрезы под ним (на практике пустой факт всегда у листьев).
        if not children and _is_empty_fact(node.get("fact")):
            continue
        uid = counter[0]
        counter[0] += 1
        rankings = node.get("rankings")
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
                "person_key": person.get("_key"),
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
                "ex": node.get("ex"),
                "rr": node.get("rr"),
                "influent_percent": node.get("influent_percent"),
                "element": node.get("element"),
                # Не в ROW_FIELDS: в таблицу metrics не идёт, SqliteStore.load
                # разложит по отдельной metric_rankings.
                "rankings": rankings if isinstance(rankings, list) else None,
            }
        )
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

    # Канонический ключ человека для ВНУТРЕННЕЙ идентичности: табельный (как текст),
    # иначе ФИО, иначе индекс. Нужен, т.к. на проде tabnum может приходить null —
    # тогда без фолбэка разные люди схлопывались бы по NULL person_tabnum.
    for i, person in enumerate(people):
        tab = person.get("tabnum")
        fio = person.get("fio")
        person["_key"] = str(tab) if tab is not None else (fio if fio else f"person{i}")

    rows: list[dict[str, Any]] = []
    counter = [1]
    for person in people:
        _walk(person.get("metrics", []) or [], person, rows, counter, None, 1)
    return rows


_RANK_RE = re.compile(r"(\d+)\D+(\d+)")


def parse_rank(raw: Any) -> tuple[int | None, int | None]:
    """Разбирает строку ранга «458 из 500» → (458, 500).

    Формат разделителя не фиксируем (regex «число, не-числа, число»).
    Нераспознанное → (None, None); исходная строка сохраняется в rank_raw."""
    if not isinstance(raw, str):
        return None, None
    m = _RANK_RE.search(raw)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


AGG_ROW_FIELDS: tuple[str, ...] = (
    "node_uid",
    "parent_node_uid",
    "depth",
    "level",
    "metric_id",
    "metric_name",
    "dt",
    "calc_period",
    "is_current",
    "mean_fact",
    "mean_plan",
    "mean_ex",
    "median",
    "hit_rate",
    "top20_mean_fact",
    "iqr",
    "cv",
    "total_objects",
)

_AGG_VALUE_KEYS = (
    "mean_fact",
    "mean_plan",
    "mean_ex",
    "median",
    "hit_rate",
    "top20_mean_fact",
    "iqr",
    "cv",
    "total_objects",
)


def _walk_aggregates(
    metrics: list[Any],
    level: Any,
    rows: list[dict[str, Any]],
    counter: list[int],
    parent_uid: int | None,
    depth: int,
) -> None:
    for node in metrics or []:
        if not isinstance(node, dict):
            continue
        uid = counter[0]
        counter[0] += 1
        agg = node.get("aggregates") or {}
        base = {
            "node_uid": uid,
            "parent_node_uid": parent_uid,
            "depth": depth,
            "level": level,
            "metric_id": node.get("metric_id"),
            "metric_name": node.get("metric_name"),
        }

        def _row(src: dict[str, Any], is_current: int) -> dict[str, Any]:
            return {
                **base,
                "dt": src.get("dt"),
                "calc_period": src.get("calc_period"),
                "is_current": is_current,
                **{k: src.get(k) for k in _AGG_VALUE_KEYS},
            }

        if isinstance(agg, dict) and agg:
            rows.append(_row(agg, 1))
            for h in agg.get("history") or []:
                if isinstance(h, dict):
                    rows.append(_row(h, 0))
        # В этом payload дети приходят как "children_metrics" (в основном
        # датасете — "child_metrics"); принимаем оба на всякий случай.
        children = node.get("children_metrics") or node.get("child_metrics") or []
        _walk_aggregates(children, level, rows, counter, uid, depth + 1)


def load_aggregates_obj(data: Any) -> list[dict[str, Any]]:
    """Разворачивает batch-агрегаты peer-групп в плоские строки.

    Вход: список ``{"dataset": {"level", "metrics": [...]}}`` (см. докстринг
    модуля). Все срезы одного узла метрики (текущий ``aggregates`` +
    записи ``history``) делят ``node_uid``; текущий помечен ``is_current=1``.
    Битый или пустой вход → ``[]`` — агрегаты опциональны и не фатальны.
    """
    rows: list[dict[str, Any]] = []
    counter = [1]
    if not isinstance(data, list):
        return rows
    for entry in data:
        ds = entry.get("dataset") if isinstance(entry, dict) else None
        if isinstance(ds, dict):
            _walk_aggregates(
                ds.get("metrics") or [], ds.get("level"), rows, counter, None, 1
            )
    return rows
