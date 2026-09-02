"""Голден-регрессия: порт compute_analytics в analyst_agent считает ТО ЖЕ,
что json_analyzer_v5, на всех демо-датасетах.

Пока v5 в дереве — это главная страховка переноса формул. Сверяем все 24
производные колонки построчно по ключу (персона, метрика, разрез, дата).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from _fixtures import SAMPLES_DIR, load_sample  # noqa: F401

from langgraph_executor.aegra_agents.analyst_agent.db import analytics as new_analytics
from langgraph_executor.aegra_agents.analyst_agent.db import core
from langgraph_executor.aegra_agents.json_analyzer_v5.analytics import (
    compute_analytics as v5_compute,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.loader import (
    load_aggregates_obj,
    load_dataset_obj,
    load_person_aggregates_obj,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.sqlite_store import SqliteStore

ANALYTICS_COLUMNS = (
    "plan_dev_abs", "plan_dev_pct", "plan_status",
    "benchmark_dev_abs", "benchmark_dev_pct", "benchmark_status",
    "pop_change_abs", "pop_change_pct", "pop_status",
    "trend", "trend_status",
    "peer_mean", "peer_std", "peer_count", "peer_rank", "peer_percentile",
    "zscore", "peer_status", "is_anomaly",
    "group_change_pct", "rel_change_pct", "rel_status",
    "group_hit_rate", "plan_rigidity",
)

SAMPLES = sorted(p.name for p in Path(SAMPLES_DIR).glob("*.json"))


def _aggregates_for(dataset: dict) -> list[dict]:
    """Синтетические peer-агрегаты по данным сотрудника: те же метрики, средние
    чуть ниже факта. Нужны, чтобы в паритете участвовали benchmark_*/rel_*/
    group_hit_rate/plan_rigidity, а не только «пустые» ветки."""
    seen: dict[str, list[tuple[str, float]]] = {}

    def walk(nodes):
        for n in nodes or []:
            name, dt, fact = n.get("metric_name"), n.get("date"), n.get("fact")
            if name and dt and isinstance(fact, (int, float)):
                seen.setdefault(name, []).append((dt, float(fact)))
            walk(n.get("child_metrics"))

    for person in dataset.get("employees") or []:
        walk(person.get("metrics"))

    metrics = []
    for name, points in seen.items():
        points.sort()
        cur_dt, cur_fact = points[-1]
        history = [
            {"dt": dt, "calc_period": "неделя", "mean_fact": fact * 0.95,
             "median": fact * 0.94, "hit_rate": 40.0, "top20_mean_fact": fact * 1.1,
             "total_objects": 50}
            for dt, fact in points[:-1]
        ]
        metrics.append({
            "metric_id": name, "metric_name": name,
            "aggregates": {
                "dt": cur_dt, "calc_period": "неделя", "mean_fact": cur_fact * 0.9,
                "median": cur_fact * 0.9, "hit_rate": 30.0,
                "top20_mean_fact": cur_fact * 1.2, "iqr": 1.0, "cv": 0.2,
                "total_objects": 50, "history": history,
            },
        })
    return [{"dataset": {"level": "OFFICE", "level_name": "по офису", "metrics": metrics}}]


def _v5_rows(dataset: dict, aggregates: list[dict] | None) -> dict[tuple, dict]:
    store = SqliteStore()
    store.load(load_dataset_obj(dataset))
    if aggregates:
        store.load_aggregates(load_aggregates_obj(aggregates))
        store.load_person_aggregates(load_person_aggregates_obj(dataset))
    v5_compute(store)
    cols = ", ".join(f"a.{c}" for c in ANALYTICS_COLUMNS)
    rows = store.conn.execute(
        f"SELECT m.person_key, m.metric_name, m.element, m.date, {cols} "
        "FROM metrics m JOIN metric_analytics a ON a.metric_uid = m.metric_uid"
    ).fetchall()
    return {
        (r["person_key"], r["metric_name"], r["element"], r["date"]): dict(r)
        for r in rows
    }


def _new_rows(dataset: dict, aggregates: list[dict] | None) -> dict[tuple, dict]:
    # compute=False: сверяем ровно порт формул, без подавления процентов у
    # рангов и вкладов — это отдельная возможность нового агента (её проверяет
    # test_e2e_regressions).
    db = core.build_run_db(dataset, aggregates, compute=False)
    new_analytics.compute_analytics(db.conn)
    cols = ", ".join(ANALYTICS_COLUMNS)
    rows = db.conn.execute(
        f"SELECT person_key, metric AS metric_name, element, date, {cols} FROM v_fact"
    ).fetchall()
    return {
        (r["person_key"], r["metric_name"], r["element"], r["date"]): dict(r)
        for r in rows
    }


def _assert_parity(dataset: dict, aggregates: list[dict] | None, label: str) -> None:
    v5 = _v5_rows(dataset, aggregates)
    new = _new_rows(dataset, aggregates)
    assert set(new) == set(v5), f"{label}: разошёлся набор строк"
    assert v5, f"{label}: пустая выборка — тест бессмысленен"
    for key, expected in v5.items():
        actual = new[key]
        for col in ANALYTICS_COLUMNS:
            exp, act = expected[col], actual[col]
            if isinstance(exp, float) and isinstance(act, float):
                assert act == pytest.approx(exp, rel=1e-6, abs=1e-6), f"{label} {key} {col}"
            else:
                assert act == exp, f"{label} {key} {col}: {act!r} != {exp!r}"


@pytest.mark.parametrize("sample", SAMPLES)
def test_parity_without_aggregates(sample):
    _assert_parity(load_sample(sample), None, f"{sample} без агрегатов")


@pytest.mark.parametrize("sample", SAMPLES)
def test_parity_with_aggregates(sample):
    dataset = load_sample(sample)
    _assert_parity(dataset, _aggregates_for(dataset), f"{sample} с агрегатами")


def test_parity_multi_person_peer_stats():
    """Несколько сотрудников: включаются peer_mean/rank/percentile/zscore/anomaly."""
    def person(tab, fio, facts):
        return {
            "tabnum": tab, "fio": fio, "post": "Оператор", "depart": "Отдел",
            "metrics": [
                {"id": "M1", "metric_name": "AHT", "metric_description": "Время",
                 "metric_type": "обратная", "measure_type": "сек", "date": d,
                 "calc_period": "неделя", "fact": f, "plan": 300.0,
                 "benchmark": None, "element": None, "child_metrics": []}
                for d, f in facts
            ],
        }

    dataset = {
        "me": None,
        "employees": [
            person(1, "Первый", [("2026-04-06", 280.0), ("2026-04-13", 290.0)]),
            person(2, "Второй", [("2026-04-06", 310.0), ("2026-04-13", 340.0)]),
            person(3, "Третий", [("2026-04-06", 300.0), ("2026-04-13", 295.0)]),
        ],
    }
    _assert_parity(dataset, None, "три сотрудника")


def test_parity_lagging_child_dates():
    """Ребёнок отстаёт по датам от родителя — расчёты по обоим сериям совпадают."""
    def node(name, date, fact, plan, children=None):
        return {
            "id": name, "metric_name": name, "metric_description": "",
            "metric_type": "прямая", "measure_type": "шт", "date": date,
            "calc_period": "неделя", "fact": fact, "plan": plan,
            "benchmark": None, "element": None, "child_metrics": children or [],
        }

    dataset = {
        "me": None,
        "employees": [{
            "tabnum": 7, "fio": "Иванов", "post": "Оператор", "depart": "Отдел",
            "metrics": [
                node("Родитель", "2026-04-13", 100.0, 90.0),
                node("Родитель", "2026-04-20", 120.0, 90.0, children=[
                    node("Ребёнок", "2026-04-07", 40.0, 50.0),
                    node("Ребёнок", "2026-04-14", 45.0, 50.0),
                ]),
            ],
        }],
    }
    _assert_parity(dataset, None, "отстающие даты ребёнка")


def test_samples_present():
    assert SAMPLES, "нет демо-датасетов в samples_v2"
    assert json.loads((Path(SAMPLES_DIR) / SAMPLES[0]).read_text(encoding="utf-8"))
