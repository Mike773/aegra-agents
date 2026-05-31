"""Учёт направления метрики ('обратная' — меньше = лучше) в производной аналитике.

Регресс на главную ошибку json_analyzer: для 'обратной' метрики рост значения —
это УХУДШЕНИЕ, а значение выше среднего коллег — это ХУЖЕ коллег. Раньше
направление применялось только к plan_status/peer_rank, а trend/wow_change/zscore
отдавались «сырыми», и модель читала 'рост' обратной метрики как успех.
"""
from __future__ import annotations

import os
import sys

# Корень репозитория в sys.path, чтобы пакет импортировался без установки.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langgraph_executor.aegra_agents.json_analyzer import analytics
from langgraph_executor.aegra_agents.json_analyzer.loader import load_dataset_obj
from langgraph_executor.aegra_agents.json_analyzer.sqlite_store import SqliteStore


def _metric(metric_id, name, mtype, date, fact, plan=90.0, benchmark=95.0):
    return {
        "id": metric_id,
        "metric_name": name,
        "metric_type": mtype,
        "measure_type": "ед",
        "date": date,
        "calc_period": "W",
        "fact": fact,
        "plan": plan,
        "benchmark": benchmark,
    }


def _build_store(data):
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    analytics.compute_analytics(store)
    return store


# Обратная метрика (AHT) растёт нед-к-нед и выше плана → всё плохо.
# Прямая метрика (Конверсия) растёт и пробивает план → всё хорошо.
_DATASET = {
    "me": {
        "tabnum": 1,
        "fio": "Босс",
        "post": "рук",
        "depart": "d",
        "metrics": [
            _metric("aht", "AHT", "обратная", "2026-01-05", 100.0),
            _metric("aht", "AHT", "обратная", "2026-01-12", 120.0),
            _metric("conv", "Конверсия", "прямая", "2026-01-05", 10.0, plan=12.0, benchmark=11.0),
            _metric("conv", "Конверсия", "прямая", "2026-01-12", 15.0, plan=12.0, benchmark=11.0),
        ],
    },
    "employees": [
        {
            "tabnum": i + 2,
            "fio": f"Сотр{i}",
            "post": "оп",
            "depart": "d",
            "metrics": [_metric("aht", "AHT", "обратная", "2026-01-12", v)],
        }
        for i, v in enumerate([80.0, 85.0, 90.0, 200.0])
    ],
}


def _row(store, metric, person, date):
    rows = store.get_metric(metric, person=person, date=date)["rows"]
    assert len(rows) == 1, rows
    return rows[0]


def test_inverse_rising_value_is_deterioration():
    store = _build_store(_DATASET)
    r = _row(store, "AHT", "Босс", "2026-01-12")
    assert r["trend"] == "рост"               # значение растёт
    assert r["trend_status"] == "ухудшение"   # но для обратной это ухудшение
    assert r["wow_status"] == "ухудшение"
    assert r["plan_status"] == "хуже_плана"


def test_direct_rising_value_is_improvement():
    store = _build_store(_DATASET)
    r = _row(store, "Конверсия", "Босс", "2026-01-12")
    assert r["trend"] == "рост"
    assert r["trend_status"] == "улучшение"
    assert r["wow_status"] == "улучшение"
    assert r["plan_status"] == "лучше_плана"


def test_find_flags_declining_improving_respect_direction():
    store = _build_store(_DATASET)
    declining = {r["metric_name"] for r in store.find_flags("declining")["rows"]}
    improving = {r["metric_name"] for r in store.find_flags("improving")["rows"]}
    # Обратная метрика с растущим значением — в declining, не в improving.
    assert "AHT" in declining and "AHT" not in improving
    # Прямая метрика с растущим значением — в improving, не в declining.
    assert "Конверсия" in improving and "Конверсия" not in declining


def test_peer_status_respects_direction():
    store = _build_store(_DATASET)
    by_fio = {r["person_fio"]: r for r in store.rank("AHT", "2026-01-12")["rows"]}
    # Самый высокий AHT (обратная) — худший среди коллег, хотя zscore положительный.
    worst = by_fio["Сотр3"]
    assert worst["zscore"] > 0
    assert worst["peer_rank"] == 4
    assert worst["peer_status"] == "хуже_коллег"
    # Самый низкий AHT — лучший.
    best = by_fio["Сотр0"]
    assert best["peer_rank"] == 1
    assert best["peer_status"] == "лучше_коллег"


def test_summary_trend_counts_use_direction_aware_labels():
    store = _build_store(_DATASET)
    counts = analytics.build_summary(store)["trend_counts_level1"]
    assert set(counts) == {"улучшение", "ухудшение", "стабильно"}
    assert counts["ухудшение"] == 1   # AHT
    assert counts["улучшение"] == 1   # Конверсия
