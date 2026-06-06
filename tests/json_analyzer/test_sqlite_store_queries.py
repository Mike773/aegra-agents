"""Характеризующие тесты SqliteStore.get_metric / compare / rank.

Фиксируют ТЕКУЩЕЕ поведение построения WHERE (фильтры person/element/date,
ветка агрегат-vs-разрезы, ошибки) перед рефакторингом общего билдера. Это сеть
безопасности: после извлечения общего хелпера поведение должно совпасть до байта.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langgraph_executor.aegra_agents.json_analyzer import analytics
from langgraph_executor.aegra_agents.json_analyzer.loader import load_dataset_obj
from langgraph_executor.aegra_agents.json_analyzer.sqlite_store import SqliteStore


def _m(name, mtype, date, fact, *, plan=90.0, benchmark=95.0, element=None):
    row = {
        "id": name.lower(),
        "metric_name": name,
        "metric_type": mtype,
        "measure_type": "ед",
        "date": date,
        "calc_period": "W",
        "fact": fact,
        "plan": plan,
        "benchmark": benchmark,
    }
    if element is not None:
        row["element"] = element
    return row


# AHT — есть агрегатная строка (element отсутствует) у всех; две недели у Босса.
# Продажи — ТОЛЬКО разрезы по element (агрегата нет) — ветка fallback.
_DATASET = {
    "me": {
        "tabnum": 1, "fio": "Иванов Иван", "post": "рук", "depart": "d",
        "metrics": [
            _m("AHT", "обратная", "2026-01-05", 100.0),
            _m("AHT", "обратная", "2026-01-12", 120.0),
            _m("Продажи", "прямая", "2026-01-05", 10.0, element="ПродуктA"),
            _m("Продажи", "прямая", "2026-01-05", 20.0, element="ПродуктB"),
            _m("Продажи", "прямая", "2026-01-12", 15.0, element="ПродуктA"),
            _m("Продажи", "прямая", "2026-01-12", 25.0, element="ПродуктB"),
        ],
    },
    "employees": [
        {"tabnum": 2, "fio": "Петров Пётр", "post": "оп", "depart": "d",
         "metrics": [
             _m("AHT", "обратная", "2026-01-12", 80.0),
             _m("Продажи", "прямая", "2026-01-12", 5.0, element="ПродуктA"),
         ]},
        {"tabnum": 3, "fio": "Сидоров Сидор", "post": "оп", "depart": "d",
         "metrics": [
             _m("AHT", "обратная", "2026-01-12", 90.0),
             _m("Продажи", "прямая", "2026-01-12", 7.0, element="ПродуктA"),
         ]},
        {"tabnum": 4, "fio": "Кузнецов Кузьма", "post": "ст", "depart": "d",
         "metrics": [_m("AHT", "обратная", "2026-01-12", 95.0)]},
    ],
}


def _store():
    store = SqliteStore()
    store.load(load_dataset_obj(_DATASET))
    analytics.compute_analytics(store)
    return store


# --- get_metric -------------------------------------------------------------

def test_get_metric_aggregate_by_name_and_date():
    res = _store().get_metric("AHT", person="Иванов Иван", date="2026-01-12")
    assert res["count"] == 1
    row = res["rows"][0]
    assert row["element"] is None      # взята агрегатная строка
    assert row["fact"] == 120.0
    assert "разрезы_вместо_агрегата" not in res


def test_get_metric_person_by_tabnum_matches_same_row():
    by_name = _store().get_metric("AHT", person="Иванов Иван", date="2026-01-12")
    by_tab = _store().get_metric("AHT", person="1", date="2026-01-12")
    assert by_tab["count"] == 1
    assert by_tab["rows"][0]["fact"] == by_name["rows"][0]["fact"]


def test_get_metric_no_aggregate_falls_back_to_element_splits():
    res = _store().get_metric("Продажи", person="Иванов Иван", date="2026-01-05")
    assert res["count"] == 2
    assert {r["element"] for r in res["rows"]} == {"ПродуктA", "ПродуктB"}
    assert "разрезы_вместо_агрегата" in res
    assert "ПродуктA" in res["разрезы_вместо_агрегата"]
    assert "ПродуктB" in res["разрезы_вместо_агрегата"]


def test_get_metric_explicit_element_no_fallback_note():
    res = _store().get_metric(
        "Продажи", person="Иванов Иван", element="ПродуктA", date="2026-01-05"
    )
    assert res["count"] == 1
    assert res["rows"][0]["element"] == "ПродуктA"
    assert res["rows"][0]["fact"] == 10.0
    assert "разрезы_вместо_агрегата" not in res


# --- compare ----------------------------------------------------------------

def test_compare_without_person_returns_error():
    res = _store().compare("AHT")
    assert "error" in res
    assert res["rows"] == []
    assert res["count"] == 0


def test_compare_aggregate_orders_by_date():
    res = _store().compare(
        "AHT", person="Иванов Иван", dates=["2026-01-05", "2026-01-12"]
    )
    assert res["count"] == 2
    assert [r["date"] for r in res["rows"]] == ["2026-01-05", "2026-01-12"]
    assert all(r["element"] is None for r in res["rows"])
    assert "разрезы_вместо_агрегата" not in res


def test_compare_no_aggregate_fallback_orders_by_element_then_date():
    res = _store().compare(
        "Продажи", person="Иванов Иван", dates=["2026-01-05", "2026-01-12"]
    )
    assert res["count"] == 4
    assert [(r["element"], r["date"]) for r in res["rows"]] == [
        ("ПродуктA", "2026-01-05"),
        ("ПродуктA", "2026-01-12"),
        ("ПродуктB", "2026-01-05"),
        ("ПродуктB", "2026-01-12"),
    ]
    assert "разрезы_вместо_агрегата" in res


def test_compare_explicit_element_filters_single_split():
    res = _store().compare(
        "Продажи", person="Иванов Иван", element="ПродуктA",
        dates=["2026-01-05", "2026-01-12"],
    )
    assert res["count"] == 2
    assert all(r["element"] == "ПродуктA" for r in res["rows"])
    assert "разрезы_вместо_агрегата" not in res


# --- rank -------------------------------------------------------------------

def test_rank_aggregate_excludes_me_and_orders_by_peer_rank():
    res = _store().rank("AHT", "2026-01-12")
    assert res["metric_type"] == "обратная"
    fios = {r["person_fio"] for r in res["rows"]}
    assert "Иванов Иван" not in fios            # person_is_me исключён
    assert fios == {"Петров Пётр", "Сидоров Сидор", "Кузнецов Кузьма"}
    ranks = [r["peer_rank"] for r in res["rows"] if r["peer_rank"] is not None]
    assert ranks == sorted(ranks)              # упорядочено по peer_rank


def test_rank_no_aggregate_without_element_errors():
    res = _store().rank("Продажи", "2026-01-12")
    assert "error" in res
    assert res["rows"] == []
    assert res["metric_type"] == "прямая"
    assert "ПродуктA" in res["elements"]


def test_rank_no_aggregate_with_explicit_element():
    res = _store().rank("Продажи", "2026-01-12", element="ПродуктA")
    assert res["count"] == 2
    assert {r["person_fio"] for r in res["rows"]} == {"Петров Пётр", "Сидоров Сидор"}
    assert res["metric_type"] == "прямая"


def test_rank_post_filter_narrows_population():
    res = _store().rank("AHT", "2026-01-12", post="оп")
    assert {r["person_fio"] for r in res["rows"]} == {"Петров Пётр", "Сидоров Сидор"}
