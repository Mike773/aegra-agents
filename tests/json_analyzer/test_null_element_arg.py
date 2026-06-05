"""Литерал 'null' в аргументе element/person → трактуется как «не задан».

Регресс на прод-баг: GigaChat начал слать element='null' (строкой) вместо
опускания аргумента. Раньше это уходило в SQL как WHERE m.element = 'null' и
молча давало 0 строк («нет выборки в данных»). Теперь null/none/nil (любой
регистр) нормализуются в None и на обеих границах — tools._blank_to_none и
store._is_unset — трактуются как «фильтр не задан».
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langgraph_executor.aegra_agents.json_analyzer import analytics
from langgraph_executor.aegra_agents.json_analyzer.loader import load_dataset_obj
from langgraph_executor.aegra_agents.json_analyzer.sqlite_store import (
    SqliteStore,
    _is_unset,
)
from langgraph_executor.aegra_agents.json_analyzer.tools import (
    _blank_to_none,
    build_tools,
)


def _m(name, mtype, fact, plan, element=None):
    node = {
        "id": name,
        "metric_name": name,
        "metric_type": mtype,
        "measure_type": "ед",
        "date": "2026-01-12",
        "calc_period": "W",
        "fact": fact,
        "plan": plan,
        "benchmark": plan,
    }
    if element is not None:
        node["element"] = element
    return node


# Метрика с агрегатной строкой (element IS NULL) + разрезы по element.
_DATASET = {
    "me": {
        "tabnum": 1,
        "fio": "Босс",
        "post": "р",
        "depart": "d",
        "metrics": [
            _m("Доля переводов", "обратная", 8.2, 5.0),                       # агрегат
            _m("Доля переводов", "обратная", 10.0, 5.0, element="Банковские счета"),
            _m("Доля переводов", "обратная", 6.0, 5.0, element="Переводы в рублях"),
        ],
    },
    "employees": [],
}


def _build_store():
    store = SqliteStore()
    store.load(load_dataset_obj(_DATASET))
    analytics.compute_analytics(store)
    return store


# --- нормализаторы -----------------------------------------------------------

def test_blank_to_none_maps_null_literals():
    for v in ("", "  ", "null", "NULL", " Null ", "none", "None", "nil", "NIL"):
        assert _blank_to_none(v) is None, v
    # реальные значения не трогаются
    assert _blank_to_none("Банковские счета") == "Банковские счета"
    assert _blank_to_none("Производительность") == "Производительность"


def test_is_unset_predicate():
    for v in (None, "", "  ", "null", "NULL", "none", "None", "nil"):
        assert _is_unset(v) is True, v
    assert _is_unset("Банковские счета") is False
    assert _is_unset(0) is False          # реальный tabnum/число — не «пусто»


# --- поведение store: element='null' == element опущен ----------------------

def test_get_metric_null_element_returns_aggregate_not_empty():
    store = _build_store()
    omitted = store.get_metric("Доля переводов", person="Босс")["rows"]
    as_null = store.get_metric("Доля переводов", person="Босс", element="null")["rows"]
    # агрегатная строка (element IS NULL), а не пустая выборка
    assert len(omitted) == 1 and omitted[0].get("element") is None
    assert as_null == omitted
    # реальный разрез по-прежнему фильтруется точно
    one = store.get_metric("Доля переводов", person="Босс", element="Банковские счета")["rows"]
    assert len(one) == 1 and one[0]["element"] == "Банковские счета"


def test_find_flags_null_element_is_unfiltered():
    store = _build_store()
    allf = store.find_flags("below_plan", metric="Доля переводов")["rows"]
    nullf = store.find_flags("below_plan", metric="Доля переводов", element="null")["rows"]
    assert nullf == allf
    assert len(nullf) > 0   # «обратная», факт выше плана → есть below_plan, не пусто


# --- tool-обёртка: element='null' не даёт пустоту ---------------------------

def test_tool_get_metric_null_element_not_empty():
    store = _build_store()
    tools = {t.name: t.func for t in build_tools(store, index=None, embed_query=None)}
    out = json.loads(tools["get_metric"](metric="Доля переводов", person="Босс", element="null"))
    assert out["rows"], "element='null' должен трактоваться как агрегат, а не 0 строк"
    assert out["rows"][0].get("element") is None
