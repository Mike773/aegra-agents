"""metric_tree отдаёт ОДИН уровень за вызов, а не всё поддерево.

Регресс на «при одной верхнеуровневой метрике metric_tree вываливает все
метрики». Теперь вызов возвращает корень + ТОЛЬКО прямых детей, помечает каждый
узел has_children (есть ли куда копать), а спуск глубже идёт повторными вызовами
на выбранном ребёнке. Проверяем не только структуру уровней, но и КОРРЕКТНОСТЬ
значений (имена, факты, направление-зависимые статусы) на каждом уровне.
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
from langgraph_executor.aegra_agents.json_analyzer.sqlite_store import SqliteStore
from langgraph_executor.aegra_agents.json_analyzer.tools import build_tools


def _m(name, mtype, fact, plan, benchmark=None, children=None, element=None):
    node = {
        "id": name,
        "metric_name": name,
        "metric_type": mtype,
        "measure_type": "ед",
        "date": "2026-01-12",
        "calc_period": "W",
        "fact": fact,
        "plan": plan,
        "benchmark": benchmark if benchmark is not None else plan,
    }
    if element is not None:
        node["element"] = element
    if children:
        node["child_metrics"] = children
    return node


# Трёхуровневое дерево у одного человека на одну неделю:
#   Выручка (depth1, прямая, 100<120 → хуже_плана) [есть дети]
#   ├─ Средний чек (depth2, прямая, 50>40 → лучше_плана) [есть дети]
#   │   ├─ Цена (depth3, прямая, 45>40 → лучше_плана) [лист]
#   │   └─ Доля скидок (depth3, ОБРАТНАЯ, 10>5 → хуже_плана) [лист]
#   └─ Число сделок (depth2, прямая, 2<3 → хуже_плана) [лист]
_DATASET = {
    "me": {
        "tabnum": 1,
        "fio": "Босс",
        "post": "рук",
        "depart": "d",
        "metrics": [
            _m(
                "Выручка", "прямая", 100.0, 120.0, children=[
                    _m(
                        "Средний чек", "прямая", 50.0, 40.0, children=[
                            _m("Цена", "прямая", 45.0, 40.0),
                            _m("Доля скидок", "обратная", 10.0, 5.0),
                        ],
                    ),
                    _m("Число сделок", "прямая", 2.0, 3.0),
                ],
            ),
        ],
    },
    "employees": [],
}


def _build_store():
    store = SqliteStore()
    store.load(load_dataset_obj(_DATASET))
    analytics.compute_analytics(store)
    return store


def _by_name(rows):
    return {r["metric_name"]: r for r in rows}


# --- уровни: один вызов = один уровень ---------------------------------------

def test_default_returns_only_root_and_direct_children():
    store = _build_store()
    res = store.metric_tree(name="Выручка", person="Босс", date="2026-01-12")
    names = {r["metric_name"] for r in res["rows"]}
    # корень + прямые дети, но НЕ внуки
    assert names == {"Выручка", "Средний чек", "Число сделок"}
    assert "Цена" not in names and "Доля скидок" not in names
    assert res["count"] == 3
    assert res["levels_shown"] == 1
    assert res["truncated"] is False


def test_has_children_flag_is_correct():
    store = _build_store()
    rows = _by_name(store.metric_tree(name="Выручка", person="Босс", date="2026-01-12")["rows"])
    assert rows["Выручка"]["has_children"] == 1       # есть Средний чек / Число сделок
    assert rows["Средний чек"]["has_children"] == 1   # есть Цена / Доля скидок
    assert rows["Число сделок"]["has_children"] == 0   # лист


def test_drill_down_one_level_at_a_time():
    # Спуск на следующий уровень — повторный вызов на ребёнке. Возвращает этого
    # ребёнка + его детей, и НЕ тянет родителя/сиблингов.
    store = _build_store()
    res = store.metric_tree(name="Средний чек", person="Босс", date="2026-01-12")
    names = {r["metric_name"] for r in res["rows"]}
    assert names == {"Средний чек", "Цена", "Доля скидок"}
    assert "Выручка" not in names          # родитель не подтянулся
    assert "Число сделок" not in names      # сиблинг не подтянулся
    leaves = _by_name(res["rows"])
    assert leaves["Цена"]["has_children"] == 0
    assert leaves["Доля скидок"]["has_children"] == 0


def test_max_levels_2_returns_three_tiers():
    store = _build_store()
    res = store.metric_tree(name="Выручка", person="Босс", date="2026-01-12", max_levels=2)
    names = {r["metric_name"] for r in res["rows"]}
    assert names == {"Выручка", "Средний чек", "Число сделок", "Цена", "Доля скидок"}
    assert res["count"] == 5
    assert res["levels_shown"] == 2


# --- корректность ЗНАЧЕНИЙ на каждом уровне -----------------------------------

def test_values_and_direction_statuses_are_correct():
    store = _build_store()
    top = _by_name(store.metric_tree(name="Выручка", person="Босс", date="2026-01-12")["rows"])
    # факт/план переносятся как есть
    assert top["Выручка"]["fact"] == 100.0 and top["Выручка"]["plan"] == 120.0
    # прямая, факт ниже плана → хуже_плана
    assert top["Выручка"]["plan_status"] == "хуже_плана"
    assert top["Средний чек"]["plan_status"] == "лучше_плана"   # прямая, 50>40
    assert top["Число сделок"]["plan_status"] == "хуже_плана"   # прямая, 2<3

    # На уровне внуков — обратная метрика: значение ВЫШЕ плана = хуже_плана.
    deep = _by_name(store.metric_tree(name="Средний чек", person="Босс", date="2026-01-12")["rows"])
    assert deep["Цена"]["plan_status"] == "лучше_плана"          # прямая, 45>40
    assert deep["Доля скидок"]["metric_type"] == "обратная"
    assert deep["Доля скидок"]["plan_status"] == "хуже_плана"    # обратная, 10>5 → плохо


# --- tool-обёртка: has_children сериализуется, глубина клампится --------------

def test_tool_wrapper_serializes_has_children_and_clamps_depth():
    store = _build_store()
    # metric_tree-замыкание не трогает index/embed_query — можно передать None.
    tools = build_tools(store, index=None, embed_query=None)
    mt = next(t for t in tools if t.name == "metric_tree").func

    one = json.loads(mt(metric="Выручка", person="Босс", date="2026-01-12"))
    names = {r["metric_name"] for r in one["rows"]}
    assert names == {"Выручка", "Средний чек", "Число сделок"}
    assert _by_name(one["rows"])["Выручка"]["has_children"] == 1
    assert _by_name(one["rows"])["Число сделок"]["has_children"] == 0

    # Запрос «дай всё дерево» (max_levels=99) клампится до 3 — глубже корректно,
    # но не безгранично. На нашем дереве глубиной 3 это все 5 узлов.
    huge = json.loads(mt(metric="Выручка", person="Босс", date="2026-01-12", max_levels=99))
    assert huge["levels_shown"] == 3
    assert {r["metric_name"] for r in huge["rows"]} == {
        "Выручка", "Средний чек", "Число сделок", "Цена", "Доля скидок"
    }
