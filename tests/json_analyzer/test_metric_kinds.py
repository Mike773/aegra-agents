"""Классификация видов метрик: для знаковых «вкладов» относительный % обнуляется,
вердикт динамики берётся из знака абсолютного изменения; рендер показывает абсолют.
"""
from langgraph_executor.aegra_agents.json_analyzer.analytics import (
    apply_metric_kinds,
    compute_analytics,
)
from langgraph_executor.aegra_agents.json_analyzer.loader import load_dataset_obj
from langgraph_executor.aegra_agents.json_analyzer.metric_kinds_cache import _parse_kinds
from langgraph_executor.aegra_agents.json_analyzer.sqlite_store import SqliteStore
from langgraph_executor.aegra_agents.json_analyzer.tools import _delta_cell


def _m(name, mtype, fact, date, *, element=None, plan=None):
    return {"id": f"{name}{element}{date}", "metric_name": name, "metric_type": mtype,
            "measure_type": "секунда", "date": date, "calc_period": "д",
            "fact": fact, "plan": plan, "benchmark": None, "element": element}


def _store():
    """Сотрудник: знаковый «Вклад» (обратная) по разрезам A/B + уровневое «Время»."""
    metrics = []
    for date, va, vb, t in [("2026-05-04", 10, -5, 100), ("2026-05-11", -8, 12, 130)]:
        metrics += [
            _m("Вклад", "обратная", va, date, element="A"),
            _m("Вклад", "обратная", vb, date, element="B"),
            _m("Время", "обратная", t, date, plan=120),
        ]
    data = {"me": {"fio": "Босс", "metrics": []},
            "employees": [{"fio": "Иванов", "metrics": metrics}]}
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    compute_analytics(store)
    return store


def _row(store, metric, element):
    return store.conn.execute(
        "SELECT a.pop_change_pct pct, a.pop_change_abs abs, a.pop_status ps, "
        "a.plan_dev_pct pdp FROM metrics m JOIN metric_analytics a "
        "ON a.metric_uid = m.metric_uid WHERE m.metric_name = ? "
        "AND (m.element = ? OR (? IS NULL AND m.element IS NULL)) AND m.date='2026-05-11'",
        (metric, element, element),
    ).fetchone()


def test_apply_kinds_nulls_relative_pct_for_contribution():
    store = _store()
    assert _row(store, "Вклад", "A")["pct"] is not None  # до применения % есть
    affected = apply_metric_kinds(store, {"Вклад": "вклад"})
    assert affected == 4  # 2 разреза × 2 даты
    a = _row(store, "Вклад", "A")
    assert a["pct"] is None and a["abs"] == -18  # % обнулён, абсолют сохранён
    # обратная, значение упало (10→-8) → улучшение (вердикт из знака абс., НЕ инвертирован)
    assert a["ps"] == "улучшение"
    b = _row(store, "Вклад", "B")
    assert b["abs"] == 17 and b["ps"] == "ухудшение"  # выросло (-5→12) → ухудшение


def test_apply_kinds_leaves_level_metric_intact():
    store = _store()
    apply_metric_kinds(store, {"Вклад": "вклад"})  # «Время» не указано → уровень
    t = _row(store, "Время", None)
    assert t["pct"] is not None  # относительный % сохранён у уровневой метрики
    assert store.metric_kind_of("Вклад") == "вклад"
    assert store.metric_kind_of("Время") is None


def test_delta_cell_pct_then_abs_fallback():
    assert _delta_cell(12.3, 5, "сек") == "12.3 %"
    assert _delta_cell(None, 5.0, "сек") == "5 сек"   # % нет → абсолют с единицей
    assert _delta_cell(None, -41.41, "секунда") == "-41.41 секунда"
    assert _delta_cell(None, None, "сек") == ""


def test_parse_kinds_filters_and_defaults():
    out = _parse_kinds(
        '[{"metric":"Вклад","kind":"вклад"},'
        '{"metric":"X","kind":"вклад"},'          # нет в valid_names → отброшено
        '{"metric":"Время","kind":"мусор"}]',     # невалидный вид → дефолт «уровень»
        {"Вклад", "Время"},
    )
    assert out == {"Вклад": "вклад", "Время": "уровень"}
