"""Классификация видов метрик (json_analyzer_causal): для знаковых «вкладов»
относительный % обнуляется, вердикт динамики — из знака абсолютного изменения."""
from langgraph_executor.aegra_agents.json_analyzer_causal.analytics import (
    apply_metric_kinds,
    compute_analytics,
)
from langgraph_executor.aegra_agents.json_analyzer_causal.loader import load_dataset_obj
from langgraph_executor.aegra_agents.json_analyzer_causal.metric_kinds_cache import (
    _parse_kinds,
)
from langgraph_executor.aegra_agents.json_analyzer_causal.sqlite_store import SqliteStore
from langgraph_executor.aegra_agents.json_analyzer_causal.tools import _delta_cell


def _m(name, mtype, fact, date, *, element=None, plan=None):
    return {"id": f"{name}{element}{date}", "metric_name": name, "metric_type": mtype,
            "measure_type": "секунда", "date": date, "calc_period": "д",
            "fact": fact, "plan": plan, "benchmark": None, "element": element}


def _store():
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
        "SELECT a.pop_change_pct pct, a.pop_change_abs abs, a.pop_status ps "
        "FROM metrics m JOIN metric_analytics a ON a.metric_uid = m.metric_uid "
        "WHERE m.metric_name = ? "
        "AND (m.element = ? OR (? IS NULL AND m.element IS NULL)) AND m.date='2026-05-11'",
        (metric, element, element),
    ).fetchone()


def test_apply_kinds_contribution():
    store = _store()
    assert apply_metric_kinds(store, {"Вклад": "вклад"}) == 4
    a = _row(store, "Вклад", "A")
    assert a["pct"] is None and a["abs"] == -18 and a["ps"] == "улучшение"
    b = _row(store, "Вклад", "B")
    assert b["abs"] == 17 and b["ps"] == "ухудшение"
    assert _row(store, "Время", None)["pct"] is not None
    assert store.metric_kind_of("Вклад") == "вклад"


def test_delta_cell_and_parse_kinds():
    assert _delta_cell(12.3, 5, "сек") == "12.3 %"
    assert _delta_cell(None, -41.41, "секунда") == "-41.41 секунда"
    assert _delta_cell(None, None, "сек") == ""
    out = _parse_kinds(
        '[{"metric":"Вклад","kind":"вклад"},{"metric":"Время","kind":"мусор"}]',
        {"Вклад", "Время"},
    )
    assert out == {"Вклад": "вклад", "Время": "уровень"}
