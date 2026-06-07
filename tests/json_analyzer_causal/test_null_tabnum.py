"""Устойчивость каузального слоя к null person_tabnum (на проде табельный не
приходит). Идентичность человека — по person_key (фолбэк на ФИО), поэтому
attribute_change(person=ФИО) должен возвращать непустой algebraic-разбор, а не
«человек не найден».
"""
from langgraph_executor.aegra_agents.json_analyzer_causal import causal
from langgraph_executor.aegra_agents.json_analyzer_causal.analytics import (
    compute_analytics,
)
from langgraph_executor.aegra_agents.json_analyzer_causal.loader import (
    load_dataset_obj,
)
from langgraph_executor.aegra_agents.json_analyzer_causal.sqlite_store import (
    SqliteStore,
)


def _m(name, mtype, fact, plan, date, *, influent=None, children=None):
    node = {"id": f"{name}{date}", "metric_name": name, "metric_type": mtype,
            "measure_type": "у.е.", "date": date, "calc_period": "д", "fact": fact,
            "plan": plan, "benchmark": None, "element": None}
    if influent is not None:
        node["influent_percent"] = influent
    if children:
        node["child_metrics"] = children
    return node


def _tree(date, perf, aht):
    return [_m("Производительность", "прямая", perf, 18, date, children=[
        _m("AHT", "обратная", aht, 180, date, influent=90),
    ])]


def _store_no_tabnum():
    data = {
        "me": {"fio": "Босс", "post": "рук", "depart": "О", "metrics": []},
        "employees": [{"fio": "Иванов", "post": "оп", "depart": "О",
                       "metrics": _tree("2026-05-04", 20, 200) + _tree("2026-05-11", 12, 300)}],
    }
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    compute_analytics(store)
    return store


def test_attribute_change_by_fio_when_tabnum_null():
    store = _store_no_tabnum()
    res = causal.attribute_change(store, "Производительность", person="Иванов")
    assert res.get("error") is None
    assert res["method"] == "algebraic"
    nodes = {c["node"] for c in res.get("contributions", [])}
    assert "AHT" in nodes  # драйвер найден, человек резолвится по ФИО


def test_resolve_person_returns_key_when_tabnum_null():
    store = _store_no_tabnum()
    # резолв по ФИО → person_key (= ФИО при null табельном), не None
    assert causal._resolve_person_tabnum(store, "Иванов") == "Иванов"
    # сотрудники перечисляются по person_key, а не схлопываются по NULL
    assert causal._employee_tabnums(store) == ["Иванов"]
