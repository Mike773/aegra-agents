"""json_analyzer_v2: (1) аналитика по бенчмарку НЕ строится; (2) pop-сравнения
считаются только у метрик с указанным планом."""
from langgraph_executor.aegra_agents.json_analyzer_v2.analytics import compute_analytics
from langgraph_executor.aegra_agents.json_analyzer_v2.loader import load_dataset_obj
from langgraph_executor.aegra_agents.json_analyzer_v2.sqlite_store import SqliteStore
from langgraph_executor.aegra_agents.json_analyzer_v2.store_cache import EmbeddingIndex
from langgraph_executor.aegra_agents.json_analyzer_v2.tools import build_tools


def _m(name, fact, plan, bench, date):
    return {"id": f"{name}{date}", "metric_name": name, "metric_type": "прямая",
            "measure_type": "у.е.", "date": date, "calc_period": "д",
            "fact": fact, "plan": plan, "benchmark": bench, "element": None}


def _store():
    data = {"me": {"fio": "Босс", "metrics": []}, "employees": [{"fio": "Иванов", "metrics": [
        _m("СПланом", 20, 18, 16, "2026-05-04"), _m("СПланом", 12, 18, 16, "2026-05-11"),
        _m("БезПлана", 5, None, 3, "2026-05-04"), _m("БезПлана", 9, None, 3, "2026-05-11"),
    ]}]}
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    compute_analytics(store)
    return store


def _row(store, name):
    return store.conn.execute(
        "SELECT a.benchmark_status bs, a.benchmark_dev_abs bda, a.benchmark_dev_pct bdp, "
        "a.pop_change_pct pct, a.pop_change_abs abs, a.pop_status ps "
        "FROM metrics m JOIN metric_analytics a ON a.metric_uid = m.metric_uid "
        "WHERE m.metric_name = ? AND m.date = '2026-05-11'", (name,)).fetchone()


def test_no_benchmark_analytics():
    store = _store()
    for name in ("СПланом", "БезПлана"):
        r = _row(store, name)
        assert r["bs"] is None and r["bda"] is None and r["bdp"] is None


def test_pop_only_for_metrics_with_plan():
    store = _store()
    planned = _row(store, "СПланом")
    assert planned["pct"] == -40.0 and planned["ps"] == "ухудшение"
    no_plan = _row(store, "БезПлана")
    assert no_plan["pct"] is None and no_plan["abs"] is None and no_plan["ps"] is None


def test_benchmark_column_absent_in_render():
    store = _store()
    tools = build_tools(store, EmbeddingIndex([]), lambda q: [0.0])
    get_metric = [t for t in tools if t.name == "get_metric"][0]
    out = get_metric.invoke({"metric": "СПланом", "date": "2026-05-11"})
    assert "бенчмарк" not in out.lower()  # пустой бенчмарк-столбец выпадает
    assert "СПланом" in out
