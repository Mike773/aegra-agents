"""rank_elements в json_analyzer_causal: лучшие/худшие разрезы метрики по значению
с учётом направления (план/бенчмарк не нужны)."""
from langgraph_executor.aegra_agents.json_analyzer_causal.analytics import (
    compute_analytics,
    rank_elements,
)
from langgraph_executor.aegra_agents.json_analyzer_causal.loader import (
    load_dataset_obj,
)
from langgraph_executor.aegra_agents.json_analyzer_causal.sqlite_store import (
    SqliteStore,
)
from langgraph_executor.aegra_agents.json_analyzer_causal.tools import (
    _render_rank_elements,
)


def _store(direction, facts, date="2026-05-11"):
    metrics = [
        {"id": el, "metric_name": "Влияние", "metric_type": direction,
         "measure_type": "секунда", "date": date, "calc_period": "д",
         "fact": f, "plan": None, "benchmark": None, "element": el}
        for el, f in facts.items()
    ]
    data = {"me": {"fio": "Босс", "metrics": []},
            "employees": [{"fio": "Иванов", "metrics": metrics}]}
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    compute_analytics(store)
    return store


def test_rank_elements_inverse_no_plan():
    s = _store("обратная", {"A": 25, "B": -20, "C": 5})
    r = rank_elements(s, "Влияние")
    assert r.get("error") is None and r["count"] == 3
    assert r["worst"][0]["element"] == "A"   # обратная: выше=хуже
    assert r["best"][0]["element"] == "B"


def test_rank_elements_direct_no_plan():
    s = _store("прямая", {"A": 25, "B": -20, "C": 5})
    r = rank_elements(s, "Влияние")
    assert r["best"][0]["element"] == "A"    # прямая: выше=лучше
    assert r["worst"][0]["element"] == "B"


def test_rank_elements_render_phrased():
    out = _render_rank_elements({
        "metric": "Влияние", "metric_type": "обратная", "date": "2026-05-11",
        "person_fio": "Иванов", "count": 8, "top": 5,
        "worst": [{"element": "A", "fact": 25, "measure_type": "секунда"}],
        "best": [{"element": "B", "fact": -20, "measure_type": "секунда"}],
    })
    assert "Худшие" in out and "A 25" in out and "Лучшие" in out
