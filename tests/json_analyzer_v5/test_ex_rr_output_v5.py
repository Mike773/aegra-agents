"""json_analyzer_v5: ex (% выполнения плана) и rr (RunRate) доходят до LLM.

В v4 оба поля грузились в таблицу metrics, но ни одна выборка их не выбирала —
до инструментов и, значит, до модели они не доходили. Здесь проверяем сквозной
путь: JSON → SQLite → SELECT → рендер инструмента. Поля опциональны: когда их
нет, колонка просто отпадает (_md_table выкидывает пустые столбцы).
"""
from langgraph_executor.aegra_agents.json_analyzer_v5.analytics import compute_analytics
from langgraph_executor.aegra_agents.json_analyzer_v5.loader import load_dataset_obj
from langgraph_executor.aegra_agents.json_analyzer_v5.sqlite_store import SqliteStore
from langgraph_executor.aegra_agents.json_analyzer_v5.tools import (
    _render_rows,
    _render_tree,
)

DATE = "2026-07-30"


def _node(name, fact, ex=None, rr=None, children=None):
    node = {
        "id": f"{name}-id", "metric_name": name, "metric_type": "прямая",
        "measure_type": "рубль", "date": DATE, "calc_period": "Месяц",
        "fact": fact, "plan": fact, "benchmark": None, "element": None,
        "child_metrics": children or [],
    }
    if ex is not None:
        node["ex"] = ex
    if rr is not None:
        node["rr"] = rr
    return node


def _store(metrics):
    data = {"me": {"fio": "Босс", "metrics": metrics}, "employees": []}
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    compute_analytics(store)
    return store


def test_ex_rr_in_flat_rows():
    store = _store([_node("Продажи", 100, ex=87.5, rr=120)])
    result = store.get_metric("Продажи")
    row = result["rows"][0]
    assert row["ex"] == 87.5
    assert row["rr"] == 120
    rendered = _render_rows(result)
    assert "выполнение плана, %" in rendered
    assert "87.5 %" in rendered
    assert "run rate" in rendered
    assert "120 рубль" in rendered


def test_ex_rr_in_metric_tree():
    store = _store([
        _node("Продажи", 100, ex=87.5, children=[_node("Лиды", 40, rr=55)])
    ])
    rendered = _render_tree(store.metric_tree("Продажи"))
    assert "выполнение плана 87.5 %" in rendered
    assert "run rate 55 рубль" in rendered


def test_columns_absent_when_no_ex_rr():
    """Поля приходят не всегда — тогда колонок в таблице просто нет."""
    store = _store([_node("Продажи", 100)])
    rendered = _render_rows(store.get_metric("Продажи"))
    assert "выполнение плана, %" not in rendered
    assert "run rate" not in rendered
