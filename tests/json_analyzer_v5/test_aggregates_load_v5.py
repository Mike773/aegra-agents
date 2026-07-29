"""json_analyzer_v5: загрузка batch-агрегатов peer-групп (уровни ORG/TERR/
OFFICE) в таблицу peer_aggregates.

Вход — отдельный payload из GetBatchAgentAggregateDatasetByFiltersComponent:
[{"dataset": {"level", "metrics": [...]}}]; у метрики aggregates (текущий
срез + history) и дети в children_metrics. Все срезы одного узла делят
node_uid; текущий помечен is_current=1.
"""
from langgraph_executor.aegra_agents.json_analyzer_v5.loader import (
    load_aggregates_obj,
    load_dataset_obj,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.nodes import (
    _parse_aggregates,
    _prepare_store,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.sqlite_store import SqliteStore


def _agg(dt, mean_fact, history=None):
    slice_ = {
        "cv": 75.5, "dt": dt, "iqr": 8442485, "median": 14894246,
        "mean_ex": 102.1, "hit_rate": 43.8, "mean_fact": mean_fact,
        "mean_plan": 15995040, "calc_period": "Месяц", "total_objects": 496,
        "top20_mean_fact": 30322068,
    }
    if history is not None:
        slice_["history"] = history
    return slice_


def _metric(metric_id, name, children=None, children_key="children_metrics"):
    node = {
        "metric_id": metric_id,
        "metric_name": name,
        "aggregates": _agg(
            "2026-07-30",
            16335807,
            history=[_agg("2026-05-31", 33313488), _agg("2026-04-30", 38507555)],
        ),
    }
    if children:
        node[children_key] = children
    return node


def _payload(levels=("ORG", "TERR", "OFFICE")):
    return [
        {
            "dataset": {
                "level": lvl,
                "metrics": [
                    _metric(
                        "377722562",
                        "Метрика тест",
                        children=[
                            _metric("377722562", "Пример дочерней метрики"),
                            _metric("350022538", "Еще один пример дочерней метрики"),
                        ],
                    )
                ],
            }
        }
        for lvl in levels
    ]


def test_flatten_counts_and_fields():
    rows = load_aggregates_obj(_payload())
    # 3 уровня × 3 узла (родитель + 2 ребёнка) × 3 среза (current + 2 history).
    assert len(rows) == 27
    current = [r for r in rows if r["is_current"] == 1]
    assert len(current) == 9
    r = current[0]
    assert r["level"] == "ORG"
    assert r["dt"] == "2026-07-30"
    assert r["mean_fact"] == 16335807
    assert r["mean_plan"] == 15995040
    assert r["mean_ex"] == 102.1
    assert r["median"] == 14894246
    assert r["hit_rate"] == 43.8
    assert r["top20_mean_fact"] == 30322068
    assert r["iqr"] == 8442485
    assert r["cv"] == 75.5
    assert r["total_objects"] == 496
    assert r["calc_period"] == "Месяц"
    assert {row["level"] for row in rows} == {"ORG", "TERR", "OFFICE"}


def test_children_get_parent_uid_and_depth():
    rows = load_aggregates_obj(_payload(levels=("ORG",)))
    parents = {r["node_uid"] for r in rows if r["depth"] == 1}
    children = [r for r in rows if r["depth"] == 2]
    assert len(parents) == 1
    assert children  # оба ребёнка развёрнуты
    assert all(r["parent_node_uid"] in parents for r in children)
    names = {r["metric_name"] for r in children}
    assert names == {"Пример дочерней метрики", "Еще один пример дочерней метрики"}


def test_child_metrics_key_also_accepted():
    payload = [
        {
            "dataset": {
                "level": "ORG",
                "metrics": [
                    _metric(
                        "1", "Родитель",
                        children=[_metric("2", "Ребёнок")],
                        children_key="child_metrics",
                    )
                ],
            }
        }
    ]
    rows = load_aggregates_obj(payload)
    assert {r["metric_name"] for r in rows} == {"Родитель", "Ребёнок"}


def test_node_uid_shared_between_current_and_history():
    rows = load_aggregates_obj(_payload(levels=("ORG",)))
    store = SqliteStore()
    store.load_aggregates(rows)
    series = store.conn.execute(
        "SELECT dt, is_current, mean_fact FROM peer_aggregates "
        "WHERE node_uid = 1 ORDER BY dt"
    ).fetchall()
    assert [r["dt"] for r in series] == ["2026-04-30", "2026-05-31", "2026-07-30"]
    assert [r["is_current"] for r in series] == [0, 0, 1]
    assert store.aggregates_row_count() == 9


def test_broken_or_empty_input_is_not_fatal():
    assert load_aggregates_obj(None) == []
    assert load_aggregates_obj([]) == []
    assert load_aggregates_obj({"dataset": {}}) == []  # не список
    assert load_aggregates_obj(["мусор", {"без_dataset": 1}]) == []
    assert load_aggregates_obj([{"dataset": {"level": "ORG", "metrics": ["x"]}}]) == []
    assert _parse_aggregates("{битый json") == []
    assert _parse_aggregates(None) == []


def test_prepare_store_loads_both_datasets():
    data = {
        "me": {
            "fio": "Босс",
            "metrics": [
                {
                    "id": "m1", "metric_name": "Продажи", "metric_type": "прямая",
                    "measure_type": "рубль", "date": "2026-07-30",
                    "calc_period": "Месяц", "fact": 100, "plan": 100,
                    "benchmark": None, "element": None, "child_metrics": [],
                }
            ],
        },
        "employees": [],
    }
    rows = load_dataset_obj(data)
    agg_rows = load_aggregates_obj(_payload())
    store = _prepare_store(rows, agg_rows)
    overview = store.schema_overview()
    assert overview["total_metric_rows"] == 1
    assert overview["peer_aggregate_rows"] == 27
    # Без агрегатов — поведение как у v3.
    store_v3 = _prepare_store(rows)
    assert store_v3.schema_overview()["peer_aggregate_rows"] == 0
