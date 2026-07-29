"""json_analyzer_v5: загрузка серверных rankings (место в peer-группе по
уровням ORG/TERR/OFFICE) и опциональных полей ex/rr в SQLite.

Rankings приходят готовыми с сервера и живут в отдельной таблице
metric_rankings — в отличие от peer_rank/peer_percentile в metric_analytics,
которые считаются локально по загруженному датасету.
"""
from langgraph_executor.aegra_agents.json_analyzer_v5.analytics import compute_analytics
from langgraph_executor.aegra_agents.json_analyzer_v5.loader import (
    load_dataset_obj,
    parse_rank,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.sqlite_store import SqliteStore

DATE = "2026-07-30"

RANKINGS = [
    {"rank": "458 из 500", "level": "ORG", "percentile": 8.4},
    {"rank": "2 из 500", "level": "TERR", "percentile": 98},
    {"rank": "3 из 500", "level": "OFFICE", "percentile": 97},
]


def _node(name, fact, rankings=None, ex=None, rr=None, children=None):
    node = {
        "id": f"{name}-id", "metric_name": name, "metric_type": "прямая",
        "measure_type": "рубль", "date": DATE, "calc_period": "Месяц",
        "fact": fact, "plan": fact, "benchmark": None, "element": None,
        "child_metrics": children or [],
    }
    if rankings is not None:
        node["rankings"] = rankings
    if ex is not None:
        node["ex"] = ex
    if rr is not None:
        node["rr"] = rr
    return node


def _load(metrics):
    data = {"me": {"fio": "Босс", "metrics": metrics}, "employees": []}
    rows = load_dataset_obj(data)
    store = SqliteStore()
    store.load(rows)
    compute_analytics(store)
    return rows, store


def test_parse_rank():
    assert parse_rank("458 из 500") == (458, 500)
    assert parse_rank("2 из 500") == (2, 500)
    assert parse_rank("3/500") == (3, 500)
    assert parse_rank("мусор") == (None, None)
    assert parse_rank(None) == (None, None)
    assert parse_rank(42) == (None, None)


def test_rankings_carried_in_rows_and_loaded():
    rows, store = _load([_node("Продажи", 100, rankings=RANKINGS)])
    assert rows[0]["rankings"] == RANKINGS

    got = store.conn.execute(
        "SELECT metric_uid, level, rank_pos, rank_total, rank_raw, percentile "
        "FROM metric_rankings ORDER BY rank_pos"
    ).fetchall()
    assert len(got) == 3
    by_level = {r["level"]: r for r in got}
    assert set(by_level) == {"ORG", "TERR", "OFFICE"}
    org = by_level["ORG"]
    assert (org["rank_pos"], org["rank_total"]) == (458, 500)
    assert org["rank_raw"] == "458 из 500"
    assert org["percentile"] == 8.4
    assert org["metric_uid"] == rows[0]["metric_uid"]
    assert by_level["TERR"]["rank_pos"] == 2
    assert by_level["OFFICE"]["rank_pos"] == 3


def test_unparsed_rank_keeps_raw_string():
    rankings = [{"rank": "нет данных", "level": "ORG", "percentile": None}]
    _, store = _load([_node("Продажи", 100, rankings=rankings)])
    row = store.conn.execute("SELECT * FROM metric_rankings").fetchone()
    assert row["rank_pos"] is None and row["rank_total"] is None
    assert row["rank_raw"] == "нет данных"


def test_metric_without_rankings_loads_zero_rank_rows():
    rows, store = _load([_node("Продажи", 100)])
    assert rows[0]["rankings"] is None
    assert store.rankings_row_count() == 0


def test_ex_rr_loaded_and_null_when_absent():
    _, store = _load([
        _node("Продажи", 100, ex=102.1, rr=95.0),
        _node("Кредиты", 50),
    ])
    got = {
        r["metric_name"]: r
        for r in store.conn.execute("SELECT metric_name, ex, rr FROM metrics")
    }
    assert got["Продажи"]["ex"] == 102.1
    assert got["Продажи"]["rr"] == 95.0
    assert got["Кредиты"]["ex"] is None
    assert got["Кредиты"]["rr"] is None


def test_schema_overview_counts_rankings():
    _, store = _load([_node("Продажи", 100, rankings=RANKINGS)])
    overview = store.schema_overview()
    assert overview["rankings_rows"] == 3
    assert overview["peer_aggregate_rows"] == 0
