"""analyst_agent.db.core: нормализованная схема, вью, дерево, last-of-series.

Главное правило: «последний период» считается на серию (персона × метрика ×
разрез), а не по глобальной дате датасета — дочерний показатель с отстающей
датой остаётся в v_fact_latest со своей датой.
"""
from __future__ import annotations

import sqlite3

import pytest
from _fixtures import load_sample, make_dataset_obj, make_metric, make_person  # noqa: F401

from langgraph_executor.aegra_agents.analyst_agent.db import core

EXPECTED_TABLES = {
    "person", "metric", "metric_edge", "metric_tree", "period", "fact", "fact_analytics",
    "ranking", "peer_aggregate", "person_peer", "deviation",
}
EXPECTED_VIEWS = {
    "v_person", "v_metric", "v_tree", "v_fact", "v_fact_latest", "v_peer_latest",
    "v_star", "v_element", "v_deviation",
}


def _names(conn: sqlite3.Connection, kind: str) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,)).fetchall()
    return {r[0] for r in rows}


def _tree_dataset():
    hold_old = make_metric("HOLD", date="2026-04-14", fact=5.0, plan=4.0, influent_percent=60)
    hold_prev = make_metric("HOLD", date="2026-04-07", fact=4.0, plan=4.0)
    talk = make_metric("TALK", date="2026-04-14", fact=7.0, plan=8.0, influent_percent=40)
    aht_new = make_metric("AHT", date="2026-04-20", fact=12.0, plan=10.0,
                          children=[hold_old, hold_prev, talk])
    aht_prev = make_metric("AHT", date="2026-04-13", fact=11.0, plan=10.0)
    return make_dataset_obj([aht_new, aht_prev])


def test_tables_and_views_exist_and_select():
    db = core.build_run_db(_tree_dataset())
    assert EXPECTED_TABLES <= _names(db.conn, "table")
    views = _names(db.conn, "view")
    assert EXPECTED_VIEWS <= views
    for v in views:
        db.conn.execute(f"SELECT * FROM {v} LIMIT 1").fetchall()


def test_last_of_series_independent_of_global_date():
    db = core.build_run_db(_tree_dataset())
    latest = db.conn.execute(
        "SELECT metric, date FROM v_fact_latest ORDER BY metric"
    ).fetchall()
    assert [tuple(r) for r in latest] == [
        ("AHT", "2026-04-20"),
        ("HOLD", "2026-04-14"),
        ("TALK", "2026-04-14"),
    ]
    global_latest = db.conn.execute(
        "SELECT date FROM period WHERE is_latest = 1"
    ).fetchall()
    assert [r[0] for r in global_latest] == ["2026-04-20"]


def test_tree_structure_and_share_pct():
    db = core.build_run_db(_tree_dataset())
    rows = db.conn.execute(
        "SELECT name, depth, path, parent_name FROM v_metric ORDER BY name"
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        ("AHT", 1, "AHT", None),
        ("HOLD", 2, "AHT > HOLD", "AHT"),
        ("TALK", 2, "AHT > TALK", "AHT"),
    ]
    shares = db.conn.execute(
        "SELECT child, influent_percent, share_pct FROM v_tree ORDER BY child"
    ).fetchall()
    assert [tuple(r) for r in shares] == [("HOLD", 60.0, 60.0), ("TALK", 40.0, 40.0)]


def test_duplicate_fact_via_two_parents_counted_once():
    shared = make_metric("Общая", date="2026-04-20", fact=1.0)
    a = make_metric("A", date="2026-04-20", fact=2.0, children=[shared])
    b = make_metric("B", date="2026-04-20", fact=3.0, children=[dict(shared)])
    db = core.build_run_db(make_dataset_obj([a, b]))
    n = db.conn.execute("SELECT COUNT(*) FROM v_fact WHERE metric = 'Общая'").fetchone()[0]
    assert n == 1
    assert db.report.duplicate_facts == 1
    parents = db.conn.execute(
        "SELECT parent FROM v_tree WHERE child = 'Общая' ORDER BY parent"
    ).fetchall()
    assert [r[0] for r in parents] == ["A", "B"]


def test_period_idx_and_series_lag():
    db = core.build_run_db(_tree_dataset())
    if sqlite3.sqlite_version_info < (3, 25):
        pytest.skip("window functions unavailable")
    rows = db.conn.execute(
        "SELECT date, fact, prev_fact FROM v_series WHERE metric = 'HOLD' ORDER BY period_idx"
    ).fetchall()
    assert [tuple(r) for r in rows] == [("2026-04-07", 4.0, None), ("2026-04-14", 5.0, 4.0)]


def test_tree_is_per_person():
    """Дерево у каждого сотрудника своё: у одного HOLD лежит под AHT, у другого
    HOLD — самостоятельный показатель. Чужая структура не должна протекать."""
    hold_a = make_metric("HOLD", date="2026-04-14", fact=5.0, plan=4.0, influent_percent=60)
    talk_a = make_metric("TALK", date="2026-04-14", fact=7.0, plan=8.0, influent_percent=40)
    aht_a = make_metric("AHT", date="2026-04-20", fact=12.0, plan=10.0, children=[hold_a, talk_a])
    hold_b = make_metric("HOLD", date="2026-04-14", fact=3.0, plan=4.0)
    aht_b = make_metric("AHT", date="2026-04-20", fact=9.0, plan=10.0)
    data = make_dataset_obj(
        [aht_a], tabnum=1, fio="Первый",
        employees_extra=[make_person([aht_b, hold_b], tabnum=2, fio="Второй")],
    )
    db = core.build_run_db(data)
    tree = db.conn.execute(
        "SELECT person_key, parent, child, share_pct FROM v_tree ORDER BY person_key, child"
    ).fetchall()
    assert [tuple(r) for r in tree] == [("1", "AHT", "HOLD", 60.0), ("1", "AHT", "TALK", 40.0)]
    facts = db.conn.execute(
        "SELECT person_key, depth, path, parent_name FROM v_fact_latest "
        "WHERE metric = 'HOLD' ORDER BY person_key"
    ).fetchall()
    assert [tuple(r) for r in facts] == [
        ("1", 2, "AHT > HOLD", "AHT"),
        ("2", 1, "HOLD", None),
    ]
    # Сводный каталог сохраняет доминирующего родителя.
    row = db.conn.execute("SELECT depth, path FROM v_metric WHERE name = 'HOLD'").fetchone()
    assert tuple(row) == (2, "AHT > HOLD")


def test_star_view_and_flags():
    """Строка на звезду: только влияющие показатели (is_star_metric), одной
    строкой с фактом, планом и процентом выполнения."""
    child = make_metric("Влияющий", fact=1.0, plan=2.0, is_star_metric=True)
    plain = make_metric("Справочный", fact=44.0, plan=None)
    star = make_metric("Звезда", fact=None, star_received=False, children=[child, plain])
    won = make_metric("Звезда продаж", fact=None, star_received=True, children=[
        make_metric("Конверсия", fact=13.4, plan=12.0, is_star_metric=True),
        make_metric("Кросс", fact=1.3, plan=1.2, ex=108.3, is_star_metric=True),
    ])
    db = core.build_run_db(make_dataset_obj([star, won]))
    assert db.has_stars is True
    rows = db.conn.execute(
        "SELECT star, received, n_metrics, metrics FROM v_star ORDER BY star"
    ).fetchall()
    assert len(rows) == 2
    assert tuple(rows[0])[:3] == ("Звезда", 0, 1)
    assert rows[0]["metrics"] == "Влияющий: факт 1 при плане 2, выполнение 50 %, хуже плана"
    assert "Справочный" not in rows[0]["metrics"]
    assert tuple(rows[1])[:3] == ("Звезда продаж", 1, 2)
    assert rows[1]["metrics"] == (
        "Конверсия: факт 13.4 при плане 12, выполнение 111.7 %, лучше плана; "
        "Кросс: факт 1.3 при плане 1.2, выполнение 108.3 %, лучше плана"
    )
    assert core.build_run_db(_tree_dataset()).has_stars is False


def test_star_view_only_star_rows_and_own_children():
    """Обычные показатели в v_star не попадают, а состав звезды берётся из
    дерева ЭТОГО человека: у второго сотрудника у звезды другой набор."""
    star_a = make_metric("Звезда", fact=None, star_received=False, children=[
        make_metric("CSI", fact=4.1, plan=4.5, is_star_metric=True),
    ])
    star_b = make_metric("Звезда", fact=None, star_received=True, children=[
        make_metric("FCR", fact=78.0, plan=75.0, is_star_metric=True),
    ])
    plain = make_metric("AHT", fact=12.0, plan=10.0)
    data = make_dataset_obj(
        [star_a, plain], tabnum=1,
        employees_extra=[make_person([star_b, plain], tabnum=2, fio="Второй")],
    )
    db = core.build_run_db(data)
    rows = db.conn.execute(
        "SELECT person_key, star, received, metrics FROM v_star ORDER BY person_key"
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        ("1", "Звезда", 0, "CSI: факт 4.1 при плане 4.5, выполнение 91.1 %, хуже плана"),
        ("2", "Звезда", 1, "FCR: факт 78 при плане 75, выполнение 104 %, лучше плана"),
    ]


def test_rankings_loaded():
    m = make_metric("AHT", rankings=[{"rank": "458 из 500", "level": "ORG", "percentile": 8.4}])
    db = core.build_run_db(make_dataset_obj([m]))
    row = db.conn.execute(
        "SELECT level, rank_pos, rank_total, percentile FROM ranking"
    ).fetchone()
    assert tuple(row) == ("ORG", 458, 500, 8.4)


def _aggregates():
    def entry(agg_id, level, level_name, total):
        return {
            "aggregate_id": agg_id,
            "dataset": {
                "level": level,
                "level_name": level_name,
                "metrics": [
                    {
                        "metric_id": "M1",
                        "metric_name": "aht",   # регистр другой — матчится по norm
                        "aggregates": {
                            "dt": "2026-04-20", "mean_fact": 10.0, "median": 9.0,
                            "top20_mean_fact": 8.0, "hit_rate": 40.0,
                            "total_objects": total,
                            "history": [{"dt": "2026-03-20", "mean_fact": 11.0}],
                        },
                    }
                ],
            },
        }
    return [entry("a-org", "ORG", "по банку", 5000), entry("a-off", "OFFICE", "по офису", 20)]


def test_peer_aggregates_matched_and_ordered(monkeypatch):
    monkeypatch.delenv("PEER_LEVELS", raising=False)
    from langgraph_executor.aegra_agents.shared import peer_levels
    peer_levels.reset_cache()
    person = make_person([make_metric("AHT", date="2026-04-20")], aggregates_ids=["a-off", "a-org"])
    db = core.build_run_db({"me": None, "employees": [person]}, _aggregates())
    rows = db.conn.execute(
        "SELECT level, level_name, level_order, metric_id IS NOT NULL, is_current "
        "FROM peer_aggregate ORDER BY level_order, is_current DESC"
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        ("OFFICE", "по офису", 1, 1, 1), ("OFFICE", "по офису", 1, 1, 0),
        ("ORG", "по банку", 2, 1, 1), ("ORG", "по банку", 2, 1, 0),
    ]
    peer = db.conn.execute(
        "SELECT person_key, metric, level_name, mean_fact FROM v_peer_latest ORDER BY level_order"
    ).fetchall()
    assert [tuple(r) for r in peer] == [
        ("100500", "AHT", "по офису", 10.0), ("100500", "AHT", "по банку", 10.0),
    ]


def test_resolve_metric_ladder():
    db = core.build_run_db(_tree_dataset())
    assert db.resolve_metric("AHT").name == "AHT"
    assert db.resolve_metric("  aht ").name == "AHT"
    assert db.resolve_metric("HOLD TIME").name == "HOLD"
    assert db.resolve_metric("несуществующая метрика") is None


def test_ru_lower_function_registered():
    db = core.build_run_db(_tree_dataset())
    assert db.conn.execute("SELECT ru_lower('ЁЖИК')").fetchone()[0] == "ёжик"


def test_samples_build():
    for name in ("sample_declining_ex_rr.json", "sample_star.json"):
        db = core.build_run_db(load_sample(name))
        assert db.conn.execute("SELECT COUNT(*) FROM v_fact").fetchone()[0] > 0
