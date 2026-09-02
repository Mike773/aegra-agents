"""analyst_agent.db.text2sql: агент пишет SELECT сам, но база остаётся
read-only, а выдача — ограниченной.

Ошибки должны быть самокорректирующими: модель по тексту ошибки понимает, какие
вью и колонки есть, и переписывает запрос.
"""
from __future__ import annotations

import pytest
from _fixtures import make_dataset_obj, make_metric  # noqa: F401

from langgraph_executor.aegra_agents.analyst_agent.db import core
from langgraph_executor.aegra_agents.analyst_agent.db.text2sql import SafeQueryRunner


@pytest.fixture()
def db():
    metrics = [
        make_metric("AHT", date="2026-04-06", fact=10.0, plan=9.0),
        make_metric("AHT", date="2026-04-13", fact=12.0, plan=9.0),
    ]
    return core.build_run_db(make_dataset_obj(metrics))


@pytest.fixture()
def runner(db):
    r = SafeQueryRunner(db.conn)
    r.install()
    return r


def test_select_returns_rows(runner):
    res = runner.run("SELECT metric, date, fact FROM v_fact ORDER BY date")
    assert res.error is None
    assert res.columns == ["metric", "date", "fact"]
    assert [tuple(r) for r in res.rows] == [("AHT", "2026-04-06", 10.0), ("AHT", "2026-04-13", 12.0)]
    assert res.truncated is False


def test_with_cte_allowed(runner):
    res = runner.run("WITH x AS (SELECT fact FROM v_fact) SELECT COUNT(*) AS n FROM x")
    assert res.error is None
    assert res.rows[0][0] == 2


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO fact (fact_id) VALUES (999)",
        "UPDATE fact SET fact = 0",
        "DELETE FROM fact",
        "DROP TABLE fact",
        "CREATE TABLE evil (x INT)",
        "ALTER TABLE fact ADD COLUMN evil INT",
        "ATTACH DATABASE '/tmp/evil.db' AS evil",
        "PRAGMA table_info(fact)",
    ],
)
def test_write_and_meta_statements_rejected(runner, db, sql):
    res = runner.run(sql)
    assert res.error is not None
    assert db.conn.execute("SELECT COUNT(*) FROM fact").fetchone()[0] == 2


def test_multiple_statements_rejected(runner):
    res = runner.run("SELECT 1; DELETE FROM fact")
    assert res.error is not None
    assert "один" in res.error.lower() or "одн" in res.error.lower()


def test_row_cap_marks_truncation(db):
    runner = SafeQueryRunner(db.conn, max_rows=1)
    runner.install()
    res = runner.run("SELECT metric, date FROM v_fact")
    assert res.error is None
    assert len(res.rows) == 1
    assert res.truncated is True


def test_column_cap_rejects_wide_result(db):
    runner = SafeQueryRunner(db.conn, max_cols=3)
    runner.install()
    res = runner.run("SELECT * FROM v_fact")
    assert res.error is not None
    assert "колон" in res.error.lower()


def test_cell_truncated(db):
    runner = SafeQueryRunner(db.conn, max_cell=5)
    runner.install()
    res = runner.run("SELECT 'абвгдеёжзийклмн' AS long_text")
    assert res.rows[0][0].startswith("абвгд")
    assert len(res.rows[0][0]) <= 6


def test_unknown_column_error_lists_columns(runner):
    res = runner.run("SELECT nonexistent_column FROM v_fact")
    assert res.error is not None
    assert "v_fact" in res.error
    assert "plan_status" in res.error


def test_unknown_table_error_lists_views(runner):
    res = runner.run("SELECT * FROM nonexistent_table")
    assert res.error is not None
    assert "v_fact" in res.error and "v_metric" in res.error


def test_timeout_interrupts_heavy_query(db):
    runner = SafeQueryRunner(db.conn, timeout_s=0.05)
    runner.install()
    res = runner.run(
        "WITH RECURSIVE bomb(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM bomb) "
        "SELECT COUNT(*) FROM bomb"
    )
    assert res.error is not None
    assert "тяжёл" in res.error.lower() or "долг" in res.error.lower()


def test_writable_context_restores_guards(db):
    runner = SafeQueryRunner(db.conn)
    runner.install()
    assert runner.run("INSERT INTO deviation (id, kind, polarity, metric_name, person_key, "
                      "priority, source) VALUES ('x','k','negative','AHT','1',1.0,'agent')").error
    with runner.writable():
        db.conn.execute(
            "INSERT INTO deviation (id, kind, polarity, metric_name, person_key, priority, source) "
            "VALUES ('x','k','negative','AHT','1',1.0,'agent')"
        )
        db.conn.commit()
    assert db.conn.execute("SELECT COUNT(*) FROM deviation").fetchone()[0] == 1
    assert runner.run("DELETE FROM deviation").error is not None


def test_schema_doc_mentions_views_and_rules(db):
    doc = core.schema_doc(db)
    assert "v_fact" in doc and "v_fact_latest" in doc
    assert "plan_status" in doc
    # Правило о разных датах уровней обязано быть в документации схемы.
    assert "дат" in doc.lower()
    assert "element" in doc
