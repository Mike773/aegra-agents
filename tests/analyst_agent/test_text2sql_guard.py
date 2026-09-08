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


def test_schema_doc_names_dialect_and_date_functions(db):
    """Модель считала «оставшиеся недели» в уме и каждый раз по-разному, а при
    попытке в SQL звала DUAL и WEEKS_BETWEEN: справка обязана назвать диалект и
    функции дат, а сами функции — проходить через охрану запросов."""
    doc = core.schema_doc(db)
    assert "SQLite" in doc
    assert "julianday" in doc and "LAG" in doc
    runner = SafeQueryRunner(db.conn)
    runner.install()
    res = runner.run(
        "SELECT CAST((julianday('2026-06-30') - julianday('2026-05-11')) / 7 AS INTEGER) AS w"
    )
    assert res.error is None
    assert res.rows[0][0] == 7


def test_schema_doc_prints_shared_columns_once(db):
    """v_fact_latest и v_series — те же колонки, что v_fact; печатать список из
    полусотни имён трижды незачем: это треть всей схемы в промпте."""
    doc = core.schema_doc(db)
    assert doc.count("fact_id, person_key, fio") == 1
    assert "v_fact_latest: те же колонки, что v_fact" in doc
    assert "v_series: те же колонки, что v_fact, плюс prev_fact, prev_date, first_fact" in doc


def test_schema_doc_has_no_dataset_facts_block(db):
    """Люди, периоды и число показателей уже есть в блоке «СОСТАВ ДАННЫХ»;
    в схеме их дублировать не нужно."""
    doc = core.schema_doc(db)
    assert "ЧТО В ЭТОЙ БАЗЕ" not in doc
    assert "Люди:" not in doc


def test_schema_doc_star_and_rank_hints_survive_without_facts_block():
    """Подсказки про v_star и rank_raw жили в блоке фактов — после его удаления
    они должны остаться в правилах, но только когда такие данные есть."""
    plain = core.build_run_db(make_dataset_obj([make_metric("AHT", fact=1.0, plan=1.0)]))
    assert "v_star" not in core.schema_doc(plain)
    child = make_metric("CSI", fact=4.1, plan=4.5, is_star_metric=True)
    star = make_metric("Звезда качества", fact=None, star_received=False, children=[child])
    starred = core.build_run_db(make_dataset_obj([star]))
    doc = core.schema_doc(starred)
    assert "v_star — строка на звезду" in doc


# --- схема живёт в описании query_sql, а не в системном промпте -----------

def _sql_ctx(db):
    from langgraph_executor.aegra_agents.analyst_agent.agent.runctx import RunContext
    from langgraph_executor.aegra_agents.analyst_agent.deviations import builder, ledger

    devs = builder.build_deviations(db, focus_person_key="100500")
    return RunContext(db=db, person_key="100500", direction_key="d",
                      ledger=ledger.DeviationLedger(db, devs, turn=1))


def test_query_sql_description_carries_schema(db):
    """Схема нужна только тому, кто пишет SQL: она в описании инструмента."""
    from langgraph_executor.aegra_agents.analyst_agent.tools.query_sql import make_query_sql

    desc = make_query_sql(_sql_ctx(db)).description
    assert "СХЕМА ДАННЫХ" in desc
    assert "v_fact(" in desc and "v_fact_latest" in desc
    assert "julianday" in desc and "SQLite" in desc
    # Примеры — не текстом в описании, а few-shot примерами функции.
    assert "ПРИМЕРЫ ЗАПРОСОВ" not in desc


def test_query_sql_few_shot_examples_run_on_the_db(db):
    """Примеры уходят в GigaChat как few_shot_examples функции: запрос
    руководителя → аргументы вызова. Каждый пример обязан исполняться."""
    from langchain_gigachat.utils.function_calling import convert_to_gigachat_function

    from langgraph_executor.aegra_agents.analyst_agent.tools.query_sql import make_query_sql

    tool = make_query_sql(_sql_ctx(db))
    spec = convert_to_gigachat_function(tool)
    examples = spec["few_shot_examples"]
    assert len(examples) >= 6
    for ex in examples:
        assert ex["request"] and set(ex["params"]) == {"sql", "purpose"}
        assert "100500" in ex["params"]["sql"]      # подставлен реальный человек
        out = tool.invoke(ex["params"])
        assert not out.startswith("ОШИБКА SQL"), (ex["request"], out)


def test_schema_doc_can_omit_examples(db):
    assert "ПРИМЕРЫ ЗАПРОСОВ" in core.schema_doc(db)
    assert "ПРИМЕРЫ ЗАПРОСОВ" not in core.schema_doc(db, examples=False)
