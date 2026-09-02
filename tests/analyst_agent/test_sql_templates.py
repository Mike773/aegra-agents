"""analyst_agent.db.sqlrunner: параметризованные .sql-шаблоны.

Один и тот же файл гоняют инструменты агента, обогащение промпта и отладочный
CLI — поэтому шапка файла (name/summary/params/returns/render) разбирается
машинно, а параметры биндятся как :named.
"""
from __future__ import annotations

import pytest
from _fixtures import load_sample, make_dataset_obj, make_metric  # noqa: F401

from langgraph_executor.aegra_agents.analyst_agent.db import analytics, core, sqlrunner


@pytest.fixture()
def db():
    metrics = [
        make_metric("AHT", date="2026-04-06", fact=10.0, plan=9.0),
        make_metric("AHT", date="2026-04-13", fact=12.0, plan=9.0),
        make_metric("AHT", date="2026-04-13", fact=4.0, plan=3.0, element="Разрез А"),
    ]
    d = core.build_run_db(make_dataset_obj(metrics))
    return d


def test_header_parsed():
    tpl = sqlrunner.load_sql("tool_metric_history")
    assert tpl.name == "tool_metric_history"
    assert tpl.summary
    assert "person_key" in tpl.params and tpl.params["person_key"].optional is False
    assert tpl.params["element"].optional is True
    assert tpl.returns
    assert tpl.render == "table"
    assert "SELECT" in tpl.body.upper()


def test_missing_required_param_raises(db):
    with pytest.raises(ValueError, match="person_key"):
        sqlrunner.run_template(db.conn, "tool_metric_history", metric="AHT")


def test_unknown_param_raises(db):
    with pytest.raises(ValueError, match="неизвестн"):
        sqlrunner.run_template(
            db.conn, "tool_metric_history", person_key="100500", metric="AHT", bogus=1
        )


def test_defaults_applied_and_rows_returned(db):
    res = sqlrunner.run_template(
        db.conn, "tool_metric_history", person_key="100500", metric="AHT"
    )
    assert res.error is None
    assert [r[0] for r in res.rows] == ["2026-04-06", "2026-04-13"]


def test_optional_param_filters(db):
    res = sqlrunner.run_template(
        db.conn, "tool_metric_history", person_key="100500", metric="AHT",
        element="Разрез А",
    )
    assert [r[0] for r in res.rows] == ["2026-04-13"]


def test_all_templates_run_on_samples():
    """Каждый шаблон исполняется на каждом демо-датасете без ошибок."""
    for sample in ("sample_declining_ex_rr.json", "sample_star.json"):
        d = core.build_run_db(load_sample(sample))
        person = d.conn.execute("SELECT person_key FROM person LIMIT 1").fetchone()[0]
        metric = d.conn.execute(
            "SELECT name FROM metric WHERE depth = 1 LIMIT 1"
        ).fetchone()[0]
        for tpl in sqlrunner.list_templates():
            params = {}
            for pname, spec in tpl.params.items():
                if spec.optional:
                    continue
                params[pname] = {"person_key": person, "metric": metric}.get(pname)
                assert params[pname] is not None, f"{tpl.name}: нечем заполнить {pname}"
            res = sqlrunner.run_template(d.conn, tpl.name, **params)
            assert res.error is None, f"{tpl.name} на {sample}: {res.error}"


def test_render_markdown_drops_empty_columns_and_rounds():
    res = sqlrunner.QueryResult(
        columns=["metric", "fact", "plan", "empty"],
        rows=[("AHT", 10.123456, None, None), ("HOLD", 2.0, None, None)],
    )
    md = sqlrunner.render_markdown(res)
    assert "empty" not in md
    assert "plan" not in md
    assert "10.12" in md
    assert md.startswith("| metric")


def test_render_markdown_notes_truncation():
    res = sqlrunner.QueryResult(
        columns=["a"], rows=[(1,)], truncated=True, note="показаны первые 1 строк"
    )
    assert "показаны первые" in sqlrunner.render_markdown(res)


def test_render_markdown_empty_result():
    md = sqlrunner.render_markdown(sqlrunner.QueryResult(columns=["a"], rows=[]))
    assert "не нашлось" in md.lower() or "пусто" in md.lower()


def test_render_kv():
    res = sqlrunner.QueryResult(columns=["people", "periods"], rows=[(3, 6)])
    text = sqlrunner.render_kv(res)
    assert "people: 3" in text and "periods: 6" in text
