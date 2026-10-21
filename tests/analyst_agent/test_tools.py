"""Инструменты агента: что они принимают и что отдают модели.

Выдача — человекочитаемый текст, а не JSON: он идёт прямо в контекст модели.
"""
from __future__ import annotations

import asyncio

from _fixtures import make_dataset_obj, make_metric, make_person  # noqa: F401

from langgraph_executor.aegra_agents.analyst_agent.agent.runctx import RunContext
from langgraph_executor.aegra_agents.analyst_agent.db import analytics, core
from langgraph_executor.aegra_agents.analyst_agent.tools import build_tools

PERSON = "100500"


def _ctx(dataset=None, **kw):
    dataset = dataset or make_dataset_obj([
        make_metric("Продажи", date="2026-04-06", fact=80.0, plan=100.0),
        make_metric("Продажи", date="2026-04-13", fact=70.0, plan=100.0,
                    children=[make_metric("Звонки", date="2026-04-13", fact=30.0,
                                          plan=50.0, influent_percent=60)]),
    ])
    db = core.build_run_db(dataset)
    return RunContext(
        db=db,
        person_key=PERSON,
        direction_key="test-dir",
        **kw,
    )


def _names(tools):
    return {t.name for t in tools}


def _call(tools, name, **args):
    tool = next(t for t in tools if t.name == name)
    result = tool.invoke(args)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


def test_default_tool_set():
    tools = build_tools(_ctx())
    assert _names(tools) == {
        "query_sql", "metric_card", "peer_context",
    }


def test_search_wiki_added_when_enabled():
    ctx = _ctx(easyrag_graph=object(), easyrag_enabled=True)
    assert "search_wiki" in _names(build_tools(ctx))


def test_peer_context_absent_without_aggregates_flag():
    ctx = _ctx(use_peer_aggregates=False)
    assert "peer_context" not in _names(build_tools(ctx))


def test_query_sql_returns_markdown_table():
    tools = build_tools(_ctx())
    out = _call(tools, "query_sql",
                sql="SELECT metric, fact FROM v_fact_latest WHERE element IS NULL",
                purpose="посмотрел последние значения")
    assert "| metric" in out and "Продажи" in out


def test_query_sql_rejects_writes_with_hint():
    tools = build_tools(_ctx())
    out = _call(tools, "query_sql", sql="DELETE FROM fact", purpose="удалить")
    assert "ОШИБКА SQL" in out
    assert "SELECT" in out


def test_query_sql_disabled_by_flag():
    assert "query_sql" not in _names(build_tools(_ctx(text2sql_enabled=False)))


def test_metric_card_shows_periods_children_and_verdicts():
    out = _call(build_tools(_ctx()), "metric_card", metric="Продажи")
    assert "Продажи" in out
    assert "2026-04-06" in out and "2026-04-13" in out
    assert "хуже плана" in out
    assert "Звонки" in out          # состав показателя
    assert "60" in out              # вес влияния


def test_metric_card_unknown_metric_suggests_candidates():
    out = _call(build_tools(_ctx()), "metric_card", metric="Неведомое")
    assert "не найден" in out.lower()
    assert "Продажи" in out


def test_metric_card_fuzzy_resolves_name():
    out = _call(build_tools(_ctx()), "metric_card", metric="продажи")
    assert "Продажи" in out


def test_metric_card_shows_own_date_for_lagging_child():
    dataset = make_dataset_obj([
        make_metric("Родитель", date="2026-04-20", fact=100.0, plan=90.0,
                    children=[make_metric("Ребёнок", date="2026-04-14",
                                          fact=5.0, plan=8.0)])
    ])
    out = _call(build_tools(_ctx(dataset)), "metric_card", metric="Ребёнок")
    assert "2026-04-14" in out


def test_metric_card_children_follow_person_tree():
    """Состав показателя — из дерева ТОГО человека, о ком спрашивают: у второго
    сотрудника у «Продажи» другой единственный ребёнок, чужой не подмешивается."""
    first = make_metric("Продажи", date="2026-04-13", fact=70.0, plan=100.0, children=[
        make_metric("Звонки", date="2026-04-13", fact=30.0, plan=50.0, influent_percent=60),
        make_metric("Встречи", date="2026-04-13", fact=3.0, plan=5.0, influent_percent=40),
    ])
    second = make_metric("Продажи", date="2026-04-13", fact=90.0, plan=100.0, children=[
        make_metric("Письма", date="2026-04-13", fact=12.0, plan=10.0),
    ])
    dataset = make_dataset_obj(
        [first], employees_extra=[make_person([second], tabnum=200600, fio="Второй Сотрудник")],
    )
    tools = build_tools(_ctx(dataset))
    out_first = _call(tools, "metric_card", metric="Продажи")
    assert "Звонки" in out_first and "Встречи" in out_first and "Письма" not in out_first
    assert out_first.count("Звонки") == 1
    out_second = _call(tools, "metric_card", metric="Продажи", person="Второй")
    assert "Письма" in out_second
    assert "Звонки" not in out_second and "Встречи" not in out_second


def test_peer_context_without_data_says_so():
    out = _call(build_tools(_ctx()), "peer_context", metric="Продажи")
    assert "не пришло" in out.lower() and "нечем" in out.lower()


def test_peer_context_renders_levels():
    aggregates = [{
        "aggregate_id": "a1",
        "dataset": {
            "level": "OFFICE", "level_name": "по офису",
            "metrics": [{
                "metric_id": "M1", "metric_name": "Продажи",
                "aggregates": {"dt": "2026-04-13", "mean_fact": 90.0, "median": 88.0,
                               "top20_mean_fact": 120.0, "hit_rate": 30.0,
                               "total_objects": 40},
            }],
        },
    }]
    db = core.build_run_db(
        make_dataset_obj([make_metric("Продажи", date="2026-04-13", fact=70.0, plan=100.0)],
                         aggregates_ids=["a1"]),
        aggregates,
    )
    ctx = RunContext(db=db, person_key=PERSON, direction_key="d")
    out = _call(build_tools(ctx), "peer_context", metric="Продажи")
    assert "по офису" in out
    assert "120" in out            # уровень сильнейших
    assert "30" in out             # доля выполняющих план
    # Служебные имена полей в выдачу не попадают.
    assert "top20_mean_fact" not in out and "hit_rate" not in out


def test_tool_descriptions_are_russian_and_nonempty():
    for tool in build_tools(_ctx(easyrag_graph=object(), easyrag_enabled=True)):
        assert tool.description.strip()
        assert any("а" <= ch.lower() <= "я" for ch in tool.description)


def test_metric_is_required_in_schema_sent_to_model():
    """Модель видит metric обязательным: с `default: null` и без `required`
    GigaChat часто звал metric_card вовсе без показателя (живой прогон)."""
    from langchain_gigachat.utils.function_calling import convert_to_gigachat_function

    ctx = _ctx(use_peer_aggregates=True)
    tools = {t.name: t for t in build_tools(ctx)}
    for name in ("metric_card", "peer_context"):
        params = convert_to_gigachat_function(tools[name])["parameters"]
        assert "metric" in params.get("required", []), name
        assert "default" not in params["properties"]["metric"], name


def test_metric_card_without_metric_answers_with_hint():
    """Обязательность — только для модели: пропуск даёт подсказку, а не ошибку."""
    out = _call(build_tools(_ctx()), "metric_card")
    assert "Не указано название показателя" in out
    assert "Продажи" in out


def _ex_rr_ctx():
    return _ctx(make_dataset_obj([
        make_metric("Продажи", date="2026-04-13", fact=70.0, plan=100.0, ex=70.0, rr=84.5,
                    children=[
                        make_metric("Звонки", date="2026-04-13", fact=30.0, plan=50.0,
                                    ex=60.0, rr=36.2, influent_percent=60),
                        make_metric("Визиты", date="2026-04-13", fact=5.0, plan=8.0,
                                    influent_percent=40),
                    ]),
        make_metric("Продажи", date="2026-04-13", fact=40.0, plan=50.0, ex=80.0, rr=48.0,
                    element="Кредиты"),
    ]))


def test_metric_card_shows_ex_and_rr_everywhere_they_exist():
    """Выполнение плана и run rate — в ряду, в разрезах и в составе, если пришли."""
    tools = build_tools(_ex_rr_ctx())
    card = _call(tools, "metric_card", metric="Продажи")
    periods = card.split("ЗНАЧЕНИЯ ПО ПЕРИОДАМ")[1].split("\n\n")[0]
    assert "выполнение плана 70 %" in periods and "run rate 84.5" in periods
    elements = card.split("РАЗРЕЗЫ")[1].split("\n\n")[0]
    assert "выполнение плана 80 %" in elements and "run rate 48" in elements
    children = card.split("СОСТАВ")[1]
    calls = next(line for line in children.splitlines() if "Звонки" in line)
    assert "выполнение плана 60 %" in calls and "run rate 36.2" in calls
    # Нет ex/rr у показателя — нет и упоминания.
    visits = next(line for line in children.splitlines() if "Визиты" in line)
    assert "выполнение плана" not in visits and "run rate" not in visits


def test_metric_card_element_series_shows_ex_and_rr():
    out = _call(build_tools(_ex_rr_ctx()), "metric_card", metric="Продажи", element="Кредиты")
    assert "РАЗРЕЗ «Кредиты»" in out
    assert "выполнение плана 80 %" in out and "run rate 48" in out
