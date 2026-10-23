"""Карта отклонений deviation_map: числа и текст проблем одним SQL.

Колонка problems — то, что читает агент; остальные колонки — те же числа для
отладки на дампе.
"""
from __future__ import annotations

import os

os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

import pytest  # noqa: E402
from _fixtures import make_dataset_obj, make_metric  # noqa: E402

from langgraph_executor.aegra_agents.analyst_agent.db import core, sqlrunner  # noqa: E402

PERSON = "100500"


def _agg(name, *, median, top20, hit_rate, iqr=None, prev_median=None, level="gosb",
         level_name="по ГОСБ", size=20):
    cur = {"dt": "2026-04-13", "median": median, "top20_mean_fact": top20,
           "hit_rate": hit_rate, "iqr": iqr, "total_objects": size, "mean_fact": median}
    if prev_median is not None:
        cur["history"] = [{"dt": "2026-04-06", "median": prev_median, "mean_fact": prev_median}]
    return {"aggregate_id": None, "dataset": {
        "level": level, "level_name": level_name,
        "metrics": [{"metric_name": name, "aggregates": cur}]}}


def _series(name, facts, plan, **kw):
    """Две точки серии; rankings и прочие поля — у последней."""
    first = make_metric(name, date="2026-04-06", fact=facts[0], plan=plan,
                        **{k: v for k, v in kw.items() if k in ("metric_type", "id")})
    last = make_metric(name, date="2026-04-13", fact=facts[1], plan=plan, **kw)
    return [first, last]


def _rows(db, template="deviation_map", **params):
    res = sqlrunner.run_template(db.conn, template, person_key=PERSON, **params)
    assert res.error is None, res.error
    return {r[res.columns.index("metric")]: dict(zip(res.columns, r)) for r in res.rows}


@pytest.fixture()
def db():
    metrics = [
        # Ниже плана, но и группа план почти не выполняет.
        *_series("Продажи", (90.0, 80.0), 100.0,
                 rankings=[{"level": "gosb", "rank": "3 из 20", "percentile": 85}]),
        # План выполнен, но далеко ниже медианы.
        *_series("Качество", (104.0, 105.0), 100.0),
        # План выполнен, на уровне лучших.
        *_series("Скорость", (120.0, 130.0), 100.0),
        # Без плана — в карту не идёт.
        *_series("Звонки", (50.0, 40.0), None),
    ]
    aggregates = [
        _agg("Продажи", median=70.0, top20=95.0, hit_rate=20.0, iqr=10.0, prev_median=75.0),
        _agg("Продажи", median=72.0, top20=96.0, hit_rate=25.0, iqr=10.0,
             level="tb", level_name="по ТБ", size=300),
        _agg("Качество", median=120.0, top20=130.0, hit_rate=50.0, iqr=10.0),
        _agg("Скорость", median=100.0, top20=125.0, hit_rate=50.0, iqr=10.0),
    ]
    return core.build_run_db(make_dataset_obj(metrics), aggregates)


def test_map_signals_and_numbers(db):
    rows = _rows(db)
    assert "Звонки" not in rows
    sales = rows["Продажи"]
    assert sales["signal"] == "отклонение"
    assert sales["basis"] == "факт/план" and sales["completion"] == 80.0
    assert sales["plan_gap_pp"] == -20.0 and sales["dyn_pp"] == -10.0
    assert sales["group_level"] == "по ГОСБ"          # ближайшая группа
    assert sales["group_zone"] == "выше медианы" and sales["peer_dist"] == 1.0
    assert sales["n_levels"] == 2 and sales["n_levels_below"] == 0
    assert rows["Качество"]["signal"] == "слабый сигнал"
    assert rows["Качество"]["peer_dist"] == -1.5
    assert rows["Скорость"]["group_zone"] == "топ-20"


def test_soft_plan_threshold_is_a_param(db):
    rows = _rows(db, soft_hit_rate=40)
    assert rows["Качество"]["signal"] == "мягкий план"


def test_rr_is_forecast_divided_by_plan():
    d = core.build_run_db(make_dataset_obj(
        _series("Продажи", (80.0, 90.0), 100.0, ex=90.0, rr=110.0)))
    row = _rows(d)["Продажи"]
    assert row["basis"] == "rr" and row["completion"] == 110.0


def test_problems_text_reads_against_group(db):
    rows = _rows(db)
    assert list(rows) == ["Продажи", "Качество", "Скорость"]
    sales = rows["Продажи"]["problems"]
    assert sales.startswith("Продажи — отклонение. План: факт 80 при плане 100")
    assert "место 3 из 20" in sales
    assert "по ГОСБ (20 чел.)" in sales and "по ТБ (300 чел.)" in sales
    assert "общую проблему группы" in sales
    assert "Динамика группы по ГОСБ" in sales
    assert "ниже медианы группы — слабый сигнал" in rows["Качество"]["problems"]
    assert "на уровне лучших 20 %" in rows["Скорость"]["problems"]
    assert ".." not in sales


def test_problems_without_groups():
    d = core.build_run_db(make_dataset_obj(_series("Продажи", (90.0, 80.0), 100.0)))
    text = _rows(d)["Продажи"]["problems"]
    assert "Данных по группам нет." in text and text == text.strip()


def test_stars_limit_map_to_open_star_metrics():
    def star(received):
        return make_metric(
            "Звезда качества", date="2026-04-13", fact=None, star_received=received,
            children=[make_metric("CSAT", date="2026-04-13", fact=4.0, plan=4.5,
                                  is_star_metric=True)],
        )

    metrics = [star(0), *_series("Продажи", (90.0, 80.0), 100.0)]
    rows = _rows(core.build_run_db(make_dataset_obj(metrics)))
    assert list(rows) == ["CSAT"]
    assert rows["CSAT"]["scope"] == "звезда: Звезда качества"
    received = _rows(core.build_run_db(make_dataset_obj([star(1)])))
    assert received == {}


# --- инструмент deviation_map --------------------------------------------

def _tool_out(db, **args):
    from types import SimpleNamespace

    from langgraph_executor.aegra_agents.analyst_agent.tools.deviation_map import (
        deviation_map_text,
    )

    ctx = SimpleNamespace(db=db, person_key=PERSON, used_data_tools=False)
    out = deviation_map_text(ctx, **args)
    assert ctx.used_data_tools
    return out


def test_tool_gives_problems_and_lists_norm_by_name(db):
    out = _tool_out(db)
    assert out.startswith("Карта отклонений: Иванов Иван Иванович, показатели до 2 уровня")
    assert "- Продажи — отклонение. План: факт 80 при плане 100" in out
    assert "- Качество — слабый сигнал." in out
    assert "Звонки" not in out                     # без плана


def test_tool_depth_and_person(db):
    assert "до 1 уровня" in _tool_out(db, depth=1)
    assert "до 9 уровня" in _tool_out(db, depth=100)
    assert "до 2 уровня" in _tool_out(db, depth="")
    assert _tool_out(db, person="Неведомый") == "Сотрудник «Неведомый» не найден в данных."
    assert "Иванов" in _tool_out(db, person="100500")


def test_tool_overflow_goes_to_names(db, monkeypatch):
    from langgraph_executor.aegra_agents.analyst_agent.tools import deviation_map as mod

    monkeypatch.setattr(mod, "_TEXT_BUDGET", 600)
    out = _tool_out(db)
    assert "- Продажи — отклонение." in out
    assert "Не поместились, подробности — через metric_card: Качество (слабый сигнал)" in out


def test_tool_empty_map():
    d = core.build_run_db(make_dataset_obj(_series("Звонки", (50.0, 40.0), None)))
    assert "пуста" in _tool_out(d)


def test_names_line_is_capped(monkeypatch):
    from langgraph_executor.aegra_agents.analyst_agent.tools import deviation_map as mod

    monkeypatch.setattr(mod, "_NAMES_BUDGET", 20)
    assert mod._names_line("В норме", ["А", "Б"]) == "В норме: А, Б."
    assert mod._names_line("В норме", [f"Показатель {i}" for i in range(5)]) == (
        "В норме: Показатель 0 и ещё 4."
    )
