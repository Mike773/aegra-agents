"""Рейтинг по звёздам: поле ``rating`` на персоне (место среди сотрудников по
уровням ГОСБ/ТБ/Сбер за квартал).

Контракты:
- рейтинг разбирается только вместе со звёздами: без звёзд в базе его нет,
  инструмент не привязывается, промпт и справка схемы не меняются;
- названия уровней берутся из агрегатов (там они есть, в рейтинге — только
  коды); без агрегатов уровень называется своим кодом;
- инструмент star_rating отдаёт живой текст: квартал, места по уровням от узкой
  группы к широкой.
"""
from __future__ import annotations

import asyncio

from _fixtures import load_sample, make_dataset_obj, make_metric, make_person  # noqa: F401

from langgraph_executor.aegra_agents.analyst_agent.agent.runctx import RunContext
from langgraph_executor.aegra_agents.analyst_agent.db import core, enrich, loader
from langgraph_executor.aegra_agents.analyst_agent.db.schema_doc import build_schema_doc
from langgraph_executor.aegra_agents.analyst_agent.deviations import ledger
from langgraph_executor.aegra_agents.analyst_agent.prompts import (
    PromptContext,
    compose_system_prompt,
)
from langgraph_executor.aegra_agents.analyst_agent.tools import build_tools

PERSON = "100500"

RATING = [
    {"year": 2026, "level": "GOSB", "place": "2", "staff": 77, "quarter": 3},
    {"year": 2026, "level": "SBER", "place": "5", "staff": 7777, "quarter": 3},
    {"year": 2026, "level": "TB", "place": "2", "staff": 777, "quarter": 3},
    {"year": 2026, "level": "GOSB", "place": "4", "staff": 70, "quarter": 2},
]


def _agg(level: str, level_name: str) -> dict:
    return {
        "aggregate_id": f"a-{level}",
        "dataset": {
            "level": level, "level_name": level_name,
            "metrics": [{
                "metric_id": "M1", "metric_name": "Продажи",
                "aggregates": {"dt": "2026-04-13", "mean_fact": 90.0,
                               "total_objects": {"GOSB": 77, "TB": 777, "SBER": 7777}[level]},
            }],
        },
    }


AGGREGATES = [_agg("SBER", "Сбер"), _agg("TB", "ТБ"), _agg("GOSB", "ГОСБ")]


def _star_metrics() -> list[dict]:
    csi = make_metric("CSI", date="2026-04-13", fact=4.1, plan=4.5, is_star_metric=True)
    star = make_metric("Звезда качества", date="2026-04-13", fact=None, plan=None,
                       star_received=False, children=[csi])
    sales = make_metric("Продажи", date="2026-04-13", fact=70.0, plan=100.0)
    return [star, sales]


def _dataset(*, stars: bool = True, rating: list | None = RATING, aggregates_ids=None) -> dict:
    metrics = _star_metrics() if stars else [
        make_metric("Продажи", date="2026-04-13", fact=70.0, plan=100.0)
    ]
    extra = {}
    if rating is not None:
        extra["rating"] = rating
    if aggregates_ids:
        extra["aggregates_ids"] = aggregates_ids
    return make_dataset_obj(metrics, **extra)


def _ctx(db):
    return RunContext(db=db, person_key=PERSON, direction_key="d",
                      ledger=ledger.DeviationLedger(db, [], turn=1))


def _call(tools, name, **args):
    tool = next(t for t in tools if t.name == name)
    result = tool.invoke(args)
    return asyncio.run(result) if asyncio.iscoroutine(result) else result


def _names(tools):
    return {t.name for t in tools}


# --- loader ----------------------------------------------------------------

def test_loader_parses_rating_rows():
    parsed = loader.parse_dataset(_dataset())
    rows = {(r.person_key, r.level, r.year, r.quarter): (r.place, r.staff) for r in parsed.ratings}
    assert rows[(PERSON, "GOSB", 2026, 3)] == (2, 77)
    assert rows[(PERSON, "SBER", 2026, 3)] == (5, 7777)
    assert rows[(PERSON, "GOSB", 2026, 2)] == (4, 70)
    assert isinstance(parsed.ratings[0].place, int)


def test_loader_skips_broken_rating_entries():
    broken = [
        {"year": 2026, "level": "GOSB", "place": "n/a", "staff": 77, "quarter": 3},
        {"year": 2026, "level": None, "place": "1", "staff": 77, "quarter": 3},
        "мусор",
        {"year": 2026, "level": "TB", "place": 3, "staff": "777", "quarter": "3"},
    ]
    parsed = loader.parse_dataset(_dataset(rating=broken))
    assert [(r.level, r.place, r.staff, r.quarter) for r in parsed.ratings] == [("TB", 3, 777, 3)]


def test_loader_rating_absent_gives_no_rows():
    assert loader.parse_dataset(_dataset(rating=None)).ratings == []


# --- база --------------------------------------------------------------------

def test_rating_view_names_levels_from_aggregates():
    db = core.build_run_db(_dataset(aggregates_ids=["a-GOSB", "a-TB", "a-SBER"]), AGGREGATES)
    assert db.has_ratings
    rows = db.conn.execute(
        "SELECT level, level_name, level_order, place, staff FROM v_rating "
        "WHERE person_key = ? AND year = 2026 AND quarter = 3 ORDER BY level_order",
        (PERSON,),
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        ("GOSB", "ГОСБ", 1, 2, 77), ("TB", "ТБ", 2, 2, 777), ("SBER", "Сбер", 3, 5, 7777),
    ]


def test_rating_view_falls_back_to_level_code_without_aggregates():
    db = core.build_run_db(_dataset())
    rows = db.conn.execute(
        "SELECT level, level_name FROM v_rating WHERE quarter = 3 ORDER BY level"
    ).fetchall()
    assert [tuple(r) for r in rows] == [("GOSB", "GOSB"), ("SBER", "SBER"), ("TB", "TB")]
    orders = {r[0] for r in db.conn.execute("SELECT level_order FROM v_rating")}
    assert None not in orders


def test_rating_ignored_without_stars():
    db = core.build_run_db(_dataset(stars=False))
    assert not db.has_stars
    assert not db.has_ratings
    assert db.conn.execute("SELECT COUNT(*) FROM rating").fetchone()[0] == 0


def test_rating_rows_per_person():
    data = _dataset()
    data["employees"].append(make_person(
        _star_metrics(), tabnum=200600, fio="Петров Пётр",
        rating=[{"year": 2026, "level": "GOSB", "place": "9", "staff": 77, "quarter": 3}],
    ))
    db = core.build_run_db(data)
    rows = db.conn.execute(
        "SELECT person_key, place FROM v_rating WHERE level = 'GOSB' AND quarter = 3 "
        "ORDER BY person_key"
    ).fetchall()
    assert [tuple(r) for r in rows] == [("100500", 2), ("200600", 9)]


# --- инструмент --------------------------------------------------------------

def test_star_rating_tool_bound_only_with_ratings():
    with_rating = build_tools(_ctx(core.build_run_db(_dataset())))
    assert "star_rating" in _names(with_rating)
    stars_only = build_tools(_ctx(core.build_run_db(_dataset(rating=None))))
    assert "star_rating" not in _names(stars_only)
    no_stars = build_tools(_ctx(core.build_run_db(_dataset(stars=False))))
    assert "star_rating" not in _names(no_stars)


def test_star_rating_tool_renders_levels_latest_quarter_first():
    db = core.build_run_db(_dataset(aggregates_ids=["a-GOSB", "a-TB", "a-SBER"]), AGGREGATES)
    out = _call(build_tools(_ctx(db)), "star_rating")
    assert "3 квартал 2026" in out
    assert "ГОСБ" in out and "ТБ" in out and "Сбер" in out
    assert "2 место из 77" in out and "5 место из 7777" in out
    # Порядок уровней — от узкой группы к широкой.
    assert out.index("ГОСБ") < out.index("ТБ") < out.index("Сбер")
    # Свежий квартал раньше прошлого.
    assert out.index("3 квартал 2026") < out.index("2 квартал 2026")
    # Служебных кодов и имён полей в выдаче нет.
    assert "GOSB" not in out and "staff" not in out and "place" not in out


def test_star_rating_tool_quarter_filter_and_person():
    data = _dataset()
    data["employees"].append(make_person(
        _star_metrics(), tabnum=200600, fio="Петров Пётр Петрович",
        rating=[{"year": 2026, "level": "GOSB", "place": "9", "staff": 77, "quarter": 3}],
    ))
    tools = build_tools(_ctx(core.build_run_db(data)))
    out = _call(tools, "star_rating", quarter="2 квартал 2026")
    assert "4 место из 70" in out and "3 квартал" not in out
    out = _call(tools, "star_rating", person="Петров")
    assert "9 место из 77" in out and "2 место из 77" not in out


def test_star_rating_tool_without_rows_says_so():
    data = _dataset()
    data["employees"].append(make_person(_star_metrics(), tabnum=200600, fio="Петров Пётр"))
    tools = build_tools(_ctx(core.build_run_db(data)))
    out = _call(tools, "star_rating", person="Петров")
    assert "нет" in out.lower()


# --- промпт и справка --------------------------------------------------------

def test_stars_block_mentions_rating_only_when_present():
    db = core.build_run_db(_dataset(aggregates_ids=["a-GOSB", "a-TB", "a-SBER"]), AGGREGATES)
    block = enrich.stars_block(db, person_key=PERSON)
    assert "рейтинг" in block.lower()
    assert "ГОСБ" in block and "2 из 77" in block
    plain = enrich.stars_block(core.build_run_db(_dataset(rating=None)), person_key=PERSON)
    assert "рейтинг" not in plain.lower()


def test_prompt_rating_block_only_with_ratings():
    with_ = compose_system_prompt(PromptContext(has_stars=True, has_ratings=True))
    without = compose_system_prompt(PromptContext(has_stars=True))
    assert "рейтинг" in with_.lower()
    assert "рейтинг" not in without.lower()
    assert len(with_) > len(without)


def test_schema_doc_lists_v_rating_only_with_ratings():
    assert "v_rating" in build_schema_doc(core.build_run_db(_dataset()))
    assert "v_rating" not in build_schema_doc(core.build_run_db(_dataset(rating=None)))


def test_samples_without_rating_are_untouched():
    for name in ("sample_declining.json", "sample_good.json"):
        db = core.build_run_db(load_sample(name))
        assert not db.has_ratings
        assert "star_rating" not in _names(build_tools(_ctx(db)))


def test_sample_star_carries_rating():
    db = core.build_run_db(load_sample("sample_star.json"))
    assert db.has_stars and db.has_ratings
    tools = build_tools(_ctx(db))
    assert "star_rating" in _names(tools)


# --- уровни в описании и фильтр по уровню ------------------------------------

def _rating_tool(db):
    return next(t for t in build_tools(_ctx(db)) if t.name == "star_rating")


def test_star_rating_description_lists_levels_from_db():
    db = core.build_run_db(_dataset(aggregates_ids=["a-GOSB", "a-TB", "a-SBER"]), AGGREGATES)
    desc = _rating_tool(db).description
    # Названия из агрегатов и коды из данных — оба, от узкой группы к широкой.
    assert "ГОСБ" in desc and "ТБ" in desc and "Сбер" in desc
    assert "GOSB" in desc and "TB" in desc and "SBER" in desc
    assert desc.index("ГОСБ") < desc.index("ТБ") < desc.index("Сбер")
    # Без агрегатов — коды.
    plain = _rating_tool(core.build_run_db(_dataset())).description
    assert "GOSB" in plain and "SBER" in plain


def test_star_rating_level_filter_accepts_name_or_code():
    db = core.build_run_db(_dataset(aggregates_ids=["a-GOSB", "a-TB", "a-SBER"]), AGGREGATES)
    tool = _rating_tool(db)
    out = tool.invoke({"level": "ГОСБ"})
    assert "ГОСБ" in out and "2 место из 77" in out
    assert "ТБ" not in out and "Сбер" not in out
    out = tool.invoke({"level": "sber"})
    assert "Сбер" in out and "5 место из 7777" in out and "ГОСБ" not in out
    out = tool.invoke({"level": "в госб"})
    assert "ГОСБ" in out


def test_star_rating_level_unknown_lists_levels():
    db = core.build_run_db(_dataset(aggregates_ids=["a-GOSB", "a-TB", "a-SBER"]), AGGREGATES)
    out = _rating_tool(db).invoke({"level": "Марс"})
    assert "Марс" in out and "ГОСБ" in out and "Сбер" in out


def test_star_rating_level_name_is_not_taken_for_person():
    db = core.build_run_db(_dataset(aggregates_ids=["a-GOSB", "a-TB", "a-SBER"]), AGGREGATES)
    # Модель кладёт название уровня в person — инструмент понимает и отвечает по уровню.
    out = _rating_tool(db).invoke({"person": "ГОСБ"})
    assert "не найден" not in out and "2 место из 77" in out
