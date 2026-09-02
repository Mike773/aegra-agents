"""analyst_agent.db.loader: разбор JSON в нормализованные записи.

Ключевые контракты:
- каталог метрик дедуплицируется по нормализованному имени, структура (рёбра)
  строится независимо от дат;
- пустые листья без факта выпадают, ноль — валидный факт, звезда без факта
  сохраняется;
- у дочерних показателей даты могут отставать от родительских — факты ребёнка
  всё равно попадают в выдачу (независимость дерева от дат).
"""
from __future__ import annotations

from _fixtures import (  # noqa: F401  (sys.path bootstrap)
    load_sample,
    make_dataset_obj,
    make_metric,
    make_person,
)

from langgraph_executor.aegra_agents.analyst_agent.db import loader


def test_people_keys_and_me_first():
    data = {
        "me": {"tabnum": 1, "fio": "Босс", "metrics": []},
        "employees": [
            {"tabnum": None, "fio": "Безтабельный", "metrics": []},
            {"tabnum": None, "fio": None, "metrics": []},
        ],
    }
    parsed = loader.parse_dataset(data)
    keys = [p.person_key for p in parsed.people]
    assert keys == ["1", "Безтабельный", "person2"]
    assert parsed.people[0].is_me is True
    assert parsed.people[1].is_me is False


def test_metric_catalog_dedup_and_first_description_wins():
    m1 = make_metric("AHT", description="Среднее время обработки")
    m2 = make_metric("AHT", date="2026-03-30", description="Другое описание")
    parsed = loader.parse_dataset(make_dataset_obj([m1, m2]))
    assert list(parsed.metrics) == [loader.norm_text("AHT")]
    rec = parsed.metrics[loader.norm_text("AHT")]
    assert rec.description == "Среднее время обработки"
    assert parsed.report.description_conflicts == 1


def test_edges_independent_of_dates_first_influent_wins():
    child_old = make_metric("HOLD", date="2026-03-14", fact=5.0, influent_percent=60)
    child_new = make_metric("HOLD", date="2026-03-30", fact=6.0, influent_percent=None)
    parent = make_metric("AHT", date="2026-04-20", children=[child_old, child_new])
    parsed = loader.parse_dataset(make_dataset_obj([parent]))
    key = (loader.norm_text("AHT"), loader.norm_text("HOLD"))
    assert parsed.edges == {key: 60.0}
    # Факты ребёнка с отстающими датами присутствуют оба.
    hold_dates = sorted(
        f.date for f in parsed.facts if f.metric_norm == loader.norm_text("HOLD")
    )
    assert hold_dates == ["2026-03-14", "2026-03-30"]


def test_empty_leaf_skipped_zero_kept_star_kept():
    empty = make_metric("Пустая", fact=None)
    blank = make_metric("Пробельная", fact="  ")
    zero = make_metric("Нулевая", fact=0)
    star = make_metric("Звезда качества", fact=None, star_received="true")
    parsed = loader.parse_dataset(make_dataset_obj([empty, blank, zero, star]))
    names = {f.metric_norm for f in parsed.facts}
    assert loader.norm_text("Пустая") not in names
    assert loader.norm_text("Пробельная") not in names
    assert loader.norm_text("Нулевая") in names
    star_fact = next(
        f for f in parsed.facts if f.metric_norm == loader.norm_text("Звезда качества")
    )
    assert star_fact.star_received == 1
    assert parsed.report.skipped_empty_leaves == 2


def test_star_children_marked_star_of():
    child = make_metric("Влияющий", fact=1.0, is_star_metric=True)
    star = make_metric("Звезда", fact=None, star_received=False, children=[child])
    parsed = loader.parse_dataset(make_dataset_obj([star]))
    star_rec = parsed.metrics[loader.norm_text("Звезда")]
    child_rec = parsed.metrics[loader.norm_text("Влияющий")]
    assert star_rec.is_star is True
    assert child_rec.is_star_metric is True
    assert child_rec.star_of == "Звезда"


def test_metric_type_normalized():
    m = make_metric("Отток", metric_type="Обратный")
    parsed = loader.parse_dataset(make_dataset_obj([m]))
    assert parsed.metrics[loader.norm_text("Отток")].direction == "обратная"


def test_facts_carry_rankings_and_ext_id():
    m = make_metric(
        "AHT",
        id="M42",
        rankings=[{"rank": "458 из 500", "level": "ORG", "percentile": 8.4}],
    )
    parsed = loader.parse_dataset(make_dataset_obj([m]))
    fact = parsed.facts[0]
    assert fact.rankings == [{"rank": "458 из 500", "level": "ORG", "percentile": 8.4}]
    assert parsed.metrics[loader.norm_text("AHT")].ext_id == "M42"


def test_person_peer_pairs_dedup():
    person = make_person([make_metric("AHT")], aggregates_ids=["a1", "a1", " ", None, 2])
    parsed = loader.parse_dataset({"me": None, "employees": [person]})
    assert parsed.person_peer == [("100500", "a1"), ("100500", "2")]


def test_parse_aggregates_adds_ym_and_history():
    raw = [
        {
            "aggregate_id": "agg-1",
            "dataset": {
                "level": "OFFICE",
                "level_name": "по офису",
                "metrics": [
                    {
                        "metric_id": "M1",
                        "metric_name": "AHT",
                        "aggregates": {
                            "dt": "2026-04-06",
                            "calc_period": "неделя",
                            "mean_fact": 10.0,
                            "history": [{"dt": "2026-03-30", "mean_fact": 9.0}],
                        },
                        "children_metrics": [],
                    }
                ],
            },
        }
    ]
    rows = loader.parse_aggregates(raw)
    assert [r["is_current"] for r in rows] == [1, 0]
    assert [r["ym"] for r in rows] == ["2026-04", "2026-03"]
    assert rows[0]["level_name"] == "по офису"
    assert rows[0]["aggregate_id"] == "agg-1"
    assert loader.parse_aggregates("мусор") == []


def test_parse_rank():
    assert loader.parse_rank("458 из 500") == (458, 500)
    assert loader.parse_rank("place 3/10") == (3, 10)
    assert loader.parse_rank(None) == (None, None)
    assert loader.parse_rank("нет данных") == (None, None)


def test_sample_files_parse():
    for name in ("sample_declining.json", "sample_star.json"):
        parsed = loader.parse_dataset(load_sample(name))
        assert parsed.facts, name
        assert parsed.metrics, name
