"""analyst_agent.db.enrich: блоки промпта из SQL-статистики.

Требования:
- каталог показывает ВСЕ метрики 1-го и 2-го уровня, а у каждой метрики 2-го
  уровня — счётчики метрик более низких слоёв («10 метрик 3-го, 20 4-го»);
- на прод-масштабе (100 метрик L1 × 5 уровней) блоки укладываются в бюджет;
- пустые секции опускаются: нет звёзд/агрегатов/разрезов — ни слова о них.
"""
from __future__ import annotations

import pytest
from _fixtures import (  # noqa: F401
    load_sample,
    make_dataset_obj,
    make_metric,
    make_person,
    make_synthetic_dataset,
)

from langgraph_executor.aegra_agents.analyst_agent.db import analytics, core, enrich


def _db(dataset, aggregates=None):
    d = core.build_run_db(dataset, aggregates)
    analytics.compute_analytics(d.conn)
    return d


@pytest.fixture()
def simple_db():
    metrics = [
        make_metric("Производительность", date="2026-04-06", fact=18.0, plan=20.0),
        make_metric(
            "Производительность", date="2026-04-13", fact=16.0, plan=20.0,
            children=[
                make_metric("AHT", date="2026-04-13", fact=310.0, plan=300.0,
                            metric_type="обратная", influent_percent=70),
                make_metric("Звонки", date="2026-04-13", fact=40.0, plan=50.0,
                            influent_percent=30),
            ],
        ),
    ]
    return _db(make_dataset_obj(metrics))


def test_profile_block_lists_people_and_periods(simple_db):
    text = enrich.dataset_profile_block(simple_db)
    assert "Иванов Иван Иванович" in text
    assert "2026-04-06" in text and "2026-04-13" in text
    assert "показател" in text.lower()


def test_profile_omits_absent_features(simple_db):
    text = enrich.dataset_profile_block(simple_db)
    assert "звезд" not in text.lower()
    assert "групп" not in text.lower()


def test_catalog_shows_two_levels_with_deeper_counters():
    """L1 и L2 полностью; у L2 — сколько метрик на 3-м, 4-м и т.д. уровнях."""
    deep = make_metric("L4", date="2026-04-13", fact=1.0, plan=1.0)
    l3a = make_metric("L3a", date="2026-04-13", fact=1.0, plan=1.0, children=[deep])
    l3b = make_metric("L3b", date="2026-04-13", fact=1.0, plan=1.0)
    l2 = make_metric("L2", date="2026-04-13", fact=2.0, plan=2.0, children=[l3a, l3b])
    l1 = make_metric("L1", date="2026-04-13", fact=3.0, plan=3.0, children=[l2])
    db = _db(make_dataset_obj([l1]))

    text = enrich.catalog_block(db)
    assert "L1" in text and "L2" in text
    # Метрики 3-го и 4-го уровней перечислять не надо — только счётчики у L2.
    assert "L3a" not in text and "L4" not in text
    assert "2 метрики 3-го" in text or "2 метрик 3-го" in text
    assert "1 метрика 4-го" in text or "1 метрик 4-го" in text


def test_catalog_marks_direction_unit_and_plan(simple_db):
    text = enrich.catalog_block(simple_db)
    assert "обратная" in text
    assert "у.е." in text or "шт" in text


def test_scoreboard_uses_last_value_of_each_series():
    """У каждого показателя своя последняя дата — отстающий не пропадает."""
    fresh = make_metric("Свежий", date="2026-04-20", fact=100.0, plan=90.0)
    lagging = make_metric("Отстающий", date="2026-04-14", fact=5.0, plan=8.0)
    db = _db(make_dataset_obj([fresh, lagging]))
    text = enrich.scoreboard_block(db, person_key="100500")
    assert "Свежий" in text and "2026-04-20" in text
    # Показатель с более старой датой остаётся в скорборде со своей датой.
    assert "Отстающий" in text and "2026-04-14" in text


def test_scoreboard_depth_two_includes_lagging_child():
    """При запросе двух уровней дочерний показатель с отстающей датой на месте."""
    child = make_metric("Ребёнок", date="2026-04-14", fact=5.0, plan=8.0)
    parent = make_metric("Родитель", date="2026-04-20", fact=100.0, plan=90.0,
                         children=[child])
    db = _db(make_dataset_obj([parent]))
    text = enrich.scoreboard_block(db, person_key="100500", max_depth=2)
    assert "Родитель" in text and "2026-04-20" in text
    assert "Ребёнок" in text and "2026-04-14" in text


def test_scoreboard_orders_problems_first(simple_db):
    text = enrich.scoreboard_block(simple_db, person_key="100500")
    lines = [ln for ln in text.splitlines() if ln.startswith("-")]
    assert lines, text
    assert "Производительность" in lines[0]


def test_stars_block_only_with_stars(simple_db):
    assert enrich.stars_block(simple_db) == ""
    child = make_metric("Влияющий", fact=1.0, plan=2.0, is_star_metric=True)
    star = make_metric("Звезда качества", fact=None, star_received=False, children=[child])
    db = _db(make_dataset_obj([star]))
    text = enrich.stars_block(db)
    assert "Звезда качества" in text and "не получена" in text.lower()


def test_peer_block_only_with_aggregates(simple_db):
    assert enrich.peer_block(simple_db, person_key="100500") == ""


def test_gaps_block_reports_missing_plan():
    metrics = [make_metric("Без плана", date="2026-04-13", fact=5.0, plan=None)]
    db = _db(make_dataset_obj(metrics))
    text = enrich.gaps_block(db, person_key="100500")
    assert "план" in text.lower()


def test_enrichment_block_fits_budget_on_production_scale():
    db = _db(make_synthetic_dataset(n_level1=100, depth=5, periods=6, elements=5))
    text = enrich.enrichment_block(db, person_key="100500", budget_chars=9000)
    assert len(text) <= 9000, len(text)
    assert "показател" in text.lower()


def test_catalog_block_fits_budget_on_production_scale():
    db = _db(make_synthetic_dataset(n_level1=100, depth=5, periods=6, elements=5))
    text = enrich.catalog_block(db, budget_chars=14000)
    assert len(text) <= 14000, len(text)
    assert text.count("\n") > 100  # все L1 представлены


def test_enrichment_block_on_samples():
    for sample in ("sample_declining_ex_rr.json", "sample_star.json"):
        db = _db(load_sample(sample))
        person = db.conn.execute("SELECT person_key FROM person LIMIT 1").fetchone()[0]
        text = enrich.enrichment_block(db, person_key=person)
        assert text.strip(), sample
