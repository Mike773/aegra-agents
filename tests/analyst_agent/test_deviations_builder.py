"""analyst_agent.deviations: карта отклонений на диалог.

Правила: что считается отклонением, что существенно, как расставляется
приоритет (влияние × масштаб × управляемость) и как записи переживают ходы.
"""
from __future__ import annotations

from _fixtures import make_dataset_obj, make_metric, make_person  # noqa: F401

from langgraph_executor.aegra_agents.analyst_agent.db import analytics, core
from langgraph_executor.aegra_agents.analyst_agent.deviations import builder

PERSON = "100500"


def _db(dataset, aggregates=None):
    d = core.build_run_db(dataset, aggregates)
    return d


def _build(db, **kw):
    return builder.build_deviations(db, focus_person_key=PERSON, **kw)


def _kinds(devs):
    return {d["kind"] for d in devs}


def test_below_plan_detected_with_numbers():
    db = _db(make_dataset_obj([make_metric("Продажи", fact=80.0, plan=100.0)]))
    devs = _build(db)
    dev = next(d for d in devs if d["kind"] == "below_plan")
    assert dev["metric_name"] == "Продажи"
    assert dev["polarity"] == "negative"
    assert dev["fact"] == 80.0 and dev["plan"] == 100.0
    assert dev["plan_dev_pct"] < 0
    assert dev["source"] == "auto" and dev["status"] == "open"
    assert dev["date"] == "2026-04-06"


def test_above_plan_is_positive():
    db = _db(make_dataset_obj([make_metric("Продажи", fact=120.0, plan=100.0)]))
    dev = next(d for d in _build(db) if d["kind"] == "above_plan")
    assert dev["polarity"] == "positive"


def test_inverse_metric_direction_respected():
    """У обратной метрики рост факта над планом — это хуже плана."""
    db = _db(make_dataset_obj([
        make_metric("AHT", fact=320.0, plan=300.0, metric_type="обратная")
    ]))
    assert "below_plan" in _kinds(_build(db))
    assert "above_plan" not in _kinds(_build(db))


def test_declining_detected_from_dynamics():
    metrics = [
        make_metric("Продажи", date="2026-04-06", fact=100.0, plan=100.0),
        make_metric("Продажи", date="2026-04-13", fact=70.0, plan=100.0),
    ]
    devs = _build(_db(make_dataset_obj(metrics)))
    dev = next(d for d in devs if d["kind"] == "declining")
    assert dev["pop_change_pct"] < 0


def test_metric_without_plan_gives_no_plan_or_dynamics_deviation():
    metrics = [
        make_metric("Без плана", date="2026-04-06", fact=100.0, plan=None),
        make_metric("Без плана", date="2026-04-13", fact=50.0, plan=None),
    ]
    kinds = _kinds(_build(_db(make_dataset_obj(metrics))))
    assert "below_plan" not in kinds
    assert "declining" not in kinds


def test_star_missed_and_at_risk():
    child = make_metric("Влияющий", fact=5.0, plan=10.0, is_star_metric=True)
    star = make_metric("Звезда качества", fact=None, star_received=False, children=[child])
    kinds = _kinds(_build(_db(make_dataset_obj([star]))))
    assert "star_missed" in kinds
    assert "star_at_risk" in kinds


def test_star_received_is_positive():
    star = make_metric("Звезда качества", fact=None, star_received=True)
    dev = next(d for d in _build(_db(make_dataset_obj([star]))) if d["kind"] == "star_received")
    assert dev["polarity"] == "positive"


def test_element_included_only_when_material():
    """Разрез попадает в карту, только если агрегат сам флагован либо разрез
    отклоняется и вдвое сильнее агрегата, и заметно сам по себе."""
    metrics = [
        make_metric("Продажи", fact=99.7, plan=100.0),                       # агрегат в плане
        make_metric("Продажи", fact=95.0, plan=100.0, element="Мелкий"),     # -5 %
        make_metric("Продажи", fact=50.0, plan=100.0, element="Крупный"),    # -50 %
    ]
    devs = _build(_db(make_dataset_obj(metrics)))
    elements = {d["element"] for d in devs if d["element"]}
    assert "Крупный" in elements
    assert "Мелкий" not in elements


def test_elements_included_when_aggregate_itself_flagged():
    """Если проблемен сам агрегат — разрезы нужны для декомпозиции, даже мелкие."""
    metrics = [
        make_metric("Продажи", fact=70.0, plan=100.0),
        make_metric("Продажи", fact=95.0, plan=100.0, element="Мелкий"),
    ]
    devs = _build(_db(make_dataset_obj(metrics)))
    assert "Мелкий" in {d["element"] for d in devs if d["element"]}


def test_priority_prefers_bigger_impact_and_scale():
    child = make_metric("Мелкий драйвер", fact=90.0, plan=100.0, influent_percent=10)
    root_bad = make_metric("Главный", fact=50.0, plan=100.0, children=[child])
    root_ok = make_metric("Второй", fact=98.0, plan=100.0)
    devs = _build(_db(make_dataset_obj([root_bad, root_ok])))
    negatives = [d for d in devs if d["polarity"] == "negative"]
    assert negatives[0]["metric_name"] == "Главный"
    assert negatives[0]["priority"] > negatives[-1]["priority"]


def test_child_priority_scaled_by_influence_share():
    weak = make_metric("Слабый драйвер", fact=50.0, plan=100.0, influent_percent=20)
    strong = make_metric("Сильный драйвер", fact=50.0, plan=100.0, influent_percent=80)
    root = make_metric("Корень", fact=50.0, plan=100.0, children=[weak, strong])
    devs = _build(_db(make_dataset_obj([root])))
    by_name = {d["metric_name"]: d for d in devs if d["kind"] == "below_plan"}
    assert by_name["Сильный драйвер"]["priority"] > by_name["Слабый драйвер"]["priority"]
    assert by_name["Корень"]["priority"] > by_name["Сильный драйвер"]["priority"]


def test_ids_are_stable_between_turns():
    db = _db(make_dataset_obj([make_metric("Продажи", fact=80.0, plan=100.0)]))
    first = _build(db, turn=1)
    second = _build(db, previous=first, turn=2)
    assert [d["id"] for d in first] == [d["id"] for d in second]
    assert second[0]["turn_created"] == 1
    assert second[0]["turn_updated"] == 2


def test_agent_notes_survive_rebuild():
    db = _db(make_dataset_obj([make_metric("Продажи", fact=80.0, plan=100.0)]))
    previous = _build(db, turn=1) + [
        {
            "id": "manual1", "kind": "hypothesis", "polarity": "negative",
            "metric_name": "Продажи", "person_key": PERSON, "priority": 0.4,
            "source": "agent", "status": "open", "note": "проверить обучение",
            "turn_created": 1, "turn_updated": 1,
        }
    ]
    devs = _build(db, previous=previous, turn=2)
    manual = next(d for d in devs if d["id"] == "manual1")
    assert manual["note"] == "проверить обучение"
    assert manual["source"] == "agent"


def test_disappeared_auto_deviation_marked_resolved():
    bad = _db(make_dataset_obj([make_metric("Продажи", fact=80.0, plan=100.0)]))
    good = _db(make_dataset_obj([make_metric("Продажи", fact=105.0, plan=100.0)]))
    first = builder.build_deviations(bad, focus_person_key=PERSON, turn=1)
    second = builder.build_deviations(good, focus_person_key=PERSON, previous=first, turn=2)
    stale = [d for d in second if d["kind"] == "below_plan"]
    assert stale and stale[0]["status"] == "resolved"


def test_max_auto_caps_by_priority():
    metrics = [
        make_metric(f"М{i}", fact=100.0 - i, plan=100.0) for i in range(30)
    ]
    devs = builder.build_deviations(
        _db(make_dataset_obj(metrics)), focus_person_key=PERSON, max_auto=5
    )
    auto = [d for d in devs if d["source"] == "auto"]
    assert len(auto) == 5
    assert auto == sorted(auto, key=lambda d: -d["priority"])


def test_materialize_and_query_from_sql():
    db = _db(make_dataset_obj([make_metric("Продажи", fact=80.0, plan=100.0)]))
    devs = _build(db)
    builder.materialize(db, devs)
    rows = db.conn.execute("SELECT metric_name, kind, priority FROM v_deviation").fetchall()
    assert rows and rows[0]["metric_name"] == "Продажи"


def test_focus_person_only():
    other = make_person([make_metric("Продажи", fact=10.0, plan=100.0)],
                        tabnum=999, fio="Другой")
    dataset = make_dataset_obj(
        [make_metric("Продажи", fact=80.0, plan=100.0)], employees_extra=[other]
    )
    devs = _build(_db(dataset))
    assert {d["person_key"] for d in devs} == {PERSON}
