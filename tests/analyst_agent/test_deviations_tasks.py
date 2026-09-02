"""Задачи руководителя, журнал отклонений и выбор главного вывода.

Если в брифинге есть конкретные задачи, карта отклонений строится от них:
названные показатели поднимаются в приоритете, остальные приглушаются, а
невыполненная числовая цель задачи становится отдельным поводом.
"""
from __future__ import annotations

from _fixtures import make_dataset_obj, make_metric  # noqa: F401

from langgraph_executor.aegra_agents.analyst_agent.db import analytics, core
from langgraph_executor.aegra_agents.analyst_agent.deviations import (
    builder,
    insight,
    ledger,
    tasks as tasks_mod,
)

PERSON = "100500"


def _db(dataset):
    d = core.build_run_db(dataset)
    return d


# --- разбор задач из брифинга --------------------------------------------

def test_parse_numbered_tasks():
    briefing = (
        "Разбери показатели сотрудника.\n"
        "1. Посмотри продажи и почему они просели\n"
        "2. Проверь качество обслуживания\n"
    )
    parsed = tasks_mod.parse_tasks(briefing)
    assert [t["title"] for t in parsed] == [
        "Посмотри продажи и почему они просели",
        "Проверь качество обслуживания",
    ]
    assert parsed[0]["id"] != parsed[1]["id"]


def test_parse_bulleted_tasks():
    parsed = tasks_mod.parse_tasks("- Продажи\n- AHT\n")
    assert len(parsed) == 2


def test_plain_briefing_has_no_tasks():
    assert tasks_mod.parse_tasks("Что происходит с показателями сотрудника?") == []
    assert tasks_mod.parse_tasks("") == []


def test_parse_numeric_target():
    parsed = tasks_mod.parse_tasks("1. Довести продажи до 120")
    assert parsed[0]["target_value"] == 120.0


def test_resolve_task_metrics_by_name_and_fuzzy():
    db = _db(make_dataset_obj([
        make_metric("Продажи", fact=80.0, plan=100.0),
        make_metric("Качество обслуживания", fact=90.0, plan=95.0),
    ]))
    parsed = tasks_mod.parse_tasks("1. Продажи\n2. качество обслуживания\n")
    resolved = tasks_mod.resolve_task_metrics(db, parsed)
    assert resolved[0]["metric_names"] == ["Продажи"]
    assert resolved[1]["metric_names"] == ["Качество обслуживания"]


def test_resolve_task_metrics_ignores_unknown():
    db = _db(make_dataset_obj([make_metric("Продажи", fact=80.0, plan=100.0)]))
    resolved = tasks_mod.resolve_task_metrics(
        db, tasks_mod.parse_tasks("1. Посмотри погоду на Марсе")
    )
    assert resolved[0]["metric_names"] == []


# --- фокус карты на задачах ----------------------------------------------

def _two_metric_db():
    return _db(make_dataset_obj([
        make_metric("Продажи", fact=80.0, plan=100.0),
        make_metric("Качество", fact=80.0, plan=100.0),
    ]))


def test_task_metrics_boosted_others_dampened():
    db = _two_metric_db()
    plain = {d["metric_name"]: d for d in builder.build_deviations(
        db, focus_person_key=PERSON) if d["kind"] == "below_plan"}
    tasks = tasks_mod.resolve_task_metrics(db, tasks_mod.parse_tasks("1. Продажи"))
    focused = {d["metric_name"]: d for d in builder.build_deviations(
        db, focus_person_key=PERSON, tasks=tasks) if d["kind"] in ("below_plan", "task_related")}

    assert focused["Продажи"]["priority"] > plain["Продажи"]["priority"]
    assert focused["Качество"]["priority"] < plain["Качество"]["priority"]
    assert focused["Продажи"]["task_id"] == tasks[0]["id"]
    # Приглушённые не выбрасываются — они всё ещё в карте.
    assert "Качество" in focused


def test_unmet_task_target_becomes_task_related():
    db = _db(make_dataset_obj([make_metric("Продажи", fact=105.0, plan=100.0)]))
    tasks = tasks_mod.resolve_task_metrics(
        db, tasks_mod.parse_tasks("1. Довести продажи до 120")
    )
    devs = builder.build_deviations(db, focus_person_key=PERSON, tasks=tasks)
    dev = next(d for d in devs if d["task_id"])
    assert dev["kind"] == "task_related"
    assert dev["polarity"] == "negative"
    assert dev["task_target"] == 120.0
    assert dev["task_dev_pct"] < 0


# --- журнал: чтение и дозапись моделью ------------------------------------

def test_ledger_list_filters_and_renders():
    db = _two_metric_db()
    devs = builder.build_deviations(db, focus_person_key=PERSON)
    log = ledger.DeviationLedger(db, devs, turn=1)
    text = log.render(scope="all")
    assert "Продажи" in text and "Качество" in text
    only_plan = log.render(scope="plan")
    assert "хуже плана" in only_plan
    assert log.render(scope="achievements") in ("", "Достижений в карте нет.")


def test_ledger_note_adds_agent_entry_visible_in_sql():
    db = _two_metric_db()
    log = ledger.DeviationLedger(db, builder.build_deviations(db, focus_person_key=PERSON), turn=2)
    out = log.note(metric="Продажи", kind="hypothesis", text="просело из-за отпусков")
    assert "Продажи" in out
    entries = [d for d in log.to_state() if d["source"] == "agent"]
    assert entries and entries[0]["note"] == "просело из-за отпусков"
    assert entries[0]["turn_created"] == 2
    row = db.conn.execute(
        "SELECT note, source FROM deviation WHERE source = 'agent'"
    ).fetchone()
    assert row["note"] == "просело из-за отпусков"


def test_ledger_note_unknown_metric_reports_candidates():
    db = _two_metric_db()
    log = ledger.DeviationLedger(db, [], turn=1)
    out = log.note(metric="Неведомый показатель", kind="hypothesis", text="что-то")
    assert "не наш" in out.lower() or "не найд" in out.lower()
    assert log.to_state() == []


# --- главный вывод для сервиса инсайтов -----------------------------------

def test_main_insight_is_top_negative():
    db = _db(make_dataset_obj([
        make_metric("Главный", fact=50.0, plan=100.0),
        make_metric("Мелкий", fact=98.0, plan=100.0),
    ]))
    devs = builder.build_deviations(db, focus_person_key=PERSON)
    ins = insight.pick_main_insight(devs)
    assert ins["type"] == "main_problem"
    assert ins["metric_name"] == "Главный"
    assert "50" in ins["text"] and "100" in ins["text"]


def test_main_insight_norm_when_everything_fine():
    db = _db(make_dataset_obj([make_metric("Продажи", fact=100.0, plan=100.0)]))
    devs = builder.build_deviations(db, focus_person_key=PERSON)
    ins = insight.pick_main_insight(devs, db=db)
    assert ins["type"] == "norm"
    assert ins["metric_name"]


def test_main_insight_achievement_when_only_positive():
    db = _db(make_dataset_obj([make_metric("Продажи", fact=150.0, plan=100.0)]))
    devs = builder.build_deviations(db, focus_person_key=PERSON)
    ins = insight.pick_main_insight(devs, db=db)
    assert ins["type"] == "achievement"


def test_main_insight_carries_metric_id():
    db = _db(make_dataset_obj([
        make_metric("Продажи", id="M-42", fact=50.0, plan=100.0)
    ]))
    devs = builder.build_deviations(db, focus_person_key=PERSON)
    assert insight.pick_main_insight(devs)["metric_id"] == "M-42"


def test_main_insight_none_without_data():
    assert insight.pick_main_insight([]) is None
