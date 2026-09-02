"""Разрезы (элементы) показателя: их бывают сотни.

Три правила, которые здесь закреплены:
1. Признак «есть собственный итог» считается НА ЧЕЛОВЕКА: у руководителя
   агрегатная строка может быть, а у сотрудника тот же показатель представлен
   только разрезами.
2. Показатель без агрегатной строки не исчезает из выдачи, но его итог НЕ
   вычисляется: складывать проценты, средние и ранги нельзя.
3. Разрезы объясняют показатель, а не конкурируют с ним: они не могут вытеснить
   агрегатные отклонения из карты и не заваливают промпт именами.
"""
from __future__ import annotations

from _fixtures import make_dataset_obj, make_metric, make_person  # noqa: F401

from langgraph_executor.aegra_agents.analyst_agent.agent.runctx import RunContext
from langgraph_executor.aegra_agents.analyst_agent.db import core, enrich
from langgraph_executor.aegra_agents.analyst_agent.deviations import builder, ledger
from langgraph_executor.aegra_agents.analyst_agent.deviations.format import deviations_block
from langgraph_executor.aegra_agents.analyst_agent.tools import build_tools

PERSON = "100500"
DATE = "2026-04-13"


def _ctx(db):
    devs = builder.build_deviations(db, focus_person_key=PERSON)
    return RunContext(db=db, person_key=PERSON, direction_key="d",
                      ledger=ledger.DeviationLedger(db, devs, turn=1))


def _call(tools, name, **args):
    return next(t for t in tools if t.name == name).invoke(args)


def _elements(name, count, *, fact=lambda i: 50.0, plan=100.0, date=DATE):
    return [
        make_metric(name, date=date, fact=fact(i), plan=plan, element=f"Разрез {i}")
        for i in range(count)
    ]


# --- 1. Признак агрегата считается на человека ----------------------------

def test_has_aggregate_is_per_person():
    """У руководителя агрегат есть, у сотрудника — только разрезы.

    Глобальный флаг дал бы сотруднику ложное «итог есть», и запрос по
    element IS NULL вернул бы пусто, скрыв его данные (урок json_analyzer_v5).
    """
    boss = make_person(
        [make_metric("Продажи", date=DATE, fact=100.0, plan=100.0)],
        tabnum=1, fio="Босс",
    )
    employee = make_person(
        _elements("Продажи", 3), tabnum=100500, fio="Сотрудник",
    )
    db = core.build_run_db({"me": boss, "employees": [employee]})
    rows = {
        (r["person_key"], r["metric"]): r
        for r in db.conn.execute(
            "SELECT person_key, metric, has_aggregate, n_elements FROM v_metric_person"
        )
    }
    assert rows[("1", "Продажи")]["has_aggregate"] == 1
    assert rows[("1", "Продажи")]["n_elements"] == 0
    assert rows[(PERSON, "Продажи")]["has_aggregate"] == 0
    assert rows[(PERSON, "Продажи")]["n_elements"] == 3


def test_v_metric_person_counts_only_own_elements():
    db = core.build_run_db(make_dataset_obj(
        _elements("Продажи", 4) + [make_metric("Другой", date=DATE, fact=1.0, plan=2.0)]
    ))
    rows = {
        r["metric"]: r
        for r in db.conn.execute(
            "SELECT metric, has_aggregate, n_elements, has_plan FROM v_metric_person "
            "WHERE person_key = ?", (PERSON,)
        )
    }
    assert rows["Продажи"]["n_elements"] == 4
    assert rows["Продажи"]["has_aggregate"] == 0
    assert rows["Другой"]["n_elements"] == 0
    assert rows["Другой"]["has_aggregate"] == 1
    assert rows["Другой"]["has_plan"] == 1


# --- 2. Описание показателя несёт количество и признак --------------------

def test_catalog_marks_metric_without_own_total():
    db = core.build_run_db(make_dataset_obj(
        _elements("Только разрезы", 5)
        + [make_metric("С итогом", date=DATE, fact=80.0, plan=100.0)]
    ))
    text = enrich.catalog_block(db, person_key=PERSON)
    assert "только разрезы: 5" in text
    assert "своего итога нет" in text
    # У обычного показателя пометки нет.
    line = next(ln for ln in text.splitlines() if "С итогом" in ln)
    assert "своего итога нет" not in line


def test_catalog_shows_element_count_when_aggregate_exists():
    db = core.build_run_db(make_dataset_obj(
        [make_metric("Продажи", date=DATE, fact=80.0, plan=100.0)] + _elements("Продажи", 7)
    ))
    line = next(
        ln for ln in enrich.catalog_block(db, person_key=PERSON).splitlines()
        if "Продажи" in ln
    )
    assert "разрезов 7" in line
    assert "своего итога нет" not in line


def test_catalog_never_prints_element_names():
    db = core.build_run_db(make_dataset_obj(_elements("Продажи", 30)))
    text = enrich.catalog_block(db, person_key=PERSON)
    assert "Разрез 0" not in text and "Разрез 17" not in text


# --- 3. Показатель без агрегата не исчезает -------------------------------

def test_scoreboard_keeps_metric_without_aggregate():
    db = core.build_run_db(make_dataset_obj(
        _elements("Только разрезы", 4, fact=lambda i: 40.0 + i)
        + [make_metric("С итогом", date=DATE, fact=80.0, plan=100.0)]
    ))
    text = enrich.scoreboard_block(db, person_key=PERSON)
    assert "С итогом" in text
    assert "Только разрезы" in text
    assert "своего итога нет" in text
    # Итог не вычисляется: суммы разрезов (160) в тексте быть не должно.
    assert "160" not in text


def test_scoreboard_does_not_duplicate_metric_with_aggregate():
    db = core.build_run_db(make_dataset_obj(
        [make_metric("Продажи", date=DATE, fact=80.0, plan=100.0)] + _elements("Продажи", 5)
    ))
    text = enrich.scoreboard_block(db, person_key=PERSON)
    assert len([ln for ln in text.splitlines() if ln.startswith("- Продажи")]) == 1


def test_zones_count_metric_without_aggregate_once():
    db = core.build_run_db(make_dataset_obj(
        _elements("Только разрезы", 6, fact=lambda i: 40.0)
    ))
    text = enrich.zones_block(db, person_key=PERSON)
    # Показатель учтён, но один раз — не шесть.
    assert "хуже плана — 1" in text


def test_metric_card_explains_missing_total():
    db = core.build_run_db(make_dataset_obj(_elements("Только разрезы", 5)))
    out = _call(build_tools(_ctx(db)), "metric_card", metric="Только разрезы")
    assert "своего итога" in out.lower()
    assert "Значений по периодам нет" not in out
    assert "Разрез 0" in out          # сами разрезы показаны


# --- 4. Разрезы не вытесняют показатели из карты --------------------------

def _many_metrics_with_elements(n_metrics=10, n_elements=60):
    metrics = []
    for m in range(n_metrics):
        metrics.append(make_metric(f"М{m}", date=DATE, fact=82.0, plan=100.0))
        metrics.extend(
            make_metric(f"М{m}", date=DATE, fact=40.0 + e * 0.5, plan=100.0,
                        element=f"Разрез {e}")
            for e in range(n_elements)
        )
    return core.build_run_db(make_dataset_obj(metrics))


def test_aggregate_deviations_survive_many_elements():
    db = _many_metrics_with_elements()
    devs = builder.build_deviations(db, focus_person_key=PERSON)
    aggregates = [d for d in devs if not d.get("element")]
    assert aggregates, "агрегатные отклонения полностью вытеснены разрезами"
    assert len({d["metric_name"] for d in aggregates}) >= 5


def test_element_deviations_capped_per_metric():
    db = _many_metrics_with_elements()
    from collections import Counter

    devs = builder.build_deviations(db, focus_person_key=PERSON)

    per_metric = Counter(d["metric_name"] for d in devs if d.get("element"))
    assert per_metric, "разрезы пропали совсем"
    assert max(per_metric.values()) <= 2


def test_prompt_block_limits_element_lines():
    db = _many_metrics_with_elements()
    devs = builder.build_deviations(db, focus_person_key=PERSON)
    text = deviations_block(devs)
    element_lines = [ln for ln in text.splitlines() if "(Разрез" in ln]
    assert len(element_lines) <= 4
    assert any("М" in ln and "(Разрез" not in ln for ln in text.splitlines())


def test_immaterial_element_not_in_map():
    """Разрез, который почти не отличается от своего агрегата, — не находка."""
    db = core.build_run_db(make_dataset_obj([
        make_metric("AHT", date=DATE, fact=350.0, plan=320.0, metric_type="обратная"),
        make_metric("AHT", date=DATE, fact=324.0, plan=320.0, metric_type="обратная",
                    element="Почти как агрегат"),
        make_metric("AHT", date=DATE, fact=600.0, plan=320.0, metric_type="обратная",
                    element="Вдвое хуже"),
    ]))
    elements = {
        d["element"] for d in builder.build_deviations(db, focus_person_key=PERSON)
        if d.get("element")
    }
    assert "Вдвое хуже" in elements
    assert "Почти как агрегат" not in elements


def test_element_gives_single_entry():
    """Один разрез — одна запись, а не segment_worst плюс declining."""
    db = core.build_run_db(make_dataset_obj([
        make_metric("Продажи", date="2026-04-06", fact=100.0, plan=100.0),
        make_metric("Продажи", date=DATE, fact=90.0, plan=100.0),
        make_metric("Продажи", date="2026-04-06", fact=90.0, plan=100.0, element="Плохой"),
        make_metric("Продажи", date=DATE, fact=30.0, plan=100.0, element="Плохой"),
    ]))
    devs = [
        d for d in builder.build_deviations(db, focus_person_key=PERSON)
        if d.get("element") == "Плохой"
    ]
    assert len(devs) == 1


def test_good_element_marked_as_segment_best():
    db = core.build_run_db(make_dataset_obj([
        make_metric("Продажи", date=DATE, fact=100.0, plan=100.0),
        make_metric("Продажи", date=DATE, fact=200.0, plan=100.0, element="Отличный"),
    ]))
    devs = [
        d for d in builder.build_deviations(db, focus_person_key=PERSON)
        if d.get("element") == "Отличный"
    ]
    assert devs and devs[0]["kind"] == "segment_best"
    assert devs[0]["polarity"] == "positive"


# --- 5. Карточка при сотнях разрезов --------------------------------------

def test_metric_card_caps_elements_and_reports_total():
    db = core.build_run_db(make_dataset_obj(
        [make_metric("Продажи", date=DATE, fact=80.0, plan=100.0)]
        + _elements("Продажи", 300, fact=lambda i: 20.0 + i * 0.2)
    ))
    out = _call(build_tools(_ctx(db)), "metric_card", metric="Продажи")
    shown = [ln for ln in out.splitlines() if ln.strip().startswith("- Разрез")]
    assert len(shown) <= 12
    assert "из 300" in out
    assert len(out) < 6000


def test_metric_card_single_element_history():
    db = core.build_run_db(make_dataset_obj([
        make_metric("Продажи", date="2026-04-06", fact=10.0, plan=20.0, element="Целевой"),
        make_metric("Продажи", date=DATE, fact=15.0, plan=20.0, element="Целевой"),
        make_metric("Продажи", date=DATE, fact=99.0, plan=20.0, element="Другой"),
    ]))
    out = _call(build_tools(_ctx(db)), "metric_card", metric="Продажи", element="Целевой")
    assert "Целевой" in out
    assert "2026-04-06" in out and DATE in out
    assert "99" not in out          # чужой разрез не попал


def test_metric_card_unknown_element_reports_candidates():
    db = core.build_run_db(make_dataset_obj(_elements("Продажи", 3)))
    out = _call(build_tools(_ctx(db)), "metric_card", metric="Продажи",
                element="Несуществующий")
    assert "не найден" in out.lower()
    assert "Разрез 0" in out


# --- 6. Правила в промпте -------------------------------------------------

def test_schema_doc_states_missing_total_rule():
    db = core.build_run_db(make_dataset_obj(_elements("Продажи", 3)))
    doc = core.schema_doc(db)
    assert "не суммируй разрезы" in doc.lower()
    assert "v_metric_person" in doc


def test_tools_guide_mentions_element_workflow():
    from langgraph_executor.aegra_agents.analyst_agent.prompts.tools_guide import (
        tools_guide_block,
    )

    text = tools_guide_block(18)
    assert "разрез" in text.lower()
    assert "LIMIT" in text or "лимит" in text.lower()


def test_prompt_block_puts_metrics_before_their_elements():
    """Сначала показатели, потом разрезы: разрез объясняет показатель."""
    db = _many_metrics_with_elements(n_metrics=3, n_elements=20)
    text = deviations_block(builder.build_deviations(db, focus_person_key=PERSON))
    body = [ln for ln in text.splitlines() if ln.startswith("- ")]
    first_element = next(i for i, ln in enumerate(body) if "(Разрез" in ln)
    assert not any("(Разрез" in ln for ln in body[:first_element])
    assert any("(Разрез" not in ln for ln in body[:first_element])


def test_prompt_block_skips_zero_change():
    """Нулевое изменение к прошлому периоду — шум, в промпт не идёт."""
    db = core.build_run_db(make_dataset_obj([
        make_metric("Продажи", date="2026-04-06", fact=50.0, plan=100.0),
        make_metric("Продажи", date=DATE, fact=50.0, plan=100.0),
    ]))
    text = deviations_block(builder.build_deviations(db, focus_person_key=PERSON))
    assert "+0.0 %" not in text
    assert "-0.0 %" not in text
