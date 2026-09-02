"""Регрессии, найденные живым прогоном на GigaChat.

1. Модель присылает аргументы инструмента пустыми НЕ только строкой: приходил
   пустой объект `{}`. Он не нормализовался в None, и вызов уходил в резолв как
   есть.
2. Вид показателя (уровень/вклад/индекс) выставлялся только когда доступен кэш
   трактовок из wiki. Без него метрика «РАНГ …» считалась обычным уровнем и
   получала вердикты «лучше плана»/«улучшается» с процентами, чего у ранга быть
   не может.
"""
from __future__ import annotations

from _fixtures import make_dataset_obj, make_metric  # noqa: F401

from langgraph_executor.aegra_agents.analyst_agent.agent.guards import clean_args
from langgraph_executor.aegra_agents.analyst_agent.agent.runctx import RunContext
from langgraph_executor.aegra_agents.analyst_agent.db import analytics, core
from langgraph_executor.aegra_agents.analyst_agent.deviations import builder, ledger
from langgraph_executor.aegra_agents.analyst_agent.tools import build_tools

PERSON = "100500"


def _db(dataset):
    d = core.build_run_db(dataset)
    return d


def _ctx(db):
    devs = builder.build_deviations(db, focus_person_key=PERSON)
    return RunContext(db=db, person_key=PERSON, direction_key="d",
                      ledger=ledger.DeviationLedger(db, devs, turn=1))


def _call(tools, name, **args):
    return next(t for t in tools if t.name == name).invoke(args)


# --- 1. пустые аргументы любого типа --------------------------------------

def test_clean_args_normalizes_empty_containers():
    cleaned = clean_args({"metric": {}, "person": [], "date": "  ", "depth": 0})
    assert cleaned["metric"] is None
    assert cleaned["person"] is None
    assert cleaned["date"] is None
    # Ноль — валидное значение, его не трогаем.
    assert cleaned["depth"] == 0


def test_metric_card_handles_empty_object_argument():
    db = _db(make_dataset_obj([make_metric("Продажи", fact=80.0, plan=100.0)]))
    out = _call(build_tools(_ctx(db)), "metric_card", metric={})
    # Пустой аргумент — это «не задано», а не «показателя нет в данных».
    assert "не указано название" in out.lower()
    # В подсказке нет мусора вроде «{}» — только живые кандидаты из каталога.
    assert "{}" not in out
    assert "Продажи" in out


# --- 2. вид показателя без базы знаний ------------------------------------

def test_kind_guessed_from_name_without_knowledge_cache():
    db = _db(make_dataset_obj([
        make_metric("РАНГ Производительность", fact=2.0, plan=5.0),
        make_metric("Вклад в отток", fact=-3.0, plan=1.0),
        make_metric("Производительность", fact=18.0, plan=20.0),
    ]))
    kinds = {
        r["name"]: r["kind"]
        for r in db.conn.execute("SELECT name, kind FROM metric")
    }
    assert kinds["РАНГ Производительность"] == "индекс"
    assert kinds["Вклад в отток"] == "вклад"
    assert kinds["Производительность"] == "уровень"


def test_index_metric_does_not_get_plan_verdicts_in_deviations():
    db = _db(make_dataset_obj([
        make_metric("РАНГ Производительность", fact=2.0, plan=5.0),
    ]))
    kinds = {d["kind"] for d in builder.build_deviations(db, focus_person_key=PERSON)}
    # У ранга не бывает «лучше плана» и «улучшается»: это позиция, а не уровень.
    assert "above_plan" not in kinds
    assert "improving" not in kinds
    assert "below_plan" not in kinds


def test_index_metric_percentages_suppressed():
    db = _db(make_dataset_obj([
        make_metric("РАНГ AHT", date="2026-04-06", fact=5.0, plan=3.0),
        make_metric("РАНГ AHT", date="2026-04-13", fact=2.0, plan=3.0),
    ]))
    row = db.conn.execute(
        "SELECT plan_dev_pct, pop_change_pct FROM v_fact_latest WHERE metric = 'РАНГ AHT'"
    ).fetchone()
    assert row["plan_dev_pct"] is None
    assert row["pop_change_pct"] is None


def test_level_metric_keeps_percentages():
    db = _db(make_dataset_obj([
        make_metric("Продажи", date="2026-04-06", fact=80.0, plan=100.0),
        make_metric("Продажи", date="2026-04-13", fact=60.0, plan=100.0),
    ]))
    row = db.conn.execute(
        "SELECT plan_dev_pct, pop_change_pct FROM v_fact_latest WHERE metric = 'Продажи'"
    ).fetchone()
    assert row["plan_dev_pct"] is not None
    assert row["pop_change_pct"] is not None


def test_knowledge_overrides_guessed_kind():
    from langgraph_executor.aegra_agents.analyst_agent.db import loader
    from langgraph_executor.aegra_agents.metric_knowledge import repository
    from langgraph_executor.aegra_agents.metric_knowledge.types import KnowledgeRow

    db = _db(make_dataset_obj([
        make_metric("Индекс лояльности", fact=70.0, plan=65.0, description="Опросная метрика")
    ]))
    assert db.conn.execute("SELECT kind FROM metric").fetchone()["kind"] == "индекс"

    key = loader.metric_key("Индекс лояльности", "Опросная метрика")
    repository.apply_knowledge(db, {
        key: KnowledgeRow(metric_key=key, name_key=loader.name_key("Индекс лояльности"),
                          metric_name="Индекс лояльности", summary="Уровень лояльности",
                          rules={"kind": "уровень"}, status="found", confidence=0.9)
    })
    # Трактовка из базы знаний точнее эвристики по названию.
    assert db.conn.execute("SELECT kind FROM metric").fetchone()["kind"] == "уровень"


# --- 3. фокус-персона в ветке составного разбора ---------------------------

def test_plan_node_keeps_deviations_when_tabnum_not_in_dataset():
    """employee_tabnum из configurable может не совпадать с ключом в данных.

    Подготовка хода это учитывает (фолбэк на первого сотрудника), а ветка
    составного разбора пересобирала карту по несуществующему ключу и обнуляла
    её — на живом прогоне инструмент отвечал «карта отклонений пуста».
    """
    from langgraph_executor.aegra_agents.analyst_agent.nodes.prepare import (
        focus_person_key,
    )

    db = _db(make_dataset_obj(
        [make_metric("Продажи", fact=80.0, plan=100.0)], tabnum=4032085
    ))
    # Табельного «2000» в датасете нет — берём первого сотрудника.
    assert focus_person_key(db, "2000") == "4032085"
    assert builder.build_deviations(db, focus_person_key=focus_person_key(db, "2000"))


# --- 4. повторный вызов с «пустыми» аргументами ----------------------------

def test_repeat_key_ignores_absent_and_empty_args():
    """«Аргумента нет» и «аргумент пустой» — один и тот же вызов.

    Иначе модель, зовущая инструмент без обязательного параметра, крутит его по
    кругу: ключи `{}` и `{"metric": None}` считались разными вызовами.
    """
    from langgraph_executor.aegra_agents.analyst_agent.agent.guards import RunGuards

    guards = RunGuards(budget=5)
    guards.register("metric_card", clean_args({"metric": {}}))
    assert guards.is_repeat("metric_card", clean_args({}))
    assert guards.is_repeat("metric_card", clean_args({"metric": "", "person": None}))
    # Осмысленный вызов повтором не считается.
    assert not guards.is_repeat("metric_card", clean_args({"metric": "Продажи"}))
