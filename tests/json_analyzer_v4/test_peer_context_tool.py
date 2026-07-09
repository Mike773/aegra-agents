"""json_analyzer_v4: инструмент peer_context — единый составной рендер peer-данных
(динамика на фоне группы, уровни, рейтинги). Секции независимы: пустые
пропускаются молча, состав данных плавает между инстансами и поставками.
"""
from langgraph_executor.aegra_agents.json_analyzer_v4.analytics import compute_analytics
from langgraph_executor.aegra_agents.json_analyzer_v4.loader import (
    load_aggregates_obj,
    load_dataset_obj,
)
from langgraph_executor.aegra_agents.json_analyzer_v4.sqlite_store import SqliteStore
from langgraph_executor.aegra_agents.json_analyzer_v4.store_cache import EmbeddingIndex
from langgraph_executor.aegra_agents.json_analyzer_v4.tools import build_tools

PREV, CUR = "2026-06-30", "2026-07-30"


def _node(date, fact, plan=100.0, name="Продажи", rankings=None):
    node = {
        "id": name, "metric_name": name, "metric_type": "прямая",
        "measure_type": "рубль", "date": date, "calc_period": "Месяц",
        "fact": fact, "plan": plan, "benchmark": None, "element": None,
        "child_metrics": [],
    }
    if rankings is not None:
        node["rankings"] = rankings
    return node


def _slice(dt, **fields):
    return {"dt": dt, "calc_period": "Месяц", **fields}


def _payload(*datasets):
    return [
        {"dataset": {"level": lvl, "metrics": [
            {"metric_id": "1", "metric_name": "Продажи", "aggregates": agg}
        ]}}
        for lvl, agg in datasets
    ]


def _store(metrics, agg_payload=None):
    data = {
        "me": {"fio": "Босс", "metrics": []},
        "employees": [{"fio": "Иванов", "tabnum": 1, "metrics": metrics}],
    }
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    if agg_payload is not None:
        store.load_aggregates(load_aggregates_obj(agg_payload))
    compute_analytics(store)
    return store


def _tool(store, name="peer_context"):
    tools = build_tools(store, EmbeddingIndex([]), lambda q: [0.0])
    return next(t for t in tools if t.name == name)


FULL_AGG = _payload(
    ("УЗЕЛ", {**_slice(CUR, mean_fact=800, median=700, top20_mean_fact=2000,
                       hit_rate=26.0, total_objects=40),
              "history": [_slice(PREV, mean_fact=1000, top20_mean_fact=1800)]}),
    ("БАНК", {**_slice(CUR, mean_fact=900, median=750, top20_mean_fact=2500,
                       hit_rate=44.0, total_objects=500),
              "history": [_slice(PREV, mean_fact=1000)]}),
)

RANKS_TWO_DATES = [
    _node(PREV, 100, rankings=[{"rank": "20 из 500", "level": "БАНК", "percentile": 96.0}]),
    _node(CUR, 80, rankings=[
        {"rank": "458 из 500", "level": "БАНК", "percentile": 8.4},
        {"rank": "2 из 40", "level": "УЗЕЛ", "percentile": 98.0},
    ]),
]


def test_registered_and_full_render():
    store = _store(RANKS_TWO_DATES, FULL_AGG)
    out = _tool(store).invoke({"metric": "Продажи"})
    assert "Динамика на фоне группы" in out
    assert "хуже_группы" in out or "на_уровне_группы" in out
    assert "Разрыв с топ-20%" in out
    assert "жёсткий_план" in out
    assert "Уровни peer-групп" in out
    assert "УЗЕЛ" in out and "БАНК" in out  # имена уровней — как в данных
    assert "458 из 500" in out
    assert "Позиция в рейтинге (БАНК)" in out and "96" in out and "8.4" in out


def test_localization_needs_two_levels():
    """Один уровень (произвольное имя) → секция уровней есть, локализации нет."""
    one_level = _payload(
        ("КУСТ-7", {**_slice(CUR, mean_fact=800, total_objects=40),
                    "history": [_slice(PREV, mean_fact=1000)]}),
    )
    out = _tool(_store([_node(CUR, 80)], one_level)).invoke({"metric": "Продажи"})
    assert "КУСТ-7" in out
    assert "Локализация" not in out


def test_localization_systemic():
    """Оба уровня ухудшаются → системное."""
    store = _store([_node(PREV, 100), _node(CUR, 80)], FULL_AGG)
    out = _tool(store).invoke({"metric": "Продажи"})
    assert "Локализация: системное" in out


def test_no_history_dynamics_without_rel():
    """Без history секция динамики есть (бенчмарк/жёсткость), vs_группы нет."""
    no_history = _payload(
        ("УЗЕЛ", _slice(CUR, mean_fact=800, top20_mean_fact=2000, hit_rate=26.0,
                        total_objects=40)),
    )
    out = _tool(_store([_node(PREV, 100), _node(CUR, 80)], no_history)).invoke(
        {"metric": "Продажи"}
    )
    assert "gap_до_топ20" in out
    assert "жёсткий_план" in out
    # Колонка vs_группы пуста во всех строках — из таблицы скрыта (в шапке
    # секции статичное пояснение остаётся).
    assert "| vs_группы" not in out


def test_position_dynamics_needs_two_points():
    single_point = [_node(CUR, 80, rankings=[
        {"rank": "458 из 500", "level": "БАНК", "percentile": 8.4},
    ])]
    out = _tool(_store(single_point)).invoke({"metric": "Продажи"})
    assert "458 из 500" in out
    assert "Позиция в рейтинге" not in out

    out2 = _tool(_store(RANKS_TWO_DATES)).invoke({"metric": "Продажи"})
    assert "Позиция в рейтинге (БАНК)" in out2


def test_no_peer_data_at_all():
    out = _tool(_store([_node(CUR, 80)])).invoke({"metric": "Продажи"})
    assert "peer-данные не загружены" in out


def test_unknown_metric_validated():
    out = _tool(_store([_node(CUR, 80)])).invoke({"metric": "Несуществующая"})
    assert "не найдена" in out


def test_rank_single_employee_redirects_to_peer_context():
    """Один сотрудник в датасете: rank не ранжирует (не из чего), а
    детерминированно отправляет к peer_context — вывод «худший/лучший в
    группе» из одной строки становится невозможным."""
    store = _store([_node(CUR, 80)])
    out = _tool(store, "rank").invoke({"metric": "Продажи", "date": CUR})
    assert "один сотрудник" in out
    assert "peer_context" in out
    assert "|" not in out  # таблицы ранжирования нет


def test_rank_team_still_works():
    """Регрессия: на команде из 2+ сотрудников rank работает как раньше."""
    data = {
        "me": {"fio": "Босс", "metrics": []},
        "employees": [
            {"fio": "Иванов", "tabnum": 1, "metrics": [_node(CUR, 80)]},
            {"fio": "Сидоров", "tabnum": 2, "metrics": [_node(CUR, 120)]},
        ],
    }
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    compute_analytics(store)
    out = _tool(store, "rank").invoke({"metric": "Продажи", "date": CUR})
    assert "Иванов" in out and "Сидоров" in out
    assert "peer_context" not in out


def test_docstrings_split_team_vs_peer_group():
    """Докстринги (описания для модели) явно разводят два вида сравнения."""
    store = _store([_node(CUR, 80)])
    tools = {t.name: t for t in build_tools(store, EmbeddingIndex([]), lambda q: [0.0])}
    assert "ВНУТРИ КОМАНДЫ" in tools["rank"].description
    assert "peer_context" in tools["rank"].description
    assert "БОЛЬШОЙ peer-группой" in tools["peer_context"].description
    assert "rank" in tools["peer_context"].description


def test_flat_columns_show_and_hide_peer_verdicts():
    """get_metric: vs_группы/жёсткость_плана видны при данных и скрыты без них."""
    with_peer = _store(RANKS_TWO_DATES, FULL_AGG)
    out = _tool(with_peer, "get_metric").invoke({"metric": "Продажи"})
    assert "жёсткость_плана" in out
    assert "vs_группы" in out

    without_peer = _store(RANKS_TWO_DATES)
    out2 = _tool(without_peer, "get_metric").invoke({"metric": "Продажи"})
    assert "жёсткость_плана" not in out2
    assert "vs_группы" not in out2
