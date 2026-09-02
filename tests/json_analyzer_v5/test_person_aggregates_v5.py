"""json_analyzer_v5: привязка «предагрегат → человек».

Предагрегаты загружаются индивидуально (по aggregates_ids персоны), поэтому
записи агрегатов несут aggregate_id, а связка person_key → aggregate_id
хранится в person_aggregates. Аналитика (compute_analytics, build_peer_context)
берёт для человека только его группы сравнения; данные без пометок работают
по-старому (полная обратная совместимость).
"""
from langgraph_executor.aegra_agents.json_analyzer_v5.analytics import (
    build_peer_context,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.loader import (
    load_aggregates_obj,
    load_dataset_obj,
    load_person_aggregates_obj,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.nodes import _prepare_store


def _person_metric(fact):
    return {
        "id": "m1", "metric_name": "Продажи", "metric_type": "прямая",
        "measure_type": "рубль", "date": "2026-07-30", "calc_period": "Месяц",
        "fact": fact, "plan": 100, "benchmark": None, "element": None,
        "child_metrics": [],
    }


# Два сотрудника с ОДНОЙ метрикой, но РАЗНЫМИ предагрегатами: без привязки их
# группы сравнения перемешались бы (матч только по имени метрики).
_DATA = {
    "me": None,
    "employees": [
        {"tabnum": "111", "fio": "Иванов Иван", "aggregates_ids": ["agg_1"],
         "metrics": [_person_metric(100)]},
        {"tabnum": "222", "fio": "Петров Пётр", "aggregates_ids": ["agg_2"],
         "metrics": [_person_metric(100)]},
    ],
}


def _agg_slice(mean_fact, top20):
    return {
        "dt": "2026-07-30", "calc_period": "Месяц", "mean_fact": mean_fact,
        "mean_plan": 100, "mean_ex": 98.0, "median": mean_fact,
        "hit_rate": 50.0, "top20_mean_fact": top20, "iqr": 10.0, "cv": 42.0,
        "total_objects": 40,
    }


def _agg_record(agg_id, mean_fact, top20):
    return {
        "aggregate_id": agg_id,
        "dataset": {
            "level": "OFFICE",
            "level_name": "по офису",
            "metrics": [{
                "metric_id": "m1", "metric_name": "Продажи",
                "aggregates": _agg_slice(mean_fact, top20),
            }],
        },
    }


_AGGS = [_agg_record("agg_1", 110, 200), _agg_record("agg_2", 130, 400)]


def _store():
    return _prepare_store(
        load_dataset_obj(_DATA),
        load_aggregates_obj(_AGGS),
        None,
        load_person_aggregates_obj(_DATA),
    )


# --- loader ------------------------------------------------------------------

def test_aggregate_id_lands_in_rows_and_is_optional():
    rows = load_aggregates_obj(_AGGS)
    assert {r["aggregate_id"] for r in rows} == {"agg_1", "agg_2"}
    # Старая форма без пометки — None (регресс).
    legacy = dict(_agg_record("x", 1, 2))
    legacy.pop("aggregate_id")
    assert [r["aggregate_id"] for r in load_aggregates_obj([legacy])] == [None]


def test_person_pairs_collected_and_deduped():
    pairs = load_person_aggregates_obj(_DATA)
    assert pairs == [
        {"person_key": "111", "aggregate_id": "agg_1"},
        {"person_key": "222", "aggregate_id": "agg_2"},
    ]
    # Дубли пар и пустые id не плодятся; персоны без поля — молча пропускаются.
    noisy = {
        "me": {"tabnum": "9", "fio": "Босс", "metrics": [],
               "aggregates_ids": ["a", "a", None, " "]},
        "employees": [{"tabnum": "10", "fio": "Без поля", "metrics": []}],
    }
    assert load_person_aggregates_obj(noisy) == [
        {"person_key": "9", "aggregate_id": "a"}
    ]


# --- привязка в аналитике ----------------------------------------------------

def test_peer_context_uses_own_aggregate_per_person():
    store = _store()
    ivanov = build_peer_context(store, "Продажи", person="Иванов")
    petrov = build_peer_context(store, "Продажи", person="Петров")
    assert ivanov["dynamics"][0]["group_top20"] == 200
    assert petrov["dynamics"][0]["group_top20"] == 400
    # Уровни тоже срезаны по человеку: у каждого своя группа, не смесь.
    assert [lv["mean_fact"] for lv in ivanov["levels"]] == [110]
    assert [lv["mean_fact"] for lv in petrov["levels"]] == [130]


def test_compute_analytics_benchmark_from_own_aggregate():
    store = _store()
    rows = store.conn.execute(
        "SELECT m.person_key AS pk, a.benchmark_dev_abs AS dev "
        "FROM metrics m JOIN metric_analytics a ON a.metric_uid = m.metric_uid "
        "ORDER BY m.person_key"
    ).fetchall()
    # Бенчмарк = факт против top20 СВОЕЙ группы: 100-200 и 100-400.
    assert [(r["pk"], r["dev"]) for r in rows] == [("111", -100), ("222", -300)]


def test_unmarked_rows_visible_to_everyone():
    # Записи без aggregate_id (старая форма) видны всем, даже при наличии связки.
    legacy = dict(_agg_record("x", 999, 500))
    legacy.pop("aggregate_id")
    store = _prepare_store(
        load_dataset_obj(_DATA),
        load_aggregates_obj([legacy]),
        None,
        load_person_aggregates_obj(_DATA),
    )
    out = build_peer_context(store, "Продажи", person="Иванов")
    assert out["dynamics"][0]["group_top20"] == 500


def test_no_binding_falls_back_to_all_rows():
    # Датасет без aggregates_ids + помеченные записи → фильтра нет, поведение
    # прежнее (широкая совместимость со старым форматом данных).
    data = {
        "me": None,
        "employees": [{"tabnum": "111", "fio": "Иванов Иван",
                       "metrics": [_person_metric(100)]}],
    }
    store = _prepare_store(
        load_dataset_obj(data),
        load_aggregates_obj([_agg_record("agg_1", 110, 200)]),
    )
    out = build_peer_context(store, "Продажи", person="Иванов")
    assert out["dynamics"][0]["group_top20"] == 200
