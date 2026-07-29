"""json_analyzer_v5: peer-динамика в compute_analytics — детрендинг личной
динамики относительно группы (rel_*), бенчмарк против top20_mean_fact,
жёсткость плана по hit_rate, выбор референсного уровня БЕЗ хардкода имён.

Имена уровней инстанс-специфичны (ORG/TERR/OFFICE — лишь пример), поэтому в
фикстурах используются нестандартные имена. Данные приходят неполными — каждый
вердикт гейтится на свои входы независимо (пофакторная деградация).
"""
from langgraph_executor.aegra_agents.json_analyzer_v5.analytics import (
    apply_metric_kinds,
    compute_analytics,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.loader import (
    load_aggregates_obj,
    load_dataset_obj,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.sqlite_store import SqliteStore

PREV, CUR = "2026-06-30", "2026-07-30"


def _node(date, fact, plan=100.0, mtype="прямая", name="Продажи"):
    return {
        "id": name, "metric_name": name, "metric_type": mtype,
        "measure_type": "рубль", "date": date, "calc_period": "Месяц",
        "fact": fact, "plan": plan, "benchmark": None, "element": None,
        "child_metrics": [],
    }


def _slice(dt, mean_fact=None, hit_rate=None, top20=None, **extra):
    return {
        "dt": dt, "calc_period": "Месяц", "mean_fact": mean_fact,
        "hit_rate": hit_rate, "top20_mean_fact": top20, **extra,
    }


def _agg_metric(name, current, history=None):
    aggregates = dict(current)
    if history is not None:
        aggregates["history"] = history
    return {"metric_id": name, "metric_name": name, "aggregates": aggregates}


def _payload(*datasets):
    """datasets: (level, [метрики])."""
    return [{"dataset": {"level": lvl, "metrics": ms}} for lvl, ms in datasets]


def _store(metrics, agg_payload=None, ref_level=None):
    data = {
        "me": {"fio": "Босс", "metrics": []},
        "employees": [{"fio": "Иванов", "tabnum": 1, "metrics": metrics}],
    }
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    if agg_payload is not None:
        store.load_aggregates(load_aggregates_obj(agg_payload))
    compute_analytics(store, ref_level=ref_level)
    return store


def _cell(store, date, col, name="Продажи"):
    row = store.conn.execute(
        "SELECT a.* FROM metric_analytics a JOIN metrics m USING (metric_uid) "
        "WHERE m.date = ? AND m.metric_name = ?",
        (date, name),
    ).fetchone()
    return row[col]


def _two_period_group(prev_mean, cur_mean, level="КУСТ", **cur_extra):
    return _payload((level, [
        _agg_metric(
            "Продажи",
            _slice(CUR, mean_fact=cur_mean, **cur_extra),
            history=[_slice(PREV, mean_fact=prev_mean)],
        )
    ]))


def test_detrending_falls_with_group():
    """Личное −20% при групповом −20% → движение вместе с группой."""
    store = _store(
        [_node(PREV, 100), _node(CUR, 80)],
        _two_period_group(1000, 800),
    )
    assert _cell(store, CUR, "rel_status") == "на_уровне_группы"
    assert _cell(store, CUR, "rel_change_pct") == 0.0
    assert _cell(store, CUR, "group_change_pct") == -20.0


def test_detrending_falls_against_group():
    """Личное −20% при групповом −5% → падает против группы."""
    store = _store(
        [_node(PREV, 100), _node(CUR, 80)],
        _two_period_group(1000, 950),
    )
    assert _cell(store, CUR, "rel_status") == "хуже_группы"
    assert _cell(store, CUR, "rel_change_pct") == -15.0


def test_detrending_inverse_metric():
    """Обратная метрика: значение падает быстрее группы → это улучшение."""
    store = _store(
        [_node(PREV, 100, mtype="обратная"), _node(CUR, 80, mtype="обратная")],
        _two_period_group(1000, 950),
    )
    assert _cell(store, CUR, "rel_status") == "лучше_группы"


def test_benchmark_vs_top20_and_no_match():
    """benchmark_* считается против top20 матчащегося периода; периоды без
    матча по году-месяцу остаются None."""
    store = _store(
        [_node(PREV, 100), _node(CUR, 80)],
        _payload(("КУСТ", [_agg_metric("Продажи", _slice(CUR, top20=200))])),
    )
    assert _cell(store, CUR, "benchmark_dev_abs") == -120.0
    assert _cell(store, CUR, "benchmark_status") == "хуже_бенчмарка"
    assert _cell(store, PREV, "benchmark_status") is None  # матча на PREV нет


def test_plan_rigidity():
    hard = _store(
        [_node(CUR, 80)],
        _payload(("КУСТ", [_agg_metric("Продажи", _slice(CUR, hit_rate=26.0))])),
    )
    assert _cell(hard, CUR, "plan_rigidity") == "жёсткий_план"
    assert _cell(hard, CUR, "group_hit_rate") == 26.0

    soft = _store(
        [_node(CUR, 80)],
        _payload(("КУСТ", [_agg_metric("Продажи", _slice(CUR, hit_rate=80.0))])),
    )
    assert _cell(soft, CUR, "plan_rigidity") == "мягкий_план"

    no_plan = _store(
        [_node(CUR, 80, plan=None)],
        _payload(("КУСТ", [_agg_metric("Продажи", _slice(CUR, hit_rate=26.0))])),
    )
    assert _cell(no_plan, CUR, "plan_rigidity") is None
    assert _cell(no_plan, CUR, "group_hit_rate") is None


def test_ref_level_no_hardcoded_names():
    """Референсный уровень: min total_objects (узкая группа), имена любые."""
    payload = _payload(
        ("REGION", [_agg_metric(
            "Продажи", _slice(CUR, top20=500, total_objects=500, is_ignored=None))]),
        ("CLUSTER", [_agg_metric(
            "Продажи", _slice(CUR, top20=200, total_objects=40))]),
    )
    store = _store([_node(CUR, 80)], payload)
    # Выбран CLUSTER (40 объектов < 500): бенчмарк против его top20=200.
    assert _cell(store, CUR, "benchmark_dev_abs") == -120.0

    override = _store([_node(CUR, 80)], payload, ref_level="REGION")
    assert _cell(override, CUR, "benchmark_dev_abs") == -420.0


def test_ref_level_tie_first_in_payload():
    """При равных total_objects — первый уровень по появлению в payload."""
    payload = _payload(
        ("ЗЕЛЁНЫЙ", [_agg_metric("Продажи", _slice(CUR, top20=300, total_objects=50))]),
        ("СИНИЙ", [_agg_metric("Продажи", _slice(CUR, top20=999, total_objects=50))]),
    )
    store = _store([_node(CUR, 80)], payload)
    assert _cell(store, CUR, "benchmark_dev_abs") == -220.0  # против ЗЕЛЁНЫЙ


def test_metric_kinds_null_rel_fields():
    """Для метрик вида «вклад» относительные % бессмысленны — rel_* обнуляются."""
    store = _store(
        [_node(PREV, 100), _node(CUR, 80)],
        _two_period_group(1000, 950),
    )
    assert _cell(store, CUR, "rel_status") == "хуже_группы"
    apply_metric_kinds(store, {"Продажи": "вклад"})
    assert _cell(store, CUR, "rel_status") is None
    assert _cell(store, CUR, "rel_change_pct") is None
    assert _cell(store, CUR, "group_change_pct") is None


def test_partial_no_history():
    """Агрегаты без history: детрендинг None, бенчмарк и жёсткость работают."""
    store = _store(
        [_node(PREV, 100), _node(CUR, 80)],
        _payload(("КУСТ", [_agg_metric(
            "Продажи", _slice(CUR, mean_fact=800, hit_rate=26.0, top20=200))])),
    )
    assert _cell(store, CUR, "rel_status") is None
    assert _cell(store, CUR, "group_change_pct") is None
    assert _cell(store, CUR, "benchmark_status") == "хуже_бенчмарка"
    assert _cell(store, CUR, "plan_rigidity") == "жёсткий_план"


def test_partial_missing_fields_gate_independently():
    """Дыры в полях среза отключают только зависящий от них вердикт."""
    no_hit_rate = _store(
        [_node(CUR, 80)],
        _payload(("КУСТ", [_agg_metric("Продажи", _slice(CUR, top20=200))])),
    )
    assert _cell(no_hit_rate, CUR, "plan_rigidity") is None
    assert _cell(no_hit_rate, CUR, "benchmark_status") == "хуже_бенчмарка"

    no_top20 = _store(
        [_node(CUR, 80)],
        _payload(("КУСТ", [_agg_metric("Продажи", _slice(CUR, hit_rate=26.0))])),
    )
    assert _cell(no_top20, CUR, "benchmark_status") is None
    assert _cell(no_top20, CUR, "plan_rigidity") == "жёсткий_план"


def test_partial_metric_coverage():
    """Агрегаты пришли по одной метрике из двух — у второй всё None."""
    store = _store(
        [_node(CUR, 80), _node(CUR, 5, name="Кредиты")],
        _payload(("КУСТ", [_agg_metric("Продажи", _slice(CUR, hit_rate=26.0, top20=200))])),
    )
    assert _cell(store, CUR, "benchmark_status") == "хуже_бенчмарка"
    assert _cell(store, CUR, "benchmark_status", name="Кредиты") is None
    assert _cell(store, CUR, "plan_rigidity", name="Кредиты") is None


def test_no_aggregates_v3_mode():
    """Без агрегатов новые колонки None, compute_analytics не падает."""
    store = _store([_node(PREV, 100), _node(CUR, 80)])
    for col in ("group_change_pct", "rel_change_pct", "rel_status",
                "group_hit_rate", "plan_rigidity", "benchmark_status"):
        assert _cell(store, CUR, col) is None
    # Базовая аналитика жива.
    assert _cell(store, CUR, "pop_change_pct") == -20.0
