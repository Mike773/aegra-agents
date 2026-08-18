"""json_analyzer_v5.diagnosis: детерминированная диагностика против когорты.

Методология: производные метрики (robust_z, факт/медиана), реалистичность плана,
значимость отклонения по совпадению признаков, фильтр «выброс vs устойчивость»,
сценарии динамики относительно когорты, прогноз, кластер, скоринг. Здесь
проверяются ключевые сценарии и честная деградация при неполных данных.
"""
import json

from langgraph_executor.aegra_agents.json_analyzer_v5.diagnosis import (
    _ACHIEVEMENT_WEIGHTS,
    _PROBLEM_WEIGHTS,
    build_diagnosis_store,
    dataset_fingerprint,
    diagnose_employee,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.loader import (
    load_aggregates_obj,
    load_dataset_obj,
)

_DATES = [
    "2026-02-28", "2026-03-31", "2026-04-30",
    "2026-05-31", "2026-06-30", "2026-07-31",
]


def _node(date, fact, plan=100, metric_type="прямая", rankings=None, children=None):
    node = {
        "id": "m1", "metric_name": "Продажи", "metric_type": metric_type,
        "measure_type": "рубль", "date": date, "calc_period": "Месяц",
        "fact": fact, "plan": plan, "benchmark": None, "element": None,
        "child_metrics": children or [],
    }
    if rankings is not None:
        node["rankings"] = rankings
    return node


def _dataset(facts, plan=100, metric_type="прямая", rankings_by_date=None):
    metrics = [
        _node(
            d, f, plan=plan, metric_type=metric_type,
            rankings=(rankings_by_date or {}).get(d),
        )
        for d, f in zip(_DATES, facts)
    ]
    return {
        "me": None,
        "employees": [{"tabnum": "111", "fio": "Иванов Иван", "metrics": metrics}],
    }


def _agg_slice(dt, *, median, mean_fact=None, mean_plan=100, mean_ex=95.0,
               hit_rate=70.0, top20=130.0, iqr=10.0, cv=20.0, total=120):
    return {
        "dt": dt, "calc_period": "Месяц",
        "mean_fact": mean_fact if mean_fact is not None else median,
        "mean_plan": mean_plan, "mean_ex": mean_ex, "median": median,
        "hit_rate": hit_rate, "top20_mean_fact": top20, "iqr": iqr, "cv": cv,
        "total_objects": total,
    }


def _aggregates(slices):
    """Одна запись агрегатов: текущий срез = последний, остальные — history."""
    current = slices[-1]
    return [{
        "dataset": {
            "level": "OFFICE", "level_name": "по офису",
            "metrics": [{
                "metric_id": "m1", "metric_name": "Продажи",
                "aggregates": {**current, "history": slices[:-1]},
            }],
        },
    }]


def _flat_aggs(median=100.0, **kw):
    return _aggregates([_agg_slice(d, median=median, **kw) for d in _DATES])


def _diagnose(data, aggs=None):
    store = build_diagnosis_store(
        load_dataset_obj(data),
        load_aggregates_obj(aggs) if aggs else None,
    )
    return diagnose_employee(store, "111"), store


def _the_metric(diag):
    assert len(diag["metrics"]) == 1
    return diag["metrics"][0]


# --- устойчивая проблема vs разовый выброс ----------------------------------- #

def test_persistent_problem_is_critical():
    diag, _ = _diagnose(
        _dataset([100, 98, 90, 80, 70, 60]),
        _flat_aggs(median=100.0),
    )
    m = _the_metric(diag)
    assert m["gap"]["label"] == "критически_низкий"
    assert m["noise"]["label"] == "устойчивая_проблема"
    assert m["noise"]["below_median_streak"] >= 2
    assert m["cohort_scenario"] == "падает_сильнее_когорты"
    assert m["final_class"] == "критическая_проблема"
    assert m["problem_score"] > 0.6
    assert "разрыв_с_медианой_расширяется" in m["weak_signals"]
    assert diag["top_problem"]["metric_name"] == "Продажи"
    assert diag["overall_status"] == "критическая_проблема"
    # robust_z считается из iqr когорты, не из фолбэка.
    assert m["derived"]["robust_z_source"] == "iqr"
    assert m["derived"]["robust_z"] < -2


def test_one_off_spike_is_not_critical():
    diag, _ = _diagnose(
        _dataset([100, 101, 99, 100, 102, 70]),
        _flat_aggs(median=100.0),
    )
    m = _the_metric(diag)
    assert m["noise"]["label"] == "разовый_выброс"
    assert "резкое_падение_при_стабильной_истории" in m["patterns"]
    # По разовому выбросу критических выводов не делаем — максимум зона риска.
    assert m["final_class"] != "критическая_проблема"


# --- реалистичность плана ----------------------------------------------------- #

def test_inflated_plan_downweights_plan_signal():
    # Когорта массово не выполняет план: невыполнение — не индивидуальный сигнал.
    diag, _ = _diagnose(
        _dataset([90, 91, 92, 91, 90, 90], plan=120),
        _flat_aggs(median=90.0, mean_ex=70.0, hit_rate=30.0, top20=95.0,
                   mean_plan=100),
    )
    m = _the_metric(diag)
    assert m["plan_check"]["label"] in ("план_вероятно_завышен", "план_не_критерий")
    assert m["plan_check"]["plan_weight"] <= 0.5
    assert m["gap"]["label"] == "норма"
    assert m["decomposition"]["nature"] == "план"
    assert diag["overall_status"] == "норма"
    assert diag["top_problem"] is None


# --- достижение: несколько признаков + гейт заниженного плана ----------------- #

def test_achievement_requires_multiple_signals():
    diag, _ = _diagnose(
        _dataset([110, 115, 120, 125, 130, 135]),
        _flat_aggs(median=100.0, top20=128.0, mean_ex=97.0, hit_rate=68.0),
    )
    m = _the_metric(diag)
    assert m["gap"]["label"] == "вероятное_достижение"
    assert m["noise"]["label"] == "устойчивое_достижение"
    assert m["cohort_scenario"] == "растёт_быстрее_когорты"
    assert m["final_class"] == "значимое_достижение"
    assert diag["top_achievement"]["metric_name"] == "Продажи"
    assert diag["overall_status"] == "значимое_достижение"


def test_underrated_plan_blocks_achievement():
    diag, _ = _diagnose(
        _dataset([110, 115, 120, 125, 130, 135]),
        _flat_aggs(median=100.0, top20=128.0, mean_ex=120.0, hit_rate=95.0),
    )
    m = _the_metric(diag)
    assert m["plan_check"]["label"] == "план_вероятно_занижен"
    assert m["final_class"] != "значимое_достижение"
    assert diag["top_achievement"] is None


# --- норма -------------------------------------------------------------------- #

def test_norm_case():
    diag, _ = _diagnose(
        _dataset([100, 100, 100, 100, 100, 100]),
        _flat_aggs(median=100.0),
    )
    m = _the_metric(diag)
    assert m["gap"]["label"] == "норма"
    assert m["final_class"] == "норма"
    assert m["forecast"]["plan_risk"] == "низкий"
    assert diag["top_problem"] is None
    assert diag["overall_status"] == "норма"


# --- обратная метрика --------------------------------------------------------- #

def test_inverse_metric_orientation():
    # Обратная метрика: факт НИЖЕ медианы — это хорошо, знаки инвертируются.
    diag, _ = _diagnose(
        _dataset([80, 78, 76, 75, 74, 72], metric_type="обратная"),
        _flat_aggs(median=100.0, top20=60.0),
    )
    m = _the_metric(diag)
    assert m["derived"]["fact_vs_median"] < 0          # сырое: ниже медианы
    assert m["derived"]["fact_vs_median_oriented"] > 0  # ориентированное: лучше
    assert m["gap"]["label"] == "вероятное_достижение"
    assert m["noise"]["above_median_streak"] >= 2
    assert m["final_class"] == "значимое_достижение"


# --- деградация при неполных данных ------------------------------------------- #

def test_degradation_without_aggregates():
    diag, _ = _diagnose(_dataset([100, 98, 90, 80, 70, 60]), aggs=None)
    m = _the_metric(diag)
    assert diag["confidence"]["level"] == "низкий"
    assert any("агрегаты" in r for r in diag["confidence"]["reasons"])
    # Когортные признаки не проверялись, диагностика не падает.
    assert m["gap"]["signals_checked"] == []
    assert m["derived"]["robust_z"] is None
    assert m["cohort_scenario"] == "нет_данных"


def test_degradation_without_rankings():
    diag, _ = _diagnose(
        _dataset([100, 98, 90, 80, 70, 60]),
        _flat_aggs(median=100.0),
    )
    m = _the_metric(diag)
    assert not m["coverage"]["has_rankings"]
    assert m["percentile"] is None
    assert "percentile" not in m["gap"]["signals_checked"]
    assert any("rankings" in r for r in diag["confidence"]["reasons"])
    assert m["levels_interpretation"] == "нет_данных"


def test_rankings_add_percentile_signal():
    rankings = {
        _DATES[0]: [{"rank": "60 из 100", "level": "OFFICE", "percentile": 40.0}],
        _DATES[-1]: [{"rank": "90 из 100", "level": "OFFICE", "percentile": 10.0}],
    }
    diag, _ = _diagnose(
        _dataset([100, 98, 90, 80, 70, 60], rankings_by_date=rankings),
        _flat_aggs(median=100.0),
    )
    m = _the_metric(diag)
    assert m["percentile"] == 10.0
    assert "percentile" in m["gap"]["signals_checked"]
    assert "перцентиль_снижается" in m["weak_signals"]


# --- бинарные (звёздные) метрики ---------------------------------------------- #

def test_binary_metric_branch():
    data = _dataset([100, 100, 100, 100, 100, 100])
    data["employees"][0]["metrics"].append({
        "id": "b1", "metric_name": "Обучение пройдено", "date": _DATES[-1],
        "calc_period": "Месяц", "star_received": False, "child_metrics": [],
    })
    diag, _ = _diagnose(data, _flat_aggs(median=100.0))
    assert [b["metric_name"] for b in diag["binary_metrics"]] == ["Обучение пройдено"]
    # В числовую диагностику и скоринг бинарная метрика не попадает.
    assert all(m["metric_name"] != "Обучение пройдено" for m in diag["metrics"])
    missed = [s for s in diag["secondary_signals"]
              if s["metric_name"] == "Обучение пройдено"]
    assert missed and missed[0]["kind"] == "weak_signal"


# --- детерминизм и веса скоринга ---------------------------------------------- #

def test_determinism_and_score_weights():
    data = _dataset([100, 98, 90, 80, 70, 60])
    aggs = _flat_aggs(median=100.0)
    first, _ = _diagnose(data, aggs)
    second, _ = _diagnose(data, aggs)
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second, ensure_ascii=False, sort_keys=True
    )
    m = _the_metric(first)
    expected_problem = round(sum(
        m["score_components"]["problem"][k] * w
        for k, w in _PROBLEM_WEIGHTS.items()
    ), 4)
    expected_achievement = round(sum(
        m["score_components"]["achievement"][k] * w
        for k, w in _ACHIEVEMENT_WEIGHTS.items()
    ), 4)
    assert m["problem_score"] == expected_problem
    assert m["achievement_score"] == expected_achievement


def test_unmatched_tabnum_falls_back_to_single_employee():
    # Табельный из configurable не совпал с данными (в датасете он другой/null):
    # при единственном сотруднике фокус однозначен — датасет грузился под него.
    store = build_diagnosis_store(
        load_dataset_obj(_dataset([100, 100, 100, 100, 100, 100])),
        load_aggregates_obj(_flat_aggs(median=100.0)),
    )
    diag = diagnose_employee(store, "2000")
    assert diag.get("error") is None
    assert diag["person"]["fio"] == "Иванов Иван"


def test_fingerprint_tracks_person_data():
    data = _dataset([100, 100, 100, 100, 100, 100])
    _, store = _diagnose(data, _flat_aggs(median=100.0))
    fp1 = dataset_fingerprint(store, "111")
    assert fp1 == dataset_fingerprint(store, "111")
    changed = _dataset([100, 100, 100, 100, 100, 999])
    _, store2 = _diagnose(changed, _flat_aggs(median=100.0))
    assert dataset_fingerprint(store2, "111") != fp1
