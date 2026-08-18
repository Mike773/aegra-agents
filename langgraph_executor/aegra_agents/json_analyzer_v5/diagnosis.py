"""Детерминированная диагностика сотрудника относительно когорты.

Реализует методологию «проблема/достижение против когорты» (10 шагов): производные
метрики (robust_z, факт/медиана, относительный индекс), реалистичность плана,
значимость отклонения, фильтр «выброс vs устойчивая проблема», сценарии динамики
относительно когорты, декомпозиция разрыва, матрица паттернов-гипотез, слабые
сигналы, прогноз, кластеризация и скоринг top-1 проблемы/достижения.

Всё считается кодом, без LLM: формулировки для руководителя по этому JSON пишет
оркестратор одним отдельным вызовом. Деградация честная: нет агрегатов когорты /
rankings / плана — соответствующие признаки не проверяются, а уровень уверенности
и signals_matched это фиксируют.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from typing import Any

from .analytics import (
    _agg_rows_for_person,
    _build_agg_lookup,
    _direction_better,
    _focus_person,
    _has_plan,
    _level_order,
    _norm_metric_name,
    _ym,
    compute_analytics,
)
from .sqlite_store import SqliteStore

DIAGNOSIS_VERSION = 1

# --- Пороги методологии (числа из скилла) ----------------------------------- #
_PCTL_LOW = 20.0            # перцентиль «критически низко»
_PCTL_HIGH = 80.0           # перцентиль «вероятное достижение»
_ROBUST_Z_CRIT = 2.0        # |robust_z| значимого отклонения
_MEDIAN_DEV_SIGNIF = 0.15   # |факт/медиана − 1| значимого отклонения
_ATTAINMENT_SIGNIF_PP = 5.0  # |% выполнения − средний % когорты|, п.п.
_MEAN_EX_OVERRATED = 80.0   # средний % выполнения когорты ниже → план завышен
_MEAN_EX_DEAD = 60.0        # совсем низкий → план не критерий
_MEAN_EX_UNDERRATED = 110.0
_HIT_RATE_OVERRATED = 45.0  # доля выполнивших план ниже → план завышен
_HIT_RATE_UNDERRATED = 90.0
_PLAN_RATIO_HIGH = 1.10     # план сотрудника / средний план когорты
_RP_TOLERANCE = 0.05        # мёртвая зона изменения относительного индекса
_STREAK_PERSISTENT = 2      # серия периодов ниже/выше медианы = устойчивость
_ROLLING_WINDOW = 3         # окно скользящего среднего
_STABLE_PCT = 5.0           # мёртвая зона «стабильно», %
_NEAR_PCT = 0.10            # «близко» к плану/медиане/топ-20 (для прогноза/бенчмарка)
_SLICE_SCORE_FACTOR = 0.8   # понижающий множитель скоринга для element-разрезов
_TOP_DRIVERS = 3

_PROBLEM_SCORE_RISK = 0.35      # порог «зона риска»
_ACHIEVEMENT_SCORE_MIN = 0.45   # порог кандидата в достижения без строгого класса

_LOW, _MID, _HIGH, _NO_DATA = "низкий", "средний", "высокий", "нет_данных"


def _round(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


def _orient(value: float | None, metric_type: str | None) -> float | None:
    """Ориентированная величина: > 0 всегда означает «лучше». Для 'обратной'
    метрики (меньше = лучше) знак инвертируется."""
    if value is None:
        return None
    return value if metric_type != "обратная" else -value


def _ratio(a: float | None, b: float | None) -> float | None:
    """a/b − 1; None при отсутствии или нулевом знаменателе."""
    if a is None or b is None or b == 0:
        return None
    return a / b - 1.0


def _attainment(
    fact: float | None,
    plan: float | None,
    ex: float | None,
    metric_type: str | None,
) -> float | None:
    """% выполнения плана: пришедший ex, иначе расчёт факт/план — но только для
    'прямой' метрики. Для 'обратной' (меньше = лучше) факт/план — не выполнение,
    а его инверсия; без пришедшего ex честнее не считать вовсе."""
    if ex is not None:
        return ex
    if fact is None or not _has_plan(plan) or metric_type == "обратная":
        return None
    return fact / plan * 100.0


def build_diagnosis_store(
    rows: list[dict[str, Any]],
    agg_rows: list[dict[str, Any]] | None = None,
    ref_level: str | None = None,
    person_pairs: list[dict[str, Any]] | None = None,
) -> SqliteStore:
    """Публичная точка входа для внешних потребителей (оркестратор v4):
    SQLite + аналитика, зеркало приватного _prepare_store из nodes."""
    store = SqliteStore()
    store.load(rows)
    if agg_rows:
        store.load_aggregates(agg_rows)
    if person_pairs:
        store.load_person_aggregates(person_pairs)
    compute_analytics(store, ref_level=ref_level)
    return store


def dataset_fingerprint(store: SqliteStore, person_key: Any) -> str:
    """Отпечаток данных фокусной персоны — для диагностики расхождения кеша
    (сменились ли данные под тем же source_id). Кеш по нему НЕ инвалидируется."""
    rows = [
        (r["metric_name"], r["element"], r["date"], r["fact"], r["plan"])
        for r in store.conn.execute(
            "SELECT metric_name, element, date, fact, plan FROM metrics "
            "WHERE person_key = ? ORDER BY metric_name, element, date",
            (person_key,),
        )
    ]
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Шаг 0: покрытие данных и уверенность сравнения
# --------------------------------------------------------------------------- #
def _series_coverage(
    points: list[dict[str, Any]],
    slot: dict[str, dict],
    rank_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    latest_agg = _latest_agg(slot)
    return {
        "personal_points": len(points),
        "cohort_points": len(slot),
        "cohort_size": latest_agg.get("total_objects") if latest_agg else None,
        "has_aggregates": bool(slot),
        "has_rankings": bool(rank_rows),
        "has_plan": any(_has_plan(p["plan"]) for p in points),
    }


def _overall_confidence(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Уровень уверенности сравнения по скиллу (шаг 0). «Высокого» не бывает:
    сопоставимость состава когорты (роль/стаж/база) в данных не проверяется."""
    reasons = ["состав когорты (роль/стаж/база) в данных не проверяется"]
    covs = [m["coverage"] for m in metrics]
    level = _MID
    if not covs or not any(c["has_aggregates"] for c in covs):
        return {"level": _LOW, "reasons": reasons + ["агрегаты когорты не пришли"]}
    sizes = [c["cohort_size"] for c in covs if c["cohort_size"]]
    if sizes and max(sizes) < 5:
        level = _LOW
        reasons.append("когорта меньше 5 объектов — медиана/перцентили ненадёжны")
    max_points = max(c["personal_points"] for c in covs)
    if max_points < 3:
        level = _LOW
        reasons.append("личная история короче 3 периодов — устойчивость не отличить от выброса")
    if not any(c["has_rankings"] for c in covs):
        reasons.append("rankings/перцентиль не пришли — позиция оценена по robust_z и медиане")
    cohort_pts = max(c["cohort_points"] for c in covs)
    if cohort_pts < 3:
        reasons.append(f"история когорты — {cohort_pts} период(а)")
    return {"level": level, "reasons": reasons}


# --------------------------------------------------------------------------- #
# Шаг 1: производные метрики
# --------------------------------------------------------------------------- #
def _latest_agg(slot: dict[str, dict]) -> dict[str, Any]:
    if not slot:
        return {}
    return slot[max(slot)]


def _matched_points(
    points: list[dict[str, Any]], slot: dict[str, dict]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Личные точки, для которых есть срез когорты того же года-месяца."""
    out = []
    for p in points:
        agg = slot.get(_ym(p["date"]) or "")
        if agg is not None:
            out.append((p, agg))
    return out


def _derived_metrics(
    points: list[dict[str, Any]],
    slot: dict[str, dict],
    metric_type: str | None,
    dataset_zscore: float | None,
) -> dict[str, Any]:
    last = points[-1]
    fact, plan = last["fact"], last["plan"]
    agg = slot.get(_ym(last["date"]) or "") or _latest_agg(slot)
    median = agg.get("median")
    iqr = agg.get("iqr")
    attainment = _attainment(fact, plan, last["ex"], metric_type)

    robust_z: float | None = None
    robust_z_source: str | None = None
    if fact is not None and median is not None and iqr not in (None, 0):
        robust_z = (fact - median) / (iqr / 1.349)
        robust_z_source = "iqr"
    elif dataset_zscore is not None:
        # Фолбэк: параметрический z по людям загруженного датасета (1–5 человек),
        # сильно слабее когортного — уверенность признака ниже.
        robust_z = dataset_zscore
        robust_z_source = "fallback_dataset_zscore"

    relative_performance = [
        {"date": p["date"], "value": _round(p["fact"] / a["median"])}
        for p, a in _matched_points(points, slot)
        if p["fact"] is not None and a.get("median") not in (None, 0)
    ]

    out = {
        "fact": fact,
        "plan": plan,
        "attainment": _round(attainment),
        "fact_vs_median": _round(_ratio(fact, median)),
        "fact_vs_avg": _round(_ratio(fact, agg.get("mean_fact"))),
        "fact_vs_top20": _round(_ratio(fact, agg.get("top20_mean_fact"))),
        "gap_to_top20": _round(
            agg["top20_mean_fact"] - fact
            if fact is not None and agg.get("top20_mean_fact") is not None
            else None
        ),
        "gap_to_median": _round(
            fact - median if fact is not None and median is not None else None
        ),
        "plan_attainment_vs_cohort": _round(
            attainment - agg["mean_ex"]
            if attainment is not None and agg.get("mean_ex") is not None
            else None
        ),
        "employee_plan_vs_cohort_plan": _round(
            _ratio(plan, agg.get("mean_plan")) if _has_plan(plan) else None
        ),
        "robust_z": _round(robust_z),
        "robust_z_source": robust_z_source,
        "relative_performance": relative_performance,
        "cohort": {
            "median": median,
            "mean_fact": agg.get("mean_fact"),
            "mean_plan": agg.get("mean_plan"),
            "mean_ex": agg.get("mean_ex"),
            "hit_rate": agg.get("hit_rate"),
            "top20_mean_fact": agg.get("top20_mean_fact"),
            "iqr": iqr,
            "cv": agg.get("cv"),
            "total_objects": agg.get("total_objects"),
        },
    }
    # Ориентированные версии ключевых величин: > 0 = «лучше» независимо от
    # направления метрики; по ним ставятся все метки.
    for key in ("fact_vs_median", "fact_vs_avg", "fact_vs_top20", "robust_z"):
        out[f"{key}_oriented"] = _round(_orient(out[key], metric_type))
    return out


# --------------------------------------------------------------------------- #
# Шаг 2: реалистичность плана
# --------------------------------------------------------------------------- #
def _plan_reality(derived: dict[str, Any], has_plan: bool) -> dict[str, Any]:
    if not has_plan:
        return {"label": "план_отсутствует", "signals": [], "plan_weight": 0.0}
    c = derived["cohort"]
    over: list[str] = []
    under: list[str] = []
    if c["mean_ex"] is not None and c["mean_ex"] < _MEAN_EX_OVERRATED:
        over.append(f"средний % выполнения когорты {c['mean_ex']}")
    if c["hit_rate"] is not None and c["hit_rate"] < _HIT_RATE_OVERRATED:
        over.append(f"план выполняют лишь {c['hit_rate']}% когорты")
    if (
        c["top20_mean_fact"] is not None
        and c["mean_plan"] is not None
        and c["top20_mean_fact"] < c["mean_plan"]
    ):
        over.append("даже топ-20% не дотягивает до среднего плана")
    if (
        c["median"] is not None
        and c["mean_plan"] not in (None, 0)
        and c["median"] < c["mean_plan"] * 0.95
    ):
        over.append("медиана факта когорты заметно ниже среднего плана")
    plan_ratio = derived["employee_plan_vs_cohort_plan"]
    near_median = (
        derived["fact_vs_median_oriented"] is not None
        and abs(derived["fact_vs_median_oriented"]) <= _NEAR_PCT
    )
    if plan_ratio is not None and plan_ratio > _PLAN_RATIO_HIGH - 1 and near_median:
        over.append("план сотрудника заметно выше среднего плана когорты при факте у медианы")
    if c["mean_ex"] is not None and c["mean_ex"] > _MEAN_EX_UNDERRATED:
        under.append(f"средний % выполнения когорты {c['mean_ex']} — перевыполняют все")
    if c["hit_rate"] is not None and c["hit_rate"] > _HIT_RATE_UNDERRATED:
        under.append(f"план выполняют {c['hit_rate']}% когорты")

    if len(over) >= 3 or (
        c["mean_ex"] is not None and c["mean_ex"] < _MEAN_EX_DEAD
    ):
        return {"label": "план_не_критерий", "signals": over, "plan_weight": 0.2}
    if len(over) >= 2:
        return {"label": "план_вероятно_завышен", "signals": over, "plan_weight": 0.5}
    if len(under) >= 2:
        return {"label": "план_вероятно_занижен", "signals": under, "plan_weight": 0.5}
    return {"label": "план_реалистичный", "signals": over + under, "plan_weight": 1.0}


# --------------------------------------------------------------------------- #
# Шаг 3: значимость отклонения (по совпадению нескольких признаков)
# --------------------------------------------------------------------------- #
def _gap_significance(
    derived: dict[str, Any],
    percentile: float | None,
    plan_weight: float,
) -> dict[str, Any]:
    low: list[str] = []
    high: list[str] = []
    checked: list[str] = []

    if percentile is not None:
        checked.append("percentile")
        if percentile <= _PCTL_LOW:
            low.append(f"перцентиль {percentile}")
        elif percentile >= _PCTL_HIGH:
            high.append(f"перцентиль {percentile}")
    fvm = derived["fact_vs_median_oriented"]
    if fvm is not None:
        checked.append("fact_vs_median")
        if fvm <= -_MEDIAN_DEV_SIGNIF:
            low.append(f"факт против медианы {round(fvm * 100, 1)}%")
        elif fvm >= _MEDIAN_DEV_SIGNIF:
            high.append(f"факт против медианы +{round(fvm * 100, 1)}%")
    rz = derived["robust_z_oriented"]
    if rz is not None:
        checked.append(f"robust_z({derived['robust_z_source']})")
        if rz <= -_ROBUST_Z_CRIT:
            low.append(f"robust_z {round(rz, 2)}")
        elif rz >= _ROBUST_Z_CRIT:
            high.append(f"robust_z +{round(rz, 2)}")
    att = derived["plan_attainment_vs_cohort"]
    # %-выполнения против когорты — план-зависимый признак: при искажённом
    # плане (plan_weight < 0.5) в подсчёт не идёт.
    if att is not None and plan_weight >= 0.5:
        checked.append("plan_attainment_vs_cohort")
        if att < -_ATTAINMENT_SIGNIF_PP:
            low.append(f"% выполнения ниже среднего когорты на {round(-att, 1)} п.п.")
        elif att > _ATTAINMENT_SIGNIF_PP:
            high.append(f"% выполнения выше среднего когорты на {round(att, 1)} п.п.")

    if len(low) >= 2:
        label = "критически_низкий"
    elif len(low) == 1:
        label = "ниже_нормы"
    elif len(high) >= 2:
        label = "вероятное_достижение"
    elif len(high) == 1:
        label = "выше_нормы"
    else:
        label = "норма"
    return {
        "label": label,
        "signals_matched": low + high,
        "signals_checked": checked,
    }


# --------------------------------------------------------------------------- #
# Шаг 4: фильтрация шума — выброс или устойчивость
# --------------------------------------------------------------------------- #
def _median_streak(
    matched: list[tuple[dict[str, Any], dict[str, Any]]], metric_type: str | None
) -> int:
    """Серия последних периодов по одну сторону медианы: −k подряд ниже,
    +k подряд выше (ориентированно), 0 — нет матчей/на медиане."""
    streak = 0
    for p, a in reversed(matched):
        if p["fact"] is None or a.get("median") is None:
            break
        delta = _orient(p["fact"] - a["median"], metric_type)
        if delta is None or delta == 0:
            break
        sign = 1 if delta > 0 else -1
        if streak == 0 or (streak > 0) == (sign > 0):
            streak += sign
        else:
            break
    return streak


def _noise_filter(
    points: list[dict[str, Any]],
    matched: list[tuple[dict[str, Any], dict[str, Any]]],
    metric_type: str | None,
    cohort_cv: float | None,
) -> dict[str, Any]:
    facts = [p["fact"] for p in points if p["fact"] is not None]
    streak = _median_streak(matched, metric_type)

    rolling_delta_pct: float | None = None
    if len(facts) > _ROLLING_WINDOW:
        ma_last = statistics.fmean(facts[-_ROLLING_WINDOW:])
        ma_prev = statistics.fmean(facts[-_ROLLING_WINDOW - 1:-1])
        if ma_prev != 0:
            rolling_delta_pct = (ma_last - ma_prev) / abs(ma_prev) * 100.0

    # «До этого был стабилен» — по истории БЕЗ последней точки: скользящее
    # среднее любое резкое падение сдвигает, а признак выброса — именно
    # стабильная предыстория при резком последнем периоде.
    history = facts[:-1]
    history_stable = False
    if len(history) >= 2:
        h_mean = statistics.fmean(history)
        if h_mean != 0:
            history_stable = (
                statistics.pstdev(history) / abs(h_mean) * 100.0 < _STABLE_PCT
            )

    cv_person: float | None = None
    if len(facts) >= 3:
        mean = statistics.fmean(facts)
        if mean != 0:
            cv_person = statistics.pstdev(facts) / abs(mean) * 100.0
    volatility_flag = (
        cv_person is not None and cohort_cv is not None and cv_person > cohort_cv
    )

    if streak <= -_STREAK_PERSISTENT:
        label = "устойчивая_проблема"
    elif streak >= _STREAK_PERSISTENT:
        label = "устойчивое_достижение"
    elif streak == -1 and history_stable:
        label = "разовый_выброс"
    elif streak == 1 and history_stable:
        label = "разовый_всплеск"
    else:
        label = "стабильно"
    return {
        "label": label,
        "below_median_streak": max(0, -streak),
        "above_median_streak": max(0, streak),
        "rolling_mean_delta_pct": _round(rolling_delta_pct),
        "cv_person": _round(cv_person),
        "cv_cohort": cohort_cv,
        "volatility_flag": volatility_flag,
    }


# --------------------------------------------------------------------------- #
# Шаг 5: динамика относительно когорты
# --------------------------------------------------------------------------- #
def _rp_change(
    derived: dict[str, Any], metric_type: str | None
) -> float | None:
    rp = derived["relative_performance"]
    if len(rp) < 2:
        return None
    return _orient(rp[-1]["value"] - rp[0]["value"], metric_type)


def _group_direction(
    group_change_pct: float | None, metric_type: str | None
) -> str | None:
    """'улучшается' / 'ухудшается' / 'стабильна' по изменению среднего когорты."""
    if group_change_pct is None:
        return None
    if abs(group_change_pct) < _STABLE_PCT:
        return "стабильна"
    return (
        "улучшается"
        if _direction_better(group_change_pct > 0, metric_type)
        else "ухудшается"
    )


def _cohort_scenario(
    rp_change: float | None,
    trend_status: str | None,
    group_dir: str | None,
) -> str:
    if rp_change is None:
        return _NO_DATA
    if rp_change <= -_RP_TOLERANCE:
        return (
            "отстаёт_от_роста_когорты"
            if group_dir == "улучшается"
            else "падает_сильнее_когорты"
        )
    if rp_change >= _RP_TOLERANCE:
        return "растёт_быстрее_когорты"
    if trend_status == "ухудшение" and group_dir == "ухудшается":
        return "падает_вместе_с_когортой"
    return "стабильно"


# --------------------------------------------------------------------------- #
# Шаг 6: декомпозиция разрыва
# --------------------------------------------------------------------------- #
def _decompose(
    derived: dict[str, Any],
    plan_check: dict[str, Any],
    gap: dict[str, Any],
    noise: dict[str, Any],
    scenario: str,
    last_analytics: dict[str, Any],
    drivers: list[dict[str, Any]],
) -> dict[str, Any]:
    gaps = {
        "to_plan_abs": last_analytics.get("plan_dev_abs"),
        "to_plan_pct": last_analytics.get("plan_dev_pct"),
        "to_median": derived["gap_to_median"],
        "to_top20": derived["gap_to_top20"],
        "attainment_vs_cohort_pp": derived["plan_attainment_vs_cohort"],
    }
    if plan_check["label"] in ("план_вероятно_завышен", "план_не_критерий") and gap[
        "label"
    ] in ("норма", "выше_нормы"):
        nature = "план"
    elif gap["label"] in ("критически_низкий", "ниже_нормы") and noise["label"] == (
        "устойчивая_проблема"
    ):
        nature = "индивидуальный_результат"
    elif noise["volatility_flag"] and gap["label"] == "норма":
        nature = "стабильность"
    elif scenario in ("падает_сильнее_когорты", "отстаёт_от_роста_когорты"):
        nature = "тренд"
    elif (
        gap["label"] in ("норма", "выше_нормы")
        and derived["fact_vs_top20_oriented"] is not None
        and derived["fact_vs_top20_oriented"] <= -_MEDIAN_DEV_SIGNIF
    ):
        nature = "отставание_от_лучших"
    elif gap["label"] in ("критически_низкий", "ниже_нормы"):
        nature = "индивидуальный_результат"
    else:
        nature = "нет_проблемы"
    return {"nature": nature, "gaps": {k: _round(v) for k, v in gaps.items()}, "drivers": drivers}


def _driver_children(
    store: SqliteStore, person_key: Any, metric_uid: int
) -> list[dict[str, Any]]:
    """Прямые структурные дети узла (element IS NULL) с зоной по готовым
    вердиктам — top-N по influent_percent (паттерн build_situation_overview)."""
    kids = [
        dict(r)
        for r in store.conn.execute(
            "SELECT m.metric_name, m.fact, m.influent_percent, "
            "a.plan_status, a.trend_status, a.pop_status, a.pop_change_pct "
            "FROM metrics m LEFT JOIN metric_analytics a "
            "ON a.metric_uid = m.metric_uid "
            "WHERE m.parent_uid = ? AND m.person_key = ? AND m.element IS NULL",
            (metric_uid, person_key),
        )
    ]

    def _zone(k: dict[str, Any]) -> str:
        if "хуже_плана" == k.get("plan_status") or "ухудшение" in (
            k.get("trend_status"), k.get("pop_status")
        ):
            return "problem"
        if "лучше_плана" == k.get("plan_status") or "улучшение" in (
            k.get("trend_status"), k.get("pop_status")
        ):
            return "positive"
        return "stable"

    kids.sort(
        key=lambda k: (
            k.get("influent_percent") is None,
            -(k.get("influent_percent") or 0),
            k.get("metric_name") or "",
        )
    )
    return [
        {
            "metric_name": k["metric_name"],
            "influent_percent": k["influent_percent"],
            "zone": _zone(k),
        }
        for k in kids[:_TOP_DRIVERS]
    ]


# --------------------------------------------------------------------------- #
# Шаг 7: матрица паттернов → коды гипотез
# --------------------------------------------------------------------------- #
def _match_patterns(
    derived: dict[str, Any],
    plan_check: dict[str, Any],
    gap: dict[str, Any],
    noise: dict[str, Any],
    scenario: str,
    last_analytics: dict[str, Any],
    group_dir: str | None,
) -> list[str]:
    out: list[str] = []
    plan_missed = last_analytics.get("plan_status") == "хуже_плана"
    plan_met = last_analytics.get("plan_status") in ("в_плане", "лучше_плана")
    fvm = derived["fact_vs_median_oriented"]
    att = derived["plan_attainment_vs_cohort"]
    cohort_struggles = plan_check["label"] in ("план_вероятно_завышен", "план_не_критерий")

    if plan_missed and cohort_struggles:
        out.append("план_не_выполнен_когорта_тоже_не_выполняет")
    if plan_missed and not cohort_struggles and fvm is not None and fvm < 0:
        out.append("план_не_выполнен_когорта_выполняет_сотрудник_ниже_медианы")
    if plan_met and att is not None and att <= -_ATTAINMENT_SIGNIF_PP:
        out.append("план_выполнен_но_когорта_выполнила_сильнее")
    if fvm is not None and fvm < 0 and att is not None and att > 0:
        out.append("факт_ниже_медианы_но_выполнение_выше_среднего_когорты")
    if fvm is not None and fvm > 0 and plan_missed:
        out.append("факт_выше_медианы_но_план_не_выполнен")
    if noise["label"] == "разовый_выброс":
        out.append("резкое_падение_при_стабильной_истории")
    if noise["label"] == "устойчивая_проблема" and scenario in (
        "падает_сильнее_когорты", "отстаёт_от_роста_когорты"
    ):
        out.append("падение_несколько_периодов_при_стабильной_когорте")
    if noise["volatility_flag"]:
        out.append("нестабильный_результат_высокая_волатильность")
    if fvm is not None and fvm < 0 and last_analytics.get("trend_status") == "улучшение":
        out.append("ниже_медианы_но_тренд_улучшается")
    if fvm is not None and fvm > 0 and last_analytics.get("trend_status") == "ухудшение":
        out.append("выше_медианы_но_тренд_ухудшается")
    if (
        derived["fact_vs_top20_oriented"] is not None
        and derived["fact_vs_top20_oriented"] >= 0
        and noise["volatility_flag"]
    ):
        out.append("в_топ20_но_результат_нестабилен")
    if group_dir == "ухудшается" and noise["above_median_streak"] >= 1:
        out.append("держится_при_падающей_когорте")
    return out


# --------------------------------------------------------------------------- #
# Шаг 8: слабые сигналы
# --------------------------------------------------------------------------- #
def _oriented_gap_series(
    matched: list[tuple[dict[str, Any], dict[str, Any]]],
    metric_type: str | None,
    ref_key: str,
) -> list[float]:
    out = []
    for p, a in matched:
        ref = a.get(ref_key)
        if p["fact"] is None or ref is None:
            continue
        val = _orient(p["fact"] - ref, metric_type)
        if val is not None:
            out.append(val)
    return out


def _weak_signals(
    derived: dict[str, Any],
    matched: list[tuple[dict[str, Any], dict[str, Any]]],
    metric_type: str | None,
    percentile_series: list[tuple[str, float]],
    points: list[dict[str, Any]],
    scenario: str,
    group_dir: str | None,
) -> list[str]:
    out: list[str] = []
    if len(percentile_series) >= 2 and (
        percentile_series[-1][1] - percentile_series[0][1] <= -5
    ):
        out.append("перцентиль_снижается")
    med_gaps = _oriented_gap_series(matched, metric_type, "median")
    if len(med_gaps) >= 2 and med_gaps[-1] < med_gaps[0] and med_gaps[-1] < 0:
        out.append("разрыв_с_медианой_расширяется")
    top_gaps = _oriented_gap_series(matched, metric_type, "top20_mean_fact")
    if len(top_gaps) >= 2 and top_gaps[-1] < top_gaps[0] and top_gaps[-1] < 0:
        out.append("отставание_от_топ20_растёт")
    rp = derived["relative_performance"]
    if len(rp) >= 3:
        oriented = [_orient(x["value"], metric_type) for x in rp]
        if oriented[-1] is not None and oriented[0] is not None and (
            oriented[-1] < oriented[0] - _RP_TOLERANCE
        ):
            out.append("относительный_индекс_снижается")
    facts = [p["fact"] for p in points if p["fact"] is not None]
    if len(facts) >= 2 * _ROLLING_WINDOW:
        early = statistics.pstdev(facts[:_ROLLING_WINDOW])
        late = statistics.pstdev(facts[-_ROLLING_WINDOW:])
        if early > 0 and late > early * 1.3:
            out.append("волатильность_растёт")
    if scenario == "падает_сильнее_когорты" and group_dir == "ухудшается":
        out.append("падает_сильнее_падающей_когорты")
    return out


# --------------------------------------------------------------------------- #
# Шаг 9: прогноз (простые методы, качественные метки риска)
# --------------------------------------------------------------------------- #
def _risk_label(next_val: float | None, ref: float | None, metric_type: str | None,
                good_low: bool = False) -> str:
    """Риск не достичь ориентира: 'низкий' — прогноз не хуже ориентира,
    'средний' — в пределах _NEAR_PCT, 'высокий' — дальше. good_low инвертирует
    ответ для «шансов» (top20_chance): там «высокий» — это хорошо."""
    if next_val is None or ref in (None, 0):
        return _NO_DATA
    delta = _orient(next_val - ref, metric_type)
    if delta is None:
        return _NO_DATA
    if delta >= 0:
        return _HIGH if good_low else _LOW
    if abs(next_val - ref) / abs(ref) <= _NEAR_PCT:
        return _MID
    return _LOW if good_low else _HIGH


def _forecast(
    points: list[dict[str, Any]],
    derived: dict[str, Any],
    metric_type: str | None,
) -> dict[str, Any]:
    facts = [p["fact"] for p in points if p["fact"] is not None]
    if len(facts) < 3:
        return {
            "next_fact_estimate": None,
            "method": None,
            "plan_risk": _NO_DATA,
            "below_median_risk": _NO_DATA,
            "top20_chance": _NO_DATA,
        }
    xs = list(range(len(facts)))
    slope, intercept = statistics.linear_regression(xs, facts)
    linear_next = intercept + slope * len(facts)
    ma_next = statistics.fmean(facts[-_ROLLING_WINDOW:])
    next_est = (linear_next + ma_next) / 2.0
    c = derived["cohort"]
    return {
        "next_fact_estimate": _round(next_est),
        "next_fact_linear": _round(linear_next),
        "next_fact_ma": _round(ma_next),
        "method": "линейный_тренд+скользящее_среднее",
        "plan_risk": _risk_label(next_est, derived["plan"], metric_type),
        "below_median_risk": _risk_label(next_est, c["median"], metric_type),
        "top20_chance": _risk_label(
            next_est, c["top20_mean_fact"], metric_type, good_low=True
        ),
    }


# --------------------------------------------------------------------------- #
# Шаг 10: кластер и бенчмаркинг топ-20%
# --------------------------------------------------------------------------- #
def _cluster(
    gap: dict[str, Any],
    noise: dict[str, Any],
    scenario: str,
    weak_signals: list[str],
    percentile: float | None,
) -> str:
    g = gap["label"]
    if g == "критически_низкий" and noise["label"] == "устойчивая_проблема":
        return "аутсайдер"
    if g in ("вероятное_достижение", "выше_нормы") and noise["label"] in (
        "устойчивое_достижение", "стабильно"
    ) and not noise["volatility_flag"]:
        return "звезда_сильный_исполнитель"
    if noise["volatility_flag"] and noise["label"] in (
        "разовый_выброс", "разовый_всплеск", "стабильно"
    ):
        return "нестабильный_исполнитель"
    if g in ("ниже_нормы", "норма") and scenario in (
        "растёт_быстрее_когорты",
    ):
        return "потенциал_растущий"
    if g in ("норма", "ниже_нормы") and (
        weak_signals or scenario in ("падает_сильнее_когорты", "отстаёт_от_роста_когорты")
    ):
        return "риск_раннее_ухудшение"
    if g in ("критически_низкий",):
        return "аутсайдер"
    return "устойчивый_середняк"


def _benchmark_top20(derived: dict[str, Any], percentile: float | None) -> dict[str, Any]:
    c = derived["cohort"]
    top20_vs_median = _ratio(c["top20_mean_fact"], c["median"])
    fvm = derived["fact_vs_median_oriented"]
    if fvm is not None and fvm < 0:
        target = "выйти_на_медиану"
    elif percentile is not None and percentile < 70:
        target = "закрепиться_в_60-70_перцентиле"
    else:
        target = "сокращать_разрыв_с_топ20"
    return {
        "gap_to_top20": derived["gap_to_top20"],
        "top20_vs_median": _round(top20_vs_median),
        "intermediate_target": target,
    }


# --------------------------------------------------------------------------- #
# Атрибуция и интерпретация рейтингов по уровням
# --------------------------------------------------------------------------- #
def _attribution(
    gap: dict[str, Any],
    noise: dict[str, Any],
    scenario: str,
    group_dir: str | None,
    percentile_series: list[tuple[str, float]],
) -> str:
    pct_declining = len(percentile_series) >= 2 and (
        percentile_series[-1][1] < percentile_series[0][1]
    )
    if (
        group_dir == "ухудшается"
        and scenario in ("падает_вместе_с_когортой", "стабильно", _NO_DATA)
        and not pct_declining
    ):
        return "внешний_фактор"
    if (
        group_dir in ("стабильна", "улучшается", None)
        and gap["label"] in ("критически_низкий", "ниже_нормы")
        and (noise["label"] == "устойчивая_проблема" or scenario in (
            "падает_сильнее_когорты", "отстаёт_от_роста_когорты"
        ))
    ):
        return "индивидуальная_проблема"
    return "неопределённо"


def _levels_interpretation(
    level_pctl: list[tuple[str, float]],
) -> str:
    """level_pctl — [(level, percentile)] от УЗКОГО уровня к широкому."""
    if len(level_pctl) < 2:
        return _NO_DATA
    values = [p for _, p in level_pctl]
    if all(p <= 30 for p in values):
        return "низкий_на_всех_уровнях"
    if all(p >= 70 for p in values):
        return "высокий_на_всех_уровнях"
    narrow, broad = values[0], values[-1]
    if narrow - broad >= 30:
        return "локально_высокий"
    if broad - narrow >= 30:
        return "локально_низкий"
    return "равномерно"


# --------------------------------------------------------------------------- #
# Приоритизация: скоринг проблемы/достижения (веса скилла)
# --------------------------------------------------------------------------- #
def _problem_components(
    gap: dict[str, Any],
    noise: dict[str, Any],
    scenario: str,
    forecast: dict[str, Any],
    weak_signals: list[str],
) -> dict[str, float]:
    deviation = {"критически_низкий": 1.0, "ниже_нормы": 0.6}.get(gap["label"], 0.0)
    persistence = min(1.0, noise["below_median_streak"] / 3.0)
    dynamics = {
        "падает_сильнее_когорты": 1.0,
        "отстаёт_от_роста_когорты": 0.6,
        "падает_вместе_с_когортой": 0.2,
    }.get(scenario, 0.0)
    fc = 0.0
    if _HIGH in (forecast["plan_risk"], forecast["below_median_risk"]):
        fc = 1.0
    elif _MID in (forecast["plan_risk"], forecast["below_median_risk"]):
        fc = 0.5
    return {
        "deviation": deviation,
        "persistence": persistence,
        "negative_dynamics": dynamics,
        "forecast": fc,
        "weak_signals": min(1.0, len(weak_signals) / 3.0),
        "volatility": 1.0 if noise["volatility_flag"] else 0.0,
    }


def _achievement_components(
    derived: dict[str, Any],
    gap: dict[str, Any],
    noise: dict[str, Any],
    scenario: str,
    forecast: dict[str, Any],
) -> dict[str, float]:
    exceed = {"вероятное_достижение": 1.0, "выше_нормы": 0.6}.get(gap["label"], 0.0)
    sustainability = min(1.0, noise["above_median_streak"] / 3.0)
    dynamics = 1.0 if scenario == "растёт_быстрее_когорты" else 0.0
    top20 = derived["fact_vs_top20_oriented"]
    proximity = 0.0
    if top20 is not None:
        proximity = 1.0 if top20 >= 0 else (0.6 if top20 >= -_NEAR_PCT else 0.0)
    fc = {_HIGH: 1.0, _MID: 0.5}.get(forecast["top20_chance"], 0.0)
    return {
        "exceed": exceed,
        "sustainability": sustainability,
        "positive_dynamics": dynamics,
        "top20_proximity": proximity,
        "forecast": fc,
        "stability": 0.0 if noise["volatility_flag"] else 1.0,
    }


_PROBLEM_WEIGHTS = {
    "deviation": 0.30,
    "persistence": 0.20,
    "negative_dynamics": 0.20,
    "forecast": 0.15,
    "weak_signals": 0.10,
    "volatility": 0.05,
}
_ACHIEVEMENT_WEIGHTS = {
    "exceed": 0.30,
    "sustainability": 0.20,
    "positive_dynamics": 0.20,
    "top20_proximity": 0.15,
    "forecast": 0.10,
    "stability": 0.05,
}


def _weighted(components: dict[str, float], weights: dict[str, float]) -> float:
    return round(sum(components[k] * w for k, w in weights.items()), 4)


# --------------------------------------------------------------------------- #
# Итоговая классификация (несколько признаков, не один)
# --------------------------------------------------------------------------- #
def _final_class(
    gap: dict[str, Any],
    noise: dict[str, Any],
    scenario: str,
    plan_check: dict[str, Any],
    derived: dict[str, Any],
    forecast: dict[str, Any],
    weak_signals: list[str],
    problem_score: float,
) -> str:
    # Разовый выброс/всплеск блокирует крайние классы: по нему нельзя делать
    # критических выводов и нельзя объявлять достижение (только мониторинг).
    one_off = noise["label"] in ("разовый_выброс", "разовый_всплеск")
    problem_signs = sum(
        [
            gap["label"] == "критически_низкий",
            noise["below_median_streak"] >= _STREAK_PERSISTENT,
            scenario in ("падает_сильнее_когорты", "отстаёт_от_роста_когорты"),
            (
                derived["plan_attainment_vs_cohort"] is not None
                and derived["plan_attainment_vs_cohort"] <= -_ATTAINMENT_SIGNIF_PP
                and plan_check["label"] == "план_реалистичный"
            ),
            _HIGH in (forecast["plan_risk"], forecast["below_median_risk"]),
            bool(weak_signals),
        ]
    )
    if gap["label"] == "критически_низкий" and problem_signs >= 3 and not one_off:
        return "критическая_проблема"

    achievement_signs = sum(
        [
            gap["label"] == "вероятное_достижение",
            (
                derived["fact_vs_top20_oriented"] is not None
                and derived["fact_vs_top20_oriented"] >= -_NEAR_PCT
            ),
            noise["above_median_streak"] >= _STREAK_PERSISTENT,
            scenario == "растёт_быстрее_когорты",
        ]
    )
    # Заниженный план — жёсткий гейт: перевыполнение при плане, который
    # выполняют почти все, достижением не считается.
    if (
        gap["label"] == "вероятное_достижение"
        and achievement_signs >= 3
        and not one_off
        and plan_check["label"] != "план_вероятно_занижен"
    ):
        return "значимое_достижение"
    if problem_score >= _PROBLEM_SCORE_RISK:
        return "зона_риска"
    return "норма"


# --------------------------------------------------------------------------- #
# Основной вход
# --------------------------------------------------------------------------- #
def _person_rank_rows(store: SqliteStore, person_key: Any) -> dict[str, list[dict]]:
    """Rankings фокусной персоны по метрикам: metric_name → строки (level,
    percentile, date), отсортированные по дате."""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in store.conn.execute(
        "SELECT m.metric_name AS metric_name, r.level AS level, "
        "r.percentile AS percentile, r.rank_raw AS rank_raw, m.date AS date "
        "FROM metric_rankings r JOIN metrics m ON m.metric_uid = r.metric_uid "
        "WHERE m.person_key = ? AND m.element IS NULL "
        "ORDER BY m.date, r.level",
        (person_key,),
    ):
        out[r["metric_name"]].append(dict(r))
    return out


def _levels_by_width(agg_rows: list[dict[str, Any]]) -> list[str]:
    """Коды уровней от узкой группы к широкой (для интерпретации рейтингов)."""
    return _level_order(agg_rows)


def _ref_percentile(
    rank_rows: list[dict], level_order_codes: list[str]
) -> tuple[float | None, list[tuple[str, float]], list[tuple[str, float]]]:
    """(перцентиль референсного уровня, серия перцентилей по датам этого уровня,
    срез последней даты по уровням от узкого к широкому)."""
    dated = [r for r in rank_rows if r["percentile"] is not None]
    if not dated:
        return None, [], []
    latest_date = max(r["date"] or "" for r in dated)
    order = {code: i for i, code in enumerate(level_order_codes)}
    latest = sorted(
        (r for r in dated if (r["date"] or "") == latest_date),
        key=lambda r: order.get(r["level"], len(order)),
    )
    level_pctl = [(r["level"], r["percentile"]) for r in latest]
    ref_level = latest[0]["level"] if latest else None
    series = [
        (r["date"], r["percentile"])
        for r in dated
        if r["level"] == ref_level and r["date"]
    ]
    ref = level_pctl[0][1] if level_pctl else None
    return ref, series, level_pctl


def diagnose_employee(
    store: SqliteStore,
    person: Any | None = None,
    ref_level: str | None = None,
) -> dict[str, Any]:
    """Полная диагностика фокусного сотрудника по методологии — детерминированный
    JSON с метками всех шагов, скорингом и top-1 проблемой/достижением."""
    pkey, fio = _focus_person(store, person)
    if pkey is None and person is not None:
        # employee_tabnum из configurable может не совпасть с данными (табельный
        # в датасете null или в другой форме). Датасет грузился ПОД этого
        # сотрудника: при ровно одном сотруднике фокус однозначен и без матча.
        emps = [
            dict(r)
            for r in store.conn.execute(
                "SELECT DISTINCT person_key, person_fio FROM metrics "
                "WHERE person_is_me = 0"
            )
        ]
        if len(emps) == 1:
            pkey, fio = emps[0]["person_key"], emps[0]["person_fio"]
    if pkey is None:
        return {"error": "Фокусный сотрудник не найден в датасете."}
    person_row = store.conn.execute(
        "SELECT MAX(person_tabnum) AS tabnum, MAX(person_post) AS post "
        "FROM metrics WHERE person_key = ?",
        (pkey,),
    ).fetchone()

    # Агрегаты, видимые персоне (предагрегаты индивидуальны), и lookup по
    # референсному уровню — тот же путь, что в compute_analytics.
    all_agg = [dict(r) for r in store.conn.execute("SELECT * FROM peer_aggregates")]
    agg_rows = _agg_rows_for_person(all_agg, pkey, store.person_aggregate_ids())
    lookup = _build_agg_lookup(agg_rows, ref_level)
    level_codes = _levels_by_width(agg_rows)
    ref_level_name = None
    if level_codes:
        ref_level_name = store.level_names().get(
            str(level_codes[0]).strip().casefold()
        )
    rank_by_metric = _person_rank_rows(store, pkey)

    rows = [
        dict(r)
        for r in store.conn.execute(
            "SELECT m.metric_uid, m.parent_uid, m.depth, m.person_key, "
            "m.metric_id, m.metric_name, m.metric_type, m.measure_type, "
            "m.element, m.date, "
            "m.fact, m.plan, m.ex, m.influent_percent, m.star_received, "
            "m.is_star_metric, "
            "a.plan_status, a.plan_dev_abs, a.plan_dev_pct, a.trend_status, "
            "a.pop_status, a.pop_change_pct, a.group_change_pct, "
            "a.rel_change_pct, a.rel_status, a.zscore "
            "FROM metrics m LEFT JOIN metric_analytics a "
            "ON a.metric_uid = m.metric_uid "
            "WHERE m.person_key = ? "
            "ORDER BY m.metric_name, m.element, m.date, m.metric_uid",
            (pkey,),
        )
    ]

    series: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        series[(r["metric_name"], r["element"])].append(r)

    # У метрики с агрегатным срезом диагностируем его; метрику, представленную
    # только разрезами, диагностируем поразрезно с понижающим множителем скоринга.
    keys: list[tuple[str, str | None]] = []
    names_with_agg = {name for (name, el) in series if el is None}
    for key in series:
        name, el = key
        if el is None or name not in names_with_agg:
            keys.append(key)
    keys.sort(key=lambda k: (k[0] or "", k[1] or ""))

    metrics_out: list[dict[str, Any]] = []
    binary_out: list[dict[str, Any]] = []
    for name, element in keys:
        points = series[(name, element)]
        facts = [p for p in points if p["fact"] is not None]
        if not facts:
            star_rows = [p for p in points if p["star_received"] is not None]
            if star_rows and element is None:
                last = star_rows[-1]
                binary_out.append(
                    {
                        "metric_name": name,
                        "date": last["date"],
                        "star_received": last["star_received"],
                        "is_star_metric": last["is_star_metric"],
                    }
                )
            continue
        entry = _diagnose_series(
            store,
            points,
            lookup.get(_norm_metric_name(name), {}),
            rank_by_metric.get(name, []),
            level_codes,
        )
        metrics_out.append(entry)

    confidence = _overall_confidence(metrics_out)

    problem_pool = [
        m for m in metrics_out
        if m["final_class"] in ("критическая_проблема", "зона_риска")
    ]
    top_problem = max(problem_pool, key=lambda m: m["problem_score"], default=None)
    ach_pool = [m for m in metrics_out if m["final_class"] == "значимое_достижение"]
    if not ach_pool:
        # Фолбэк-кандидаты: высокий achievement_score, но те же жёсткие гейты,
        # что у строгого класса (не разовый всплеск, план не занижен).
        ach_pool = [
            m for m in metrics_out
            if m["achievement_score"] >= _ACHIEVEMENT_SCORE_MIN
            and m["noise"]["label"] != "разовый_всплеск"
            and m["plan_check"]["label"] != "план_вероятно_занижен"
        ]
    top_achievement = max(
        ach_pool, key=lambda m: m["achievement_score"], default=None
    )

    def _brief(m: dict[str, Any] | None) -> dict[str, Any] | None:
        if m is None:
            return None
        return {
            "metric_name": m["metric_name"],
            "element": m["element"],
            "final_class": m["final_class"],
            "problem_score": m["problem_score"],
            "achievement_score": m["achievement_score"],
        }

    secondary: list[dict[str, Any]] = []
    for m in metrics_out:
        if m is top_problem or m is top_achievement:
            continue
        if m["final_class"] in ("критическая_проблема", "зона_риска") or (
            m["problem_score"] >= 0.3
        ):
            secondary.append(
                {"metric_name": m["metric_name"], "element": m["element"],
                 "kind": "problem", "score": m["problem_score"]}
            )
        elif m["achievement_score"] >= _ACHIEVEMENT_SCORE_MIN:
            secondary.append(
                {"metric_name": m["metric_name"], "element": m["element"],
                 "kind": "achievement", "score": m["achievement_score"]}
            )
        elif m["weak_signals"]:
            secondary.append(
                {"metric_name": m["metric_name"], "element": m["element"],
                 "kind": "weak_signal", "score": m["problem_score"]}
            )
    for b in binary_out:
        if not b["star_received"]:
            secondary.append(
                {"metric_name": b["metric_name"], "element": None,
                 "kind": "weak_signal", "score": None,
                 "note": "бинарный показатель не получен"}
            )
    secondary.sort(key=lambda s: (-(s["score"] or 0), s["metric_name"] or ""))

    if top_problem is not None and top_problem["final_class"] == "критическая_проблема":
        overall = "критическая_проблема"
    elif top_problem is not None and top_achievement is not None:
        overall = "смешанный"
    elif top_problem is not None:
        overall = "зона_риска"
    elif top_achievement is not None:
        overall = "значимое_достижение"
    else:
        overall = "норма"

    return {
        "version": DIAGNOSIS_VERSION,
        "person": {
            "person_key": pkey,
            "fio": fio,
            "tabnum": person_row["tabnum"] if person_row else None,
            "post": person_row["post"] if person_row else None,
        },
        "ref_level_name": ref_level_name,
        "confidence": confidence,
        "metrics": metrics_out,
        "binary_metrics": binary_out,
        "top_problem": _brief(top_problem),
        "top_achievement": _brief(top_achievement),
        "secondary_signals": secondary,
        "overall_status": overall,
    }


def _diagnose_series(
    store: SqliteStore,
    points: list[dict[str, Any]],
    slot: dict[str, dict],
    rank_rows: list[dict[str, Any]],
    level_codes: list[str],
) -> dict[str, Any]:
    last = points[-1]
    metric_type = last["metric_type"]
    matched = _matched_points(points, slot)
    coverage = _series_coverage(points, slot, rank_rows)

    derived = _derived_metrics(points, slot, metric_type, last.get("zscore"))
    plan_check = _plan_reality(derived, coverage["has_plan"])
    percentile, pctl_series, level_pctl = _ref_percentile(rank_rows, level_codes)
    gap = _gap_significance(derived, percentile, plan_check["plan_weight"])
    noise = _noise_filter(points, matched, metric_type, derived["cohort"]["cv"])
    rp_change = _rp_change(derived, metric_type)
    group_dir = _group_direction(last.get("group_change_pct"), metric_type)
    scenario = _cohort_scenario(rp_change, last.get("trend_status"), group_dir)
    weak = _weak_signals(
        derived, matched, metric_type, pctl_series, points, scenario, group_dir
    )
    drivers = (
        _driver_children(store, last["person_key"], last["metric_uid"])
        if last["depth"] == 1 and last["element"] is None
        else []
    )
    decomposition = _decompose(
        derived, plan_check, gap, noise, scenario, last, drivers
    )
    patterns = _match_patterns(
        derived, plan_check, gap, noise, scenario, last, group_dir
    )
    forecast = _forecast(points, derived, metric_type)
    cluster = _cluster(gap, noise, scenario, weak, percentile)

    p_components = _problem_components(gap, noise, scenario, forecast, weak)
    a_components = _achievement_components(derived, gap, noise, scenario, forecast)
    slice_factor = 1.0 if last["element"] is None else _SLICE_SCORE_FACTOR
    problem_score = round(_weighted(p_components, _PROBLEM_WEIGHTS) * slice_factor, 4)
    achievement_score = round(
        _weighted(a_components, _ACHIEVEMENT_WEIGHTS) * slice_factor, 4
    )
    final = _final_class(
        gap, noise, scenario, plan_check, derived, forecast, weak, problem_score
    )
    return {
        "metric_id": last["metric_id"],
        "metric_name": last["metric_name"],
        "element": last["element"],
        "metric_type": metric_type,
        "is_root": last["depth"] == 1,
        "influent_percent": last["influent_percent"],
        "coverage": coverage,
        "derived": derived,
        "plan_check": plan_check,
        "percentile": percentile,
        "levels_percentile": level_pctl,
        "gap": gap,
        "noise": noise,
        "cohort_scenario": scenario,
        "rp_change_oriented": _round(rp_change),
        "decomposition": decomposition,
        "patterns": patterns,
        "weak_signals": weak,
        "forecast": forecast,
        "cluster": cluster,
        "benchmark_top20": _benchmark_top20(derived, percentile),
        "levels_interpretation": _levels_interpretation(level_pctl),
        "attribution": _attribution(gap, noise, scenario, _group_direction(
            last.get("group_change_pct"), metric_type
        ), pctl_series),
        "final_class": final,
        "problem_score": problem_score,
        "achievement_score": achievement_score,
        "score_components": {
            "problem": p_components,
            "achievement": a_components,
        },
    }
