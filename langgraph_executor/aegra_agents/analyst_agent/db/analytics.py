"""Детерминированный (без LLM) расчёт производных полей.

Формулы — построчный порт json_analyzer_v5/analytics.py; меняется только
вход/выход: читаем нормализованные таблицы, пишем в fact_analytics. Паритет
закреплён тестом tests/analyst_agent/test_analytics_parity.py.

Считаем в Python, а не SQL-вью: z-score требует stdev (в SQLite нет), гейт
«есть план» и семантика «prev = предыдущая строка серии» уже отлажены, а
версия SQLite на проме не гарантирована.
"""
from __future__ import annotations

import os
import statistics
from collections import defaultdict
from typing import Any

from ...shared.peer_levels import level_order

ANALYTICS_COLUMNS: tuple[str, ...] = (
    "plan_dev_abs",
    "plan_dev_pct",
    "plan_status",
    "benchmark_dev_abs",
    "benchmark_dev_pct",
    "benchmark_status",
    "pop_change_abs",
    "pop_change_pct",
    "pop_status",
    "trend",
    "trend_status",
    "peer_mean",
    "peer_std",
    "peer_count",
    "peer_rank",
    "peer_percentile",
    "zscore",
    "peer_status",
    "is_anomaly",
    "group_change_pct",
    "rel_change_pct",
    "rel_status",
    "group_hit_rate",
    "plan_rigidity",
)

_PLAN_LABELS = ("лучше_плана", "в_плане", "хуже_плана")
_BENCH_LABELS = ("лучше_бенчмарка", "на_уровне_бенчмарка", "хуже_бенчмарка")
_DYNAMIC_LABELS = ("улучшение", "стабильно", "ухудшение")
_PEER_LABELS = ("лучше_коллег", "на_уровне_коллег", "хуже_коллег")
_REL_LABELS = ("лучше_группы", "на_уровне_группы", "хуже_группы")
_RIGIDITY_LABELS = ("жёсткий_план", "обычный_план", "мягкий_план")

_IN_PLAN_TOLERANCE_PCT = 1.0
_TREND_TOLERANCE_PCT = 5.0
# Зона «на уровне коллег» по модулю z-score.
_PEER_NEUTRAL_ZSCORE = 0.5
# Зона «на уровне группы» для детрендинга, процентные пункты.
_REL_TOLERANCE_PCT = 2.0
# Пороги жёсткости плана по доле выполнивших план в группе (hit_rate).
_HIT_RATE_HARD = 33.3
_HIT_RATE_SOFT = 66.7


def _anomaly_threshold() -> float:
    raw = os.environ.get("ANOMALY_ZSCORE_THRESHOLD", "2.0").strip()
    try:
        return float(raw)
    except ValueError:
        return 2.0


def direction_better(value_up: bool, direction: str | None) -> bool:
    """value_up — значение РАСТЁТ. Лучше ли это с учётом направления метрики.
    Для 'обратной' (меньше = лучше) рост значения — ухудшение."""
    higher_is_better = direction != "обратная"
    return value_up == higher_is_better


def _deviation(
    fact: float | None,
    ref: float | None,
    direction: str | None,
    labels: tuple[str, str, str],
) -> tuple[float | None, float | None, str | None]:
    better, equal, worse = labels
    if fact is None or ref is None:
        return None, None, None
    dev_abs = fact - ref
    dev_pct = (dev_abs / ref * 100.0) if ref != 0 else None
    if dev_abs == 0 or (dev_pct is not None and abs(dev_pct) < _IN_PLAN_TOLERANCE_PCT):
        return dev_abs, dev_pct, equal
    return dev_abs, dev_pct, (better if direction_better(dev_abs > 0, direction) else worse)


def _trend(facts: list[float]) -> str | None:
    if len(facts) < 2:
        return None
    first, last = facts[0], facts[-1]
    if first == 0:
        change_pct = 0.0 if last == 0 else 100.0
    else:
        change_pct = (last - first) / abs(first) * 100.0
    if abs(change_pct) < _TREND_TOLERANCE_PCT:
        return "стабильно"
    return "рост" if last > first else "падение"


def _trend_status(trend: str | None, direction: str | None) -> str | None:
    better, stable, worse = _DYNAMIC_LABELS
    if trend is None:
        return None
    if trend == "стабильно":
        return stable
    return better if direction_better(trend == "рост", direction) else worse


def _pop_status(pop_change_pct: float | None, direction: str | None) -> str | None:
    better, stable, worse = _DYNAMIC_LABELS
    if pop_change_pct is None:
        return None
    if abs(pop_change_pct) < _TREND_TOLERANCE_PCT:
        return stable
    return better if direction_better(pop_change_pct > 0, direction) else worse


def _peer_status(zscore: float | None, direction: str | None) -> str | None:
    better, neutral, worse = _PEER_LABELS
    if zscore is None:
        return None
    if abs(zscore) < _PEER_NEUTRAL_ZSCORE:
        return neutral
    return better if direction_better(zscore > 0, direction) else worse


def _rel_status(rel_change_pct: float | None, direction: str | None) -> str | None:
    better, neutral, worse = _REL_LABELS
    if rel_change_pct is None:
        return None
    if abs(rel_change_pct) < _REL_TOLERANCE_PCT:
        return neutral
    return better if direction_better(rel_change_pct > 0, direction) else worse


def _plan_rigidity(hit_rate: float | None) -> str | None:
    hard, normal, soft = _RIGIDITY_LABELS
    if hit_rate is None:
        return None
    if hit_rate < _HIT_RATE_HARD:
        return hard
    if hit_rate > _HIT_RATE_SOFT:
        return soft
    return normal


def has_plan(plan: Any) -> bool:
    """«План указан» = задан и НЕнулевой (0 — плейсхолдер, а не цель).
    Без плана временные сравнения (pop/trend) не строим."""
    return plan is not None and plan != 0


def _round(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


def _agg_rows_for_person(
    agg_rows: list[dict], person_key: Any, person_ids: dict[str, set[str]]
) -> list[dict]:
    """Строки агрегатов, видимые персоне: строка с aggregate_id видна только
    привязанным персонам; без привязки персона видит всё (обратная совместимость)."""
    ids = person_ids.get(str(person_key)) if person_key is not None else None
    if not ids:
        return agg_rows
    return [
        r for r in agg_rows
        if r.get("aggregate_id") is None or str(r["aggregate_id"]) in ids
    ]


def _level_priority(agg_rows: list[dict], ref_level: str | None) -> list[str]:
    """Уровни от узкой группы к широкой (level_order уже посчитан при сборке,
    но явный ref_level может поднять свой уровень на первое место)."""
    order: list[str] = []
    seen: set[str] = set()
    for r in sorted(agg_rows, key=lambda x: (x.get("level_order") or 0, x.get("agg_id") or 0)):
        level = r.get("level")
        if level is not None and level not in seen:
            seen.add(level)
            order.append(level)
    if ref_level:
        wanted = str(ref_level).strip().casefold()
        for lv in order:
            if str(lv).strip().casefold() == wanted:
                order = [lv] + [x for x in order if x != lv]
                break
    return order


def _build_agg_lookup(agg_rows: list[dict], ref_level: str | None) -> dict[Any, dict[str, dict]]:
    """metric_id → {ym → строка агрегатов} по референсному уровню.

    Уровень выбирается НА МЕТРИКУ: первый по приоритету, где у метрики есть
    строки. При коллизии ym приоритет у текущего среза (is_current)."""
    order = _level_priority(agg_rows, ref_level)
    by_metric_level: dict[tuple[Any, str], dict[str, dict]] = {}
    for r in agg_rows:
        metric_id, ym, level = r.get("metric_id"), r.get("ym"), r.get("level")
        if metric_id is None or ym is None or level is None:
            continue
        slot = by_metric_level.setdefault((metric_id, level), {})
        cur = slot.get(ym)
        if cur is None or (not cur["is_current"] and r["is_current"]):
            slot[ym] = r
    lookup: dict[Any, dict[str, dict]] = {}
    for metric_id in {key[0] for key in by_metric_level}:
        for level in order:
            slot = by_metric_level.get((metric_id, level))
            if slot:
                lookup[metric_id] = slot
                break
    return lookup


def compute_analytics(conn, ref_level: str | None = None) -> int:
    """Пересчитывает fact_analytics целиком (DELETE + INSERT)."""
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT f.fact_id, f.person_id, p.person_key, p.is_me, f.metric_id, "
            "m.name AS metric_name, m.direction, f.element, d.date, d.ym, "
            "f.fact, f.plan, f.benchmark "
            "FROM fact f "
            "JOIN person p ON p.person_id = f.person_id "
            "JOIN metric m ON m.metric_id = f.metric_id "
            "JOIN period d ON d.period_id = f.period_id"
        )
    ]

    all_agg_rows = [dict(r) for r in conn.execute("SELECT * FROM peer_aggregate")]
    person_ids: dict[str, set[str]] = {}
    for r in conn.execute(
        "SELECT p.person_key, pp.aggregate_id FROM person_peer pp "
        "JOIN person p ON p.person_id = pp.person_id"
    ):
        person_ids.setdefault(r["person_key"], set()).add(r["aggregate_id"])
    lookup_cache: dict[frozenset[str], dict[Any, dict[str, dict]]] = {}

    def _lookup_for(person_key: Any) -> dict[Any, dict[str, dict]]:
        ids = person_ids.get(str(person_key)) if person_key is not None else None
        cache_key = frozenset(ids or ())
        cached = lookup_cache.get(cache_key)
        if cached is None:
            cached = _build_agg_lookup(
                _agg_rows_for_person(all_agg_rows, person_key, person_ids), ref_level
            )
            lookup_cache[cache_key] = cached
        return cached

    def _agg_for(r: dict) -> dict | None:
        if r["ym"] is None:
            return None
        return _lookup_for(r["person_key"]).get(r["metric_id"], {}).get(r["ym"])

    result: dict[int, dict[str, Any]] = {}
    for r in rows:
        plan_abs, plan_pct, plan_status = _deviation(
            r["fact"], r["plan"], r["direction"], _PLAN_LABELS
        )
        # Бенчмарк считается против top20_mean_fact peer-группы (средний факт
        # топ-20% объектов), а не против поля benchmark — оно приходит пустым.
        agg = _agg_for(r)
        bench_abs, bench_pct, bench_status = _deviation(
            r["fact"],
            agg["top20_mean_fact"] if agg else None,
            r["direction"],
            _BENCH_LABELS,
        )
        group_hit_rate = agg["hit_rate"] if agg is not None and has_plan(r["plan"]) else None
        result[r["fact_id"]] = {
            "plan_dev_abs": plan_abs,
            "plan_dev_pct": plan_pct,
            "plan_status": plan_status,
            "benchmark_dev_abs": bench_abs,
            "benchmark_dev_pct": bench_pct,
            "benchmark_status": bench_status,
            "pop_change_abs": None,
            "pop_change_pct": None,
            "pop_status": None,
            "trend": None,
            "trend_status": None,
            "peer_mean": None,
            "peer_std": None,
            "peer_count": None,
            "peer_rank": None,
            "peer_percentile": None,
            "zscore": None,
            "peer_status": None,
            "is_anomaly": 0,
            "group_change_pct": None,
            "rel_change_pct": None,
            "rel_status": None,
            "group_hit_rate": group_hit_rate,
            "plan_rigidity": _plan_rigidity(group_hit_rate),
        }

    # --- динамика во времени: серия = (персона, метрика, разрез) -----------
    series: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        series[(r["person_key"], r["metric_name"], r["element"])].append(r)
    for items in series.values():
        items.sort(key=lambda x: x["date"] or "")
        series_has_plan = any(has_plan(it.get("plan")) for it in items)
        prev: dict | None = None
        for r in items:
            if (
                prev is not None
                and r["fact"] is not None
                and prev["fact"] is not None
                and has_plan(r.get("plan"))
            ):
                change_abs = r["fact"] - prev["fact"]
                change_pct = change_abs / prev["fact"] * 100.0 if prev["fact"] != 0 else None
                res = result[r["fact_id"]]
                res["pop_change_abs"] = change_abs
                res["pop_change_pct"] = change_pct
                res["pop_status"] = _pop_status(change_pct, r["direction"])
                # Детрендинг: личное изменение минус изменение группы за те же
                # периоды. Нужны mean_fact обоих периодов — иначе None.
                cur_agg, prev_agg = _agg_for(r), _agg_for(prev)
                if (
                    change_pct is not None
                    and cur_agg is not None
                    and prev_agg is not None
                    and cur_agg["mean_fact"] is not None
                    and prev_agg["mean_fact"] not in (None, 0)
                ):
                    group_pct = (
                        (cur_agg["mean_fact"] - prev_agg["mean_fact"])
                        / prev_agg["mean_fact"] * 100.0
                    )
                    rel_pct = change_pct - group_pct
                    res["group_change_pct"] = group_pct
                    res["rel_change_pct"] = rel_pct
                    res["rel_status"] = _rel_status(rel_pct, r["direction"])
            prev = r
        trend = (
            _trend([r["fact"] for r in items if r["fact"] is not None])
            if series_has_plan
            else None
        )
        for r in items:
            result[r["fact_id"]]["trend"] = trend
            result[r["fact_id"]]["trend_status"] = _trend_status(trend, r["direction"])

    # --- сравнение с коллегами: группа = (метрика, разрез, дата) -----------
    threshold = _anomaly_threshold()
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["is_me"]:
            continue
        groups[(r["metric_name"], r["element"], r["date"])].append(r)
    for items in groups.values():
        valued = [r for r in items if r["fact"] is not None]
        n = len(valued)
        if n < 2:
            continue
        facts = [r["fact"] for r in valued]
        mean = statistics.fmean(facts)
        std = statistics.pstdev(facts)
        higher_is_better = items[0]["direction"] != "обратная"
        valued.sort(key=lambda x: x["fact"], reverse=higher_is_better)
        for idx, r in enumerate(valued):
            zscore = (r["fact"] - mean) / std if std > 0 else 0.0
            res = result[r["fact_id"]]
            res["peer_mean"] = mean
            res["peer_std"] = std
            res["peer_count"] = n
            res["peer_rank"] = idx + 1
            res["peer_percentile"] = round((1 - idx / (n - 1)) * 100, 1)
            res["zscore"] = zscore
            res["peer_status"] = _peer_status(zscore, r["direction"])
            res["is_anomaly"] = 1 if abs(zscore) >= threshold else 0

    conn.execute("DELETE FROM fact_analytics")
    cols = ", ".join(("fact_id",) + ANALYTICS_COLUMNS)
    placeholders = ", ".join("?" for _ in range(len(ANALYTICS_COLUMNS) + 1))
    conn.executemany(
        f"INSERT INTO fact_analytics ({cols}) VALUES ({placeholders})",
        [
            (fact_id, *(_round(res[c]) for c in ANALYTICS_COLUMNS))
            for fact_id, res in result.items()
        ],
    )
    conn.commit()
    return len(result)


def apply_kinds_from_catalog(conn) -> int:
    """Применяет виды показателей, уже проставленные в таблице ``metric``.

    Вызывается сборкой базы: без этого шага ранг («индекс») получал бы
    относительные проценты и вердикт «лучше плана», хотя это позиция.
    """
    kinds = {
        r["name"]: r["kind"]
        for r in conn.execute("SELECT name, kind FROM metric WHERE kind IS NOT NULL")
    }
    return apply_metric_kinds(conn, kinds) if kinds else 0


def apply_metric_kinds(conn, kinds: dict[str, str]) -> int:
    """Подавляет относительные проценты у метрик-«вкладов» и «индексов».

    Деление на базу осмысленно только для метрик-«уровней»: у знаковых вкладов
    (центр у нуля) и индексов (ранг/место) проценты дают бред и даже
    инвертируют вердикт динамики. Порт apply_metric_kinds из v5: %-поля
    зануляются, pop_status пересчитывается по знаку абсолютного изменения с
    мёртвой зоной 5 % от максимума |fact| этой метрики.
    """
    if not kinds:
        return 0
    updated = 0
    for name, kind in kinds.items():
        row = conn.execute(
            "SELECT metric_id, direction FROM metric WHERE name_norm = ?",
            (" ".join(str(name).split()).casefold(),),
        ).fetchone()
        if row is None:
            continue
        conn.execute("UPDATE metric SET kind = ? WHERE metric_id = ?", (kind, row["metric_id"]))
        if kind == "уровень":
            continue
        max_fact = conn.execute(
            "SELECT MAX(ABS(fact)) FROM fact WHERE metric_id = ? AND fact IS NOT NULL",
            (row["metric_id"],),
        ).fetchone()[0]
        deadzone = (max_fact or 0.0) * 0.05
        conn.execute(
            "UPDATE fact_analytics SET pop_change_pct = NULL, plan_dev_pct = NULL, "
            "benchmark_dev_pct = NULL, group_change_pct = NULL, rel_change_pct = NULL, "
            "rel_status = NULL WHERE fact_id IN "
            "(SELECT fact_id FROM fact WHERE metric_id = ?)",
            (row["metric_id"],),
        )
        for r in conn.execute(
            "SELECT a.fact_id, a.pop_change_abs FROM fact_analytics a "
            "JOIN fact f ON f.fact_id = a.fact_id WHERE f.metric_id = ?",
            (row["metric_id"],),
        ).fetchall():
            change = r["pop_change_abs"]
            if change is None:
                status = None
            elif abs(change) <= deadzone:
                status = _DYNAMIC_LABELS[1]
            else:
                status = (
                    _DYNAMIC_LABELS[0]
                    if direction_better(change > 0, row["direction"])
                    else _DYNAMIC_LABELS[2]
                )
            conn.execute(
                "UPDATE fact_analytics SET pop_status = ? WHERE fact_id = ?",
                (status, r["fact_id"]),
            )
        updated += 1
    conn.commit()
    return updated


__all__ = [
    "ANALYTICS_COLUMNS",
    "apply_kinds_from_catalog",
    "apply_metric_kinds",
    "compute_analytics",
    "direction_better",
    "has_plan",
]
