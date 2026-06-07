"""Детерминированный (без LLM) расчёт производных метрик.

После загрузки сырых метрик считает отклонения от плана/бенчмарка, динамику
период-к-периоду, тренд, peer-статистику и аномалии — и дописывает их в таблицу
metric_analytics того же in-memory SQLite. Универсально: направление берётся из
metric_type ('прямая'/'обратная'), группы строятся по (metric_name, element, date).
"""
from __future__ import annotations

import os
import statistics
from collections import defaultdict
from typing import Any

from .sqlite_store import SqliteStore

_ANALYTICS_COLUMNS: tuple[str, ...] = (
    "metric_uid",
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
)

_PLAN_LABELS = ("лучше_плана", "в_плане", "хуже_плана")
_BENCH_LABELS = ("лучше_бенчмарка", "на_уровне_бенчмарка", "хуже_бенчмарка")
_DYNAMIC_LABELS = ("улучшение", "стабильно", "ухудшение")
_PEER_LABELS = ("лучше_коллег", "на_уровне_коллег", "хуже_коллег")

_IN_PLAN_TOLERANCE_PCT = 1.0
_TREND_TOLERANCE_PCT = 5.0
# Зона «на уровне коллег» по модулю z-score: ниже — статистически не отличим от
# среднего группы, вердикт «лучше/хуже» вводил бы в заблуждение.
_PEER_NEUTRAL_ZSCORE = 0.5


def _anomaly_threshold() -> float:
    raw = os.environ.get("ANOMALY_ZSCORE_THRESHOLD", "2.0").strip()
    try:
        return float(raw)
    except ValueError:
        return 2.0


def _deviation(
    fact: float | None,
    ref: float | None,
    metric_type: str | None,
    labels: tuple[str, str, str],
) -> tuple[float | None, float | None, str | None]:
    better, equal, worse = labels
    if fact is None or ref is None:
        return None, None, None
    dev_abs = fact - ref
    dev_pct = (dev_abs / ref * 100.0) if ref != 0 else None
    if dev_abs == 0 or (dev_pct is not None and abs(dev_pct) < _IN_PLAN_TOLERANCE_PCT):
        return dev_abs, dev_pct, equal
    higher_is_better = metric_type != "обратная"
    fact_is_higher = dev_abs > 0
    is_better = fact_is_higher == higher_is_better
    return dev_abs, dev_pct, (better if is_better else worse)


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


def _direction_better(value_up: bool, metric_type: str | None) -> bool:
    """value_up — значение РАСТЁТ. Лучше ли это с учётом направления метрики.
    Для 'обратной' (меньше = лучше) рост значения — это ухудшение."""
    higher_is_better = metric_type != "обратная"
    return value_up == higher_is_better


def _trend_status(trend: str | None, metric_type: str | None) -> str | None:
    """Вердикт тренда с учётом направления: 'рост'/'падение' ЗНАЧЕНИЯ →
    'улучшение'/'ухудшение'. Это аналог plan_status для динамики во времени —
    чтобы модель не разворачивала направление по слову 'рост' сама."""
    better, stable, worse = _DYNAMIC_LABELS
    if trend is None:
        return None
    if trend == "стабильно":
        return stable
    return better if _direction_better(trend == "рост", metric_type) else worse


def _pop_status(pop_change_pct: float | None, metric_type: str | None) -> str | None:
    """Вердикт изменения период-к-периоду с учётом направления (та же мёртвая зона
    ±_TREND_TOLERANCE_PCT, что у тренда)."""
    better, stable, worse = _DYNAMIC_LABELS
    if pop_change_pct is None:
        return None
    if abs(pop_change_pct) < _TREND_TOLERANCE_PCT:
        return stable
    return better if _direction_better(pop_change_pct > 0, metric_type) else worse


def _peer_status(zscore: float | None, metric_type: str | None) -> str | None:
    """Вердикт позиции среди коллег с учётом направления: значение выше среднего
    группы (z-score > 0) — это 'лучше_коллег' для 'прямой' и 'хуже_коллег' для
    'обратной' метрики."""
    better, neutral, worse = _PEER_LABELS
    if zscore is None:
        return None
    if abs(zscore) < _PEER_NEUTRAL_ZSCORE:
        return neutral
    return better if _direction_better(zscore > 0, metric_type) else worse


def _round(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


def compute_analytics(store: SqliteStore) -> int:
    conn = store.conn
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT metric_uid, person_tabnum, person_key, person_is_me, metric_name, "
            "metric_type, element, date, fact, plan, benchmark FROM metrics"
        )
    ]

    result: dict[int, dict[str, Any]] = {}

    for r in rows:
        plan_abs, plan_pct, plan_status = _deviation(
            r["fact"], r["plan"], r["metric_type"], _PLAN_LABELS
        )
        # json_analyzer_v2: аналитика по бенчмарку НЕ строится (benchmark_* всегда
        # None → бенчмарк-колонки выпадают из выдачи как пустые).
        result[r["metric_uid"]] = {
            "metric_uid": r["metric_uid"],
            "plan_dev_abs": plan_abs,
            "plan_dev_pct": plan_pct,
            "plan_status": plan_status,
            "benchmark_dev_abs": None,
            "benchmark_dev_pct": None,
            "benchmark_status": None,
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
        }

    series: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        # Ключ серии по person_key (фолбэк на ФИО): иначе руководитель и сотрудник
        # с null табельным схлопнулись бы в один ряд и исказили динамику/тренд.
        series[(r["person_key"], r["metric_name"], r["element"])].append(r)
    for items in series.values():
        items.sort(key=lambda x: x["date"] or "")
        prev: dict | None = None
        for r in items:
            # json_analyzer_v2: pop-сравнение (изменение к прошлому периоду) считаем
            # ТОЛЬКО для метрик с указанным планом — иначе изменение неинформативно
            # (нет цели, относительно которой судить) и часто бессмысленно у знаковых.
            if (
                prev is not None
                and r["fact"] is not None
                and prev["fact"] is not None
                and r["plan"] is not None
            ):
                change_abs = r["fact"] - prev["fact"]
                change_pct = (
                    change_abs / prev["fact"] * 100.0 if prev["fact"] != 0 else None
                )
                result[r["metric_uid"]]["pop_change_abs"] = change_abs
                result[r["metric_uid"]]["pop_change_pct"] = change_pct
                result[r["metric_uid"]]["pop_status"] = _pop_status(
                    change_pct, r["metric_type"]
                )
            prev = r
        trend = _trend([r["fact"] for r in items if r["fact"] is not None])
        for r in items:
            result[r["metric_uid"]]["trend"] = trend
            result[r["metric_uid"]]["trend_status"] = _trend_status(
                trend, r["metric_type"]
            )

    threshold = _anomaly_threshold()
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["person_is_me"]:
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
        higher_is_better = items[0]["metric_type"] != "обратная"
        valued.sort(key=lambda x: x["fact"], reverse=higher_is_better)
        for idx, r in enumerate(valued):
            zscore = (r["fact"] - mean) / std if std > 0 else 0.0
            res = result[r["metric_uid"]]
            res["peer_mean"] = mean
            res["peer_std"] = std
            res["peer_count"] = n
            res["peer_rank"] = idx + 1
            res["peer_percentile"] = round((1 - idx / (n - 1)) * 100, 1)
            res["zscore"] = zscore
            res["peer_status"] = _peer_status(zscore, r["metric_type"])
            res["is_anomaly"] = 1 if abs(zscore) >= threshold else 0

    conn.execute("DELETE FROM metric_analytics")
    cols = ", ".join(_ANALYTICS_COLUMNS)
    placeholders = ", ".join("?" for _ in _ANALYTICS_COLUMNS)
    conn.executemany(
        f"INSERT INTO metric_analytics ({cols}) VALUES ({placeholders})",
        [tuple(_round(res[c]) for c in _ANALYTICS_COLUMNS) for res in result.values()],
    )
    conn.commit()
    return len(result)


def build_summary(store: SqliteStore) -> dict[str, Any]:
    """Детерминированная сводка по датасету.

    Нужна для системного промпта stage-1 (инжектируется в инструкции под видом
    «Состав датасета» через _format_facts) и доступна как tool analytics_summary.
    """
    conn = store.conn
    overview = store.schema_overview()
    dates = overview["dates"]
    latest = dates[-1] if dates else None
    people = overview["people"]
    employees = [p for p in people if not p["person_is_me"]]

    level1 = [
        dict(r)
        for r in conn.execute(
            "SELECT DISTINCT metric_name, metric_type FROM metrics "
            "WHERE depth = 1 ORDER BY metric_name"
        )
    ]

    by_metric: list[dict[str, Any]] = []
    for m in level1:
        row = conn.execute(
            "SELECT COUNT(m.fact) AS n, AVG(m.fact) AS avg_fact, "
            "SUM(CASE WHEN a.plan_status = 'хуже_плана' THEN 1 ELSE 0 END) AS below_plan, "
            "SUM(COALESCE(a.is_anomaly, 0)) AS anomalies "
            "FROM metrics m JOIN metric_analytics a ON a.metric_uid = m.metric_uid "
            "WHERE m.metric_name = ? AND m.date = ? AND m.element IS NULL",
            (m["metric_name"], latest),
        ).fetchone()
        by_metric.append(
            {
                "metric": m["metric_name"],
                "metric_type": m["metric_type"],
                "avg_fact": round(row["avg_fact"], 2)
                if row["avg_fact"] is not None
                else None,
                "below_plan": row["below_plan"],
                "anomalies": row["anomalies"],
            }
        )

    anomalies = [
        dict(r)
        for r in conn.execute(
            "SELECT m.person_fio, m.metric_name, m.element, m.fact, "
            "ROUND(a.zscore, 2) AS zscore "
            "FROM metrics m JOIN metric_analytics a ON a.metric_uid = m.metric_uid "
            "WHERE a.is_anomaly = 1 AND m.date = ? "
            "ORDER BY ABS(a.zscore) DESC LIMIT 5",
            (latest,),
        )
    ]

    # Считаем по trend_status (вердикт с учётом направления), а не по сырому
    # trend: для 'обратных' метрик рост значения = ухудшение, и счётчик по
    # 'рост/падение' вводил бы в заблуждение.
    trends = {"улучшение": 0, "ухудшение": 0, "стабильно": 0}
    for r in conn.execute(
        "SELECT a.trend_status AS trend_status, "
        "COUNT(DISTINCT m.person_key || '|' || m.metric_name) AS c "
        "FROM metrics m JOIN metric_analytics a ON a.metric_uid = m.metric_uid "
        "WHERE m.depth = 1 AND m.element IS NULL AND a.trend_status IS NOT NULL "
        "GROUP BY a.trend_status"
    ):
        if r["trend_status"] in trends:
            trends[r["trend_status"]] = r["c"]

    return {
        "scope": {
            "people": len(people),
            "employees": len(employees),
            "metric_types": len(overview["metrics"]),
            "elements": len(overview["elements"]),
            "dates": dates,
            "metric_rows": overview["total_metric_rows"],
        },
        "latest_date": latest,
        "by_metric_latest": by_metric,
        "top_anomalies_latest": anomalies,
        "trend_counts_level1": trends,
    }


# --------------------------------------------------------------------------- #
# Универсальный обзор ситуации: зоны (проблемы/позитив/стабильность) + причинная
# цепочка драйверов произвольной глубины. Детерминированный (без LLM): зоны и
# направление берутся из ГОТОВЫХ вердиктов metric_analytics (plan_status и т.п.),
# а причинная ВЕРТИКАЛЬ строится по бизнес-весу influent_percent: на каждом узле
# доминирующий драйвер = ребёнок с наибольшим influent (|сигнал| — лишь тай-брейк
# при равных весах; если влияния нет ни у кого — ранжируем по величине изменения).
# Идея веса влияния — из json_analyzer_causal/causal.py:_algebraic_attribution, но
# без перемножения, без пересчёта направлений и без DoWhy. Рекурсия идёт ТОЛЬКО в
# доминирующий структурный драйвер, поэтому работает на любой глубине и числе метрик.
# --------------------------------------------------------------------------- #
_OVERVIEW_MAX_DEPTH = 6          # предохранитель от циклов/слишком глубоких деревьев
_OVERVIEW_TOP_DRIVERS = 3        # сколько драйверов показывать на узел
_OVERVIEW_TOP_SEGMENTS = 3       # сколько разрезов (продуктов) показывать на узел
_OVERVIEW_MIN_SHARE = 15.0       # доля доминирующего драйвера, ниже — рекурсию не продолжаем
_OVERVIEW_MAX_PROBLEM_CHAINS = 3 # для скольких проблем разворачивать полную цепочку
_OVERVIEW_MAX_HEADLINES = 8      # сколько позитивных/стабильных метрик перечислять


def _focus_person(store: SqliteStore, person: Any | None) -> tuple[Any, Any]:
    """Резолв фокусного человека → (person_key, fio).

    person задан → он (ФИО-подстрока или табельный); иначе единственный сотрудник
    (person_is_me=0); иначе руководитель; иначе любой. (None, None) если людей нет.
    Идентичность — по person_key (фолбэк на ФИО при null табельном), поэтому
    работает и когда tabnum в данных не пришёл.
    """
    if person is not None and str(person).strip():
        text = str(person).strip()
        if text.isdigit():
            row = store.conn.execute(
                "SELECT person_key, person_fio FROM metrics "
                "WHERE person_tabnum = ? OR person_key = ? LIMIT 1",
                (int(text), text),
            ).fetchone()
        else:
            row = store.conn.execute(
                "SELECT person_key, person_fio FROM metrics "
                "WHERE person_fio LIKE ? LIMIT 1",
                (f"%{text}%",),
            ).fetchone()
        return (row["person_key"], row["person_fio"]) if row else (None, None)

    emps = [
        dict(r)
        for r in store.conn.execute(
            "SELECT DISTINCT person_key, person_fio FROM metrics "
            "WHERE person_is_me = 0"
        )
    ]
    if len(emps) == 1:
        return emps[0]["person_key"], emps[0]["person_fio"]
    boss = store.conn.execute(
        "SELECT person_key, person_fio FROM metrics "
        "WHERE person_is_me = 1 LIMIT 1"
    ).fetchone()
    if boss:
        return boss["person_key"], boss["person_fio"]
    if emps:
        return emps[0]["person_key"], emps[0]["person_fio"]
    return None, None


def _overview_dates(
    store: SqliteStore, date: str | None
) -> tuple[str | None, str | None]:
    """(текущая, предыдущая) даты: текущая = заданная или последняя; предыдущая —
    ближайшая до текущей (для подписи сравнения; сам Δ уже в pop_change_pct)."""
    dates = [
        r["date"]
        for r in store.conn.execute(
            "SELECT DISTINCT date FROM metrics WHERE date IS NOT NULL ORDER BY date"
        )
    ]
    if not dates:
        return None, None
    cur = date or dates[-1]
    before = [d for d in dates if d < cur]
    return cur, (before[-1] if before else None)


def _zone_of(row: dict[str, Any]) -> str:
    """Зона метрики по готовым вердиктам (приоритет: проблема > позитив > стабильно).

    Метрика «хуже плана, но с улучшающейся динамикой» остаётся проблемой —
    приоритет у негатива, чтобы зона не маскировала отставание от плана."""
    plan, trend, pop = (
        row.get("plan_status"),
        row.get("trend_status"),
        row.get("pop_status"),
    )
    if plan == "хуже_плана" or trend == "ухудшение" or pop == "ухудшение":
        return "problem"
    if plan == "лучше_плана" or trend == "улучшение" or pop == "улучшение":
        return "positive"
    return "stable"


def _overview_signal(row: dict[str, Any]) -> float:
    """Магнитуда «движения/отставания» узла: |изменение к прошлому периоду|, иначе
    |отклонение от плана|, иначе 0. Используется и для ранжирования драйверов, и
    для сортировки проблем по силе.

    Нулевое значение трактуем как «сигнала нет» и проваливаемся к следующему полю:
    при pop_change_pct=0.0 (ровно ноль) метрика всё равно ранжируется по отклонению
    от плана, а не получает искусственный 0."""
    for key in ("pop_change_pct", "plan_dev_pct"):
        val = row.get(key)
        if val is not None and val != 0:
            return abs(val)
    return 0.0


def _driver_view(row: dict[str, Any], share_pct: float | None) -> dict[str, Any]:
    """Публичная проекция узла-драйвера (без служебных полей). share_pct=None для
    «больше всего изменилось» — там доля влияния не показывается."""
    return {
        "metric": row.get("metric_name"),
        "influent_percent": row.get("influent_percent"),
        "share_pct": share_pct,
        "plan_status": row.get("plan_status"),
        "plan_dev_pct": row.get("plan_dev_pct"),
        "trend_status": row.get("trend_status"),
        "pop_status": row.get("pop_status"),
        "pop_change_pct": row.get("pop_change_pct"),
        "pop_change_abs": row.get("pop_change_abs"),
        "fact": row.get("fact"),
        "measure_type": row.get("measure_type"),
    }


def _rank_drivers(
    structural: list[dict[str, Any]],
) -> tuple[list[tuple[float, dict[str, Any]]], bool]:
    """Ранжирует структурных детей по бизнес-весу influent_percent.

    Бизнес-вес — основной сигнал «что важно» для метрики (напр. AHT = 90% от
    Производительности): по нему строим причинную ВЕРТИКАЛЬ и спускаемся вглубь.
    Магнитуда движения/отставания идёт лишь тай-брейком при равных весах (а
    «больше всего изменилось» выносится отдельно в _build_node). Сиблинги БЕЗ
    influent исключаются, когда у кого-то он есть (так отсекаются РАНГ-метрики и
    прочие некомпонентные узлы). Если влияния нет ни у кого — ранжируем по величине
    изменения (эвристика, heuristic_equal=True).

    Возвращает [(share_pct, row), ...] по убыванию и флаг heuristic_equal.
    share_pct = доля бизнес-веса (при наличии весов) либо доля |изменения| (без них).
    """
    has_infl = any(k.get("influent_percent") is not None for k in structural)
    if has_infl:
        pool = [k for k in structural if k.get("influent_percent") is not None]
        wsum = sum(k["influent_percent"] for k in pool) or 1.0
        pool = sorted(
            pool,
            key=lambda k: (k["influent_percent"], _overview_signal(k)),
            reverse=True,
        )
        ranked = [(round(k["influent_percent"] / wsum * 100.0, 1), k) for k in pool]
        return ranked, False

    ssum = sum(_overview_signal(k) for k in structural) or 1.0
    pool = sorted(structural, key=_overview_signal, reverse=True)
    ranked = [(round(_overview_signal(k) / ssum * 100.0, 1), k) for k in pool]
    return ranked, True


def _top_segments(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Худшие/лучшие разрезы (продуктов/тематик) узла. Разрезы группируются по имени
    дочерней метрики (напр. «Влияние тематик на Talk&Hold»); берём группу с
    наибольшим суммарным |сигналом|, а ВНУТРИ неё ранжируем по факту с учётом
    направления (а не по изменению) — чтобы всплывал худший/лучший по ЗНАЧЕНИЮ."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in segments:
        groups[s.get("metric_name")].append(s)
    label, items = max(
        groups.items(),
        key=lambda kv: sum(_overview_signal(i) for i in kv[1]),
    )
    valued = [i for i in items if i.get("fact") is not None]
    if not valued:
        return {"label": label, "worst": [], "best": []}
    higher_is_better = valued[0].get("metric_type") != "обратная"
    valued.sort(key=lambda i: i["fact"], reverse=higher_is_better)  # лучшее первым

    def _seg(i: dict[str, Any]) -> dict[str, Any]:
        return {
            "element": i.get("element"),
            "fact": i.get("fact"),
            "measure_type": i.get("measure_type"),
            "pop_change_pct": i.get("pop_change_pct"),
            "pop_change_abs": i.get("pop_change_abs"),
        }

    worst = [_seg(i) for i in reversed(valued[-_OVERVIEW_TOP_SEGMENTS:])]
    best = [_seg(i) for i in valued[:_OVERVIEW_TOP_SEGMENTS]]
    return {"label": label, "worst": worst, "best": best}


def _node_header(row: dict[str, Any]) -> dict[str, Any]:
    """Шапка метрики-узла (без декомпозиции) — общая для цепочек и заголовков."""
    return {
        "metric": row.get("metric_name"),
        "element": row.get("element"),
        "metric_type": row.get("metric_type"),
        "measure_type": row.get("measure_type"),
        "fact": row.get("fact"),
        "plan_status": row.get("plan_status"),
        "plan_dev_pct": row.get("plan_dev_pct"),
        "plan_dev_abs": row.get("plan_dev_abs"),
        "trend_status": row.get("trend_status"),
        "pop_status": row.get("pop_status"),
        "pop_change_pct": row.get("pop_change_pct"),
        "pop_change_abs": row.get("pop_change_abs"),
    }


def _build_node(
    children: dict[Any, list[dict[str, Any]]],
    row: dict[str, Any],
    depth: int,
) -> dict[str, Any]:
    """Рекурсивный узел причинной цепочки: шапка + ранжированные драйверы +
    рекурсия в доминирующий + разрезы (продукты). Глубина ограничена."""
    node = _node_header(row)
    node.update(
        {
            "drivers": [],
            "main_driver": None,
            "by_segments": None,
            "biggest_mover": None,
            "note": None,
        }
    )

    kids = children.get(row.get("metric_uid"), [])
    structural = [k for k in kids if k.get("element") is None]
    segments = [k for k in kids if k.get("element") is not None]
    if segments:
        node["by_segments"] = _top_segments(segments)
    if not structural:
        if not kids:
            node["note"] = "не раскладывается (нет компонентов)"
        return node

    ranked, heuristic_equal = _rank_drivers(structural)
    node["drivers"] = [
        _driver_view(k, share) for share, k in ranked[:_OVERVIEW_TOP_DRIVERS]
    ]

    notes: list[str] = []
    if heuristic_equal:
        notes.append("влияние детей не задано — ранжировано по величине изменения (эвристика)")
    # «Больше всего изменилось» — отдельный сигнал, когда сильнее всех двигался НЕ
    # самый весомый компонент (вертикаль идёт по весу, а это — про динамику).
    # Берём только из ранжированного пула (исключая некомпонентные узлы вроде РАНГ).
    ranked_rows = [k for _, k in ranked]
    dom_row = ranked_rows[0]
    mover = max(ranked_rows, key=_overview_signal)
    if mover is not dom_row and _overview_signal(mover) > 0:
        node["biggest_mover"] = _driver_view(mover, None)
    # Честная пометка: если родитель — проблема, а ни один из показанных ключевых
    # компонентов сам не «плохой» (все в плане/улучшаются), дерево не объясняет
    # отставание — причина может быть вне него (аналог self-mechanism в каузале).
    shown = [k for _, k in ranked[:_OVERVIEW_TOP_DRIVERS]]
    if _zone_of(row) == "problem" and not any(_zone_of(k) == "problem" for k in shown):
        notes.append(
            "ключевые компоненты не ухудшались — отставание не объясняется деревом "
            "(причина может быть вне него)"
        )
    node["note"] = "; ".join(notes) or None

    top_share, top_row = ranked[0]
    if (
        depth < _OVERVIEW_MAX_DEPTH
        and top_share >= _OVERVIEW_MIN_SHARE
        and children.get(top_row.get("metric_uid"))
    ):
        node["main_driver"] = _build_node(children, top_row, depth + 1)
    return node


def build_situation_overview(
    store: SqliteStore,
    person: Any | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    """Карта ситуации фокусного человека за один проход: зоны (проблемы / позитив /
    стабильность) по корневым метрикам и причинная цепочка драйверов в каждой
    проблеме (произвольной глубины, до листьев). Универсально: любое число корней,
    любая глубина, корректная деградация при одном уровне / без influent_percent.
    """
    pkey, fio = _focus_person(store, person)
    if pkey is None:
        return {"error": "В датасете нет людей."}
    cur_date, prev_date = _overview_dates(store, date)
    if cur_date is None:
        return {"error": "В датасете нет дат."}

    rows = [
        dict(r)
        for r in store.conn.execute(
            "SELECT m.metric_uid, m.parent_uid, m.depth, m.metric_name, "
            "m.metric_type, m.measure_type, m.element, m.fact, m.influent_percent, "
            "a.plan_status, a.plan_dev_pct, a.plan_dev_abs, a.benchmark_status, "
            "a.trend_status, a.pop_status, a.pop_change_pct, a.pop_change_abs "
            "FROM metrics m LEFT JOIN metric_analytics a "
            "ON a.metric_uid = m.metric_uid "
            "WHERE m.person_key = ? AND m.date = ?",
            (pkey, cur_date),
        )
    ]
    if not rows:
        return {
            "person_fio": fio,
            "person_key": pkey,
            "date": cur_date,
            "prev_date": prev_date,
            "single_level": True,
            "problems": [],
            "positives": [],
            "stable": [],
            "note": "Нет данных по сотруднику на этот период.",
        }

    # Зоны и вердикты читаются из metric_analytics; без неё все статусы NULL и любая
    # метрика молча попала бы в «стабильно». Честно сообщаем, что аналитика не
    # посчитана, вместо ложной классификации (в проде compute_analytics всегда есть).
    if store.analytics_row_count() == 0:
        return {
            "person_fio": fio,
            "person_key": pkey,
            "date": cur_date,
            "prev_date": prev_date,
            "single_level": True,
            "problems": [],
            "positives": [],
            "stable": [],
            "note": "Аналитика не посчитана (compute_analytics) — зоны недоступны.",
        }

    children: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("parent_uid") is not None:
            children[r["parent_uid"]].append(r)
    roots = [r for r in rows if r.get("depth") == 1 and r.get("element") is None]
    if not roots:  # датасет только из разрезов (agg-) — корнями станут они
        roots = [r for r in rows if r.get("depth") == 1]

    buckets: dict[str, list[dict[str, Any]]] = {
        "problem": [],
        "positive": [],
        "stable": [],
    }
    for r in roots:
        buckets[_zone_of(r)].append(r)
    buckets["problem"].sort(key=_overview_signal, reverse=True)

    problems: list[dict[str, Any]] = []
    for i, r in enumerate(buckets["problem"]):
        if i < _OVERVIEW_MAX_PROBLEM_CHAINS:
            problems.append(_build_node(children, r, depth=1))
        else:
            problems.append(_node_header(r))

    positives = [_node_header(r) for r in buckets["positive"][:_OVERVIEW_MAX_HEADLINES]]
    stable = [_node_header(r) for r in buckets["stable"][:_OVERVIEW_MAX_HEADLINES]]
    single_level = not any(children.get(r.get("metric_uid")) for r in roots)

    return {
        "person_fio": fio,
        "person_key": pkey,
        "date": cur_date,
        "prev_date": prev_date,
        "single_level": single_level,
        "problems": problems,
        "positives": positives,
        "stable": stable,
    }


# --------------------------------------------------------------------------- #
# Лучшие/худшие разрезы (element) метрики, сравнённые МЕЖДУ СОБОЙ по факту с учётом
# направления (metric_type). Нужно для метрик без плана (план/бенчмарк не задают
# «хорошо/плохо», а сравнение сотрудника с коллегами недоступно при одном человеке):
# тогда единственный осмысленный ориентир — какой разрез лучше/хуже ОСТАЛЬНЫХ
# разрезов той же метрики у этого сотрудника.
# --------------------------------------------------------------------------- #
_RANK_ELEMENTS_TOP = 5


def rank_elements(
    store: SqliteStore,
    metric: str,
    person: Any | None = None,
    date: str | None = None,
    top: int = _RANK_ELEMENTS_TOP,
) -> dict[str, Any]:
    """Ранжирует element-строки метрики одного человека за период по факту с учётом
    направления (прямая: выше=лучше, обратная: ниже=лучше). Возвращает ranked-список
    (лучшее→худшее). План/бенчмарк не используются."""
    mt = store.metric_type_of(metric)
    if mt is None:
        return {"error": f"Метрика '{metric}' не найдена."}
    pkey, fio = _focus_person(store, person)
    if pkey is None:
        return {"error": "В датасете нет людей."}
    cur_date, _prev = _overview_dates(store, date)
    if cur_date is None:
        return {"error": "В датасете нет дат."}

    rows = [
        dict(r)
        for r in store.conn.execute(
            "SELECT m.element, m.fact, m.measure_type, a.plan_status, "
            "a.pop_change_pct, a.pop_change_abs "
            "FROM metrics m LEFT JOIN metric_analytics a "
            "ON a.metric_uid = m.metric_uid "
            "WHERE m.metric_name = ? AND m.element IS NOT NULL "
            "AND m.person_key = ? AND m.date = ? AND m.fact IS NOT NULL",
            (metric, pkey, cur_date),
        )
    ]
    if not rows:
        return {
            "metric": metric,
            "metric_type": mt,
            "date": cur_date,
            "person_fio": fio,
            "count": 0,
            "best": [],
            "worst": [],
            "top": top,
            "note": "у метрики нет разрезов (element) с данными на этот период.",
        }

    higher_is_better = mt != "обратная"
    rows.sort(key=lambda r: r["fact"], reverse=higher_is_better)  # лучшее первым

    def _view(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "element": r["element"],
            "fact": r["fact"],
            "measure_type": r.get("measure_type"),
            "pop_change_pct": r.get("pop_change_pct"),
            "pop_change_abs": r.get("pop_change_abs"),
            "plan_status": r.get("plan_status"),
        }

    best = [_view(r) for r in rows[:top]]              # лучшие (голова)
    worst = [_view(r) for r in reversed(rows[-top:])]  # худшие (хвост), худший первым
    return {
        "metric": metric,
        "metric_type": mt,
        "measure_type": rows[0].get("measure_type"),
        "date": cur_date,
        "person_fio": fio,
        "count": len(rows),
        "best": best,
        "worst": worst,
        "top": top,
    }


# --------------------------------------------------------------------------- #
# Применение классификации видов метрик (metric_kinds_cache): для не-«уровней»
# (вклад/индекс) относительное %-сравнение бессмысленно — обнуляем относительные %
# и пересчитываем вердикт динамики из ЗНАКА абсолютного изменения (иначе для
# знаковых метрик pop_status, выведенный из знака pop_change_pct, инвертируется).
# --------------------------------------------------------------------------- #
def _pop_status_from_abs(
    pop_change_abs: float | None,
    metric_type: str | None,
    deadzone: float = 0.0,
) -> str | None:
    """Вердикт динамики из АБСОЛЮТНОГО изменения (для знаковых/индексных метрик,
    где относительный % недоступен). Мёртвая зона задаётся в абсолюте (deadzone):
    |изменение| ≤ deadzone → 'стабильно', иначе по знаку с учётом направления.
    deadzone калибруется по масштабу метрики (аналог относительной мёртвой зоны)."""
    better, stable, worse = _DYNAMIC_LABELS
    if pop_change_abs is None:
        return None
    if abs(pop_change_abs) <= deadzone:
        return stable
    return better if _direction_better(pop_change_abs > 0, metric_type) else worse


def apply_metric_kinds(store: SqliteStore, kinds: dict[str, str]) -> int:
    """Применяет виды метрик к посчитанной аналитике. Для метрик вида ≠ 'уровень'
    обнуляет относительные %-поля (pop_change_pct/plan_dev_pct/benchmark_dev_pct) и
    пересчитывает pop_status из знака pop_change_abs (мёртвая зона — 5% от масштаба
    метрики max|fact|, аналог относительной зоны). plan_status/benchmark_status уже
    основаны на знаке абс. отклонения и остаются как есть (в выводе показывается
    plan_dev_abs). Виды кладёт в store.metric_kinds. Пустой ввод — no-op. Возвращает
    число затронутых строк."""
    store.metric_kinds = dict(kinds or {})
    non_level = [m for m, k in (kinds or {}).items() if k and k != "уровень"]
    if not non_level:
        return 0
    affected = 0
    for name in non_level:
        rows = store.conn.execute(
            "SELECT a.metric_uid AS uid, a.pop_change_abs AS abs, m.metric_type AS mt, "
            "m.fact AS fact FROM metric_analytics a "
            "JOIN metrics m ON m.metric_uid = a.metric_uid WHERE m.metric_name = ?",
            (name,),
        ).fetchall()
        facts = [abs(r["fact"]) for r in rows if r["fact"] is not None]
        deadzone = (max(facts) * _TREND_TOLERANCE_PCT / 100.0) if facts else 0.0
        for r in rows:
            store.conn.execute(
                "UPDATE metric_analytics SET pop_change_pct = NULL, "
                "plan_dev_pct = NULL, benchmark_dev_pct = NULL, pop_status = ? "
                "WHERE metric_uid = ?",
                (_pop_status_from_abs(r["abs"], r["mt"], deadzone), r["uid"]),
            )
            affected += 1
    store.conn.commit()
    return affected
