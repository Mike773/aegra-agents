"""Сборка нормализованной in-memory SQLite на один запуск агента.

`build_run_db` синхронна: узел графа зовёт её через ``asyncio.to_thread``.
Соединение живёт в замыкании узла и его инструментов — никаких модульных
глобалов, поэтому параллельные запуски графа не мешают друг другу.

Дерево метрик строится ТОЛЬКО из структуры JSON: даты узлов на каталог не
влияют. «Последний период» считается на серию (персона × метрика × разрез) —
см. ``fact.is_last_of_series`` и вью ``v_fact_latest``.

Дерево у каждого человека своё (``metric_edge``/``metric_tree`` с person_id):
у руководителя и сотрудников состав и вложенность показателей могут
отличаться. На каталоге ``metric`` лежит сводное дерево по всем людям — для
каталога в промпте и резолва имён.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ...shared.peer_levels import level_order
from ...shared.text_similarity import similarity_ratio
from . import analytics as analytics_mod
from . import loader
from .loader import MetricRec, ParsedDataset
from .schema_doc import build_schema_doc as schema_doc
from .schema_doc import sql_examples

_SCHEMA_PATH = Path(__file__).with_name("sql") / "schema.sql"
# Порог нечёткого совпадения имени метрики/персоны (лестница резолва).
_FUZZY_THRESHOLD = 0.8


@dataclass
class BuildReport:
    """Что случилось при сборке: пригодно и для дебага, и для блока «пробелы»."""

    skipped_empty_leaves: int = 0
    description_conflicts: int = 0
    duplicate_facts: int = 0
    unmatched_aggregates: int = 0
    warnings: list[str] = field(default_factory=list)


def _ru_lower(value: Any) -> Any:
    """casefold для кириллицы: встроенный LOWER() в SQLite только ASCII."""
    if value is None:
        return None
    return str(value).casefold()


def _pct(a: Any, b: Any) -> Any:
    """(a - b) / b * 100; None при отсутствии базы или нулевой базе."""
    if a is None or b in (None, 0):
        return None
    try:
        return (float(a) - float(b)) / float(b) * 100.0
    except (TypeError, ValueError):
        return None


def _fmt_num(value: Any) -> Any:
    """Число для текста: до двух знаков, без хвостовых нулей (1.0 → '1')."""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{num:.2f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.create_function("ru_lower", 1, _ru_lower, deterministic=True)
    conn.create_function("pct", 2, _pct, deterministic=True)
    conn.create_function("fmt_num", 1, _fmt_num, deterministic=True)
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def _dominant_parents(
    edges: dict[tuple[str, str], float | None],
) -> dict[str, tuple[str, float | None]]:
    """Ребёнок → (доминирующий родитель, вес). Доминирующий = максимальный
    influent_percent; при равенстве/отсутствии весов — первый по порядку JSON."""
    best: dict[str, tuple[str, float | None]] = {}
    for (parent, child), infl in edges.items():
        cur = best.get(child)
        if cur is None:
            best[child] = (parent, infl)
            continue
        cur_infl = cur[1]
        if infl is not None and (cur_infl is None or infl > cur_infl):
            best[child] = (parent, infl)
    return best


def _build_hierarchy(
    metrics: dict[str, MetricRec],
    edges: dict[tuple[str, str], float | None],
    scope: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """depth / path / root по доминирующим родителям. Циклы разрываются:
    узел, уже встреченный на пути, считается корнем.

    ``scope`` — какие метрики считать (дерево одного человека); None = все."""
    dominant = _dominant_parents(edges)
    out: dict[str, dict[str, Any]] = {}
    names = scope if scope is not None else set(metrics)

    def resolve(norm: str, seen: frozenset[str]) -> dict[str, Any]:
        cached = out.get(norm)
        if cached is not None:
            return cached
        rec = metrics[norm]
        parent = dominant.get(norm)
        if parent is None or parent[0] not in names or parent[0] in seen:
            info = {
                "depth": 1,
                "path": rec.name,
                "root_name": rec.name,
                "parent_name": None,
                "influent_percent": None,
            }
        else:
            pinfo = resolve(parent[0], seen | {norm})
            info = {
                "depth": pinfo["depth"] + 1,
                "path": f"{pinfo['path']} > {rec.name}",
                "root_name": pinfo["root_name"],
                "parent_name": metrics[parent[0]].name,
                "influent_percent": parent[1],
            }
        out[norm] = info
        return info

    for norm in names:
        if norm in metrics:
            resolve(norm, frozenset())
    return out


def _union_edges(
    edges: dict[tuple[str, str, str], float | None],
) -> dict[tuple[str, str], float | None]:
    """Рёбра всех людей → сводные (родитель, ребёнок): первый вес, известный
    вес предпочтительнее None. Нужны только для сводного дерева каталога."""
    out: dict[tuple[str, str], float | None] = {}
    for (_, parent, child), infl in edges.items():
        key = (parent, child)
        if key not in out or (out[key] is None and infl is not None):
            out[key] = infl
    return out


def _apply_guessed_kinds(conn: sqlite3.Connection, parsed: ParsedDataset) -> None:
    """Вид показателя по названию — без него ранг получал бы проценты и вердикт
    «лучше плана», хотя это позиция, а не уровень."""
    for norm, rec in parsed.metrics.items():
        conn.execute(
            "UPDATE metric SET kind = ? WHERE name_norm = ?",
            (loader.guess_kind(rec.name, rec.description), norm),
        )


def _insert_dataset(conn: sqlite3.Connection, parsed: ParsedDataset, report: BuildReport) -> None:
    person_ids: dict[str, int] = {}
    for i, p in enumerate(parsed.people, start=1):
        conn.execute(
            "INSERT INTO person (person_id, person_key, tabnum, fio, post, depart, is_me) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (i, p.person_key, p.tabnum, p.fio, p.post, p.depart, 1 if p.is_me else 0),
        )
        person_ids[p.person_key] = i

    # Сводное дерево каталога — по объединённым рёбрам всех людей.
    hierarchy = _build_hierarchy(parsed.metrics, _union_edges(parsed.edges))
    metric_ids: dict[str, int] = {}
    for i, (norm, rec) in enumerate(parsed.metrics.items(), start=1):
        h = hierarchy[norm]
        conn.execute(
            "INSERT INTO metric (metric_id, metric_key, name_key, ext_id, name, name_norm, "
            "description, direction, unit, calc_period, is_star, is_star_metric, star_of, "
            "depth, root_name, path, parent_name, influent_percent) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                i,
                loader.metric_key(rec.name, rec.description),
                loader.name_key(rec.name),
                rec.ext_id,
                rec.name,
                norm,
                rec.description,
                rec.direction,
                rec.unit,
                rec.calc_period,
                1 if rec.is_star else 0,
                1 if rec.is_star_metric else 0,
                rec.star_of,
                h["depth"],
                h["root_name"],
                h["path"],
                h["parent_name"],
                h["influent_percent"],
            ),
        )
        metric_ids[norm] = i

    # Рёбра и позиция в дереве — на каждого человека отдельно.
    for (person_key, parent, child), infl in parsed.edges.items():
        if person_key in person_ids and parent in metric_ids and child in metric_ids:
            conn.execute(
                "INSERT OR IGNORE INTO metric_edge "
                "(person_id, parent_id, child_id, influent_percent) VALUES (?, ?, ?, ?)",
                (person_ids[person_key], metric_ids[parent], metric_ids[child], infl),
            )
    for person_key, owned in parsed.person_metrics.items():
        if person_key not in person_ids:
            continue
        own_edges = {
            (parent, child): infl
            for (pk, parent, child), infl in parsed.edges.items()
            if pk == person_key
        }
        own_tree = _build_hierarchy(parsed.metrics, own_edges, scope=set(owned))
        conn.executemany(
            "INSERT OR IGNORE INTO metric_tree "
            "(person_id, metric_id, depth, root_name, path, parent_name, influent_percent) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    person_ids[person_key],
                    metric_ids[norm],
                    h["depth"],
                    h["root_name"],
                    h["path"],
                    h["parent_name"],
                    h["influent_percent"],
                )
                for norm, h in own_tree.items()
                if norm in metric_ids
            ],
        )

    dates = sorted({f.date for f in parsed.facts if f.date})
    period_ids: dict[str, int] = {}
    for idx, date in enumerate(dates, start=1):
        conn.execute(
            "INSERT INTO period (period_id, date, period_idx, ym, is_latest) "
            "VALUES (?, ?, ?, ?, ?)",
            (idx, date, idx, date[:7] if len(date) >= 7 else None, 1 if idx == len(dates) else 0),
        )
        period_ids[date] = idx

    fact_id = 0
    rank_rows: list[tuple] = []
    for f in parsed.facts:
        if not f.date or f.date not in period_ids or f.person_key not in person_ids:
            report.warnings.append(f"факт без даты пропущен: {f.metric_norm}")
            continue
        fact_id += 1
        cur = conn.execute(
            "INSERT OR IGNORE INTO fact (fact_id, person_id, metric_id, period_id, element, "
            "fact, plan, benchmark, ex, rr, star_received, calc_period, src_ord) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fact_id,
                person_ids[f.person_key],
                metric_ids[f.metric_norm],
                period_ids[f.date],
                f.element,
                f.fact,
                f.plan,
                f.benchmark,
                f.ex,
                f.rr,
                f.star_received,
                f.calc_period,
                f.src_ord,
            ),
        )
        if cur.rowcount == 0:
            report.duplicate_facts += 1
            fact_id -= 1
            continue
        for r in f.rankings or []:
            if not isinstance(r, dict) or r.get("level") is None:
                continue
            pos, total = loader.parse_rank(r.get("rank"))
            rank_rows.append(
                (fact_id, str(r["level"]), pos, total, r.get("rank"), r.get("percentile"))
            )
    if rank_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO ranking (fact_id, level, rank_pos, rank_total, rank_raw, "
            "percentile) VALUES (?, ?, ?, ?, ?, ?)",
            rank_rows,
        )

    # Последняя дата КАЖДОЙ серии (персона × метрика × разрез) — не глобальная.
    conn.execute(
        "UPDATE fact SET is_last_of_series = 1 WHERE fact_id IN ("
        "  SELECT f.fact_id FROM fact f"
        "  WHERE f.period_id = (SELECT MAX(f2.period_id) FROM fact f2"
        "     WHERE f2.person_id = f.person_id AND f2.metric_id = f.metric_id"
        "       AND COALESCE(f2.element, '') = COALESCE(f.element, '')))"
    )

    for person_key, agg_id in parsed.person_peer:
        if person_key in person_ids:
            conn.execute(
                "INSERT OR IGNORE INTO person_peer (person_id, aggregate_id) VALUES (?, ?)",
                (person_ids[person_key], agg_id),
            )


def _level_orders(agg_rows: list[dict[str, Any]]) -> dict[str, int]:
    """Уровни от узкой группы к широкой (порт analytics._level_order v5):
    порядок из конфига PEER_LEVELS → минимальный total_objects → порядок в payload."""
    info: dict[str, dict[str, Any]] = {}
    for r in agg_rows:
        level = r.get("level")
        if level is None:
            continue
        d = info.setdefault(str(level), {"first": r.get("node_uid") or 0, "objs": None})
        d["first"] = min(d["first"], r.get("node_uid") or 0)
        objs = r.get("total_objects")
        if r.get("is_current") and objs is not None and objs > 0:
            d["objs"] = objs if d["objs"] is None else min(d["objs"], objs)
    configured = {code.strip().casefold(): i for i, code in enumerate(level_order())}
    ordered = sorted(
        info,
        key=lambda lv: (
            configured.get(lv.strip().casefold(), len(configured)),
            info[lv]["objs"] is None,
            info[lv]["objs"] if info[lv]["objs"] is not None else 0,
            info[lv]["first"],
        ),
    )
    return {lv: i + 1 for i, lv in enumerate(ordered)}


def _insert_aggregates(
    conn: sqlite3.Connection, agg_rows: list[dict[str, Any]], report: BuildReport
) -> None:
    if not agg_rows:
        return
    orders = _level_orders(agg_rows)
    metric_by_norm = {
        row["name_norm"]: row["metric_id"]
        for row in conn.execute("SELECT metric_id, name_norm FROM metric")
    }
    for i, r in enumerate(agg_rows, start=1):
        if r.get("level") is None:
            continue
        metric_id = metric_by_norm.get(loader.norm_text(r.get("metric_name")))
        if metric_id is None:
            report.unmatched_aggregates += 1
        conn.execute(
            "INSERT INTO peer_aggregate (agg_id, aggregate_id, level, level_name, level_order, "
            "metric_id, metric_name, metric_ext_id, node_uid, parent_node_uid, dt, ym, "
            "calc_period, is_current, mean_fact, mean_plan, mean_ex, median, hit_rate, "
            "top20_mean_fact, iqr, cv, total_objects) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                i,
                r.get("aggregate_id"),
                str(r["level"]),
                r.get("level_name"),
                orders.get(str(r["level"])),
                metric_id,
                r.get("metric_name") or "",
                r.get("metric_id"),
                r.get("node_uid"),
                r.get("parent_node_uid"),
                r.get("dt"),
                r.get("ym"),
                r.get("calc_period"),
                int(r.get("is_current") or 0),
                r.get("mean_fact"),
                r.get("mean_plan"),
                r.get("mean_ex"),
                r.get("median"),
                r.get("hit_rate"),
                r.get("top20_mean_fact"),
                r.get("iqr"),
                r.get("cv"),
                r.get("total_objects"),
            ),
        )


def _insert_ratings(conn: sqlite3.Connection, parsed: ParsedDataset) -> None:
    """Рейтинг по звёздам — только когда в базе есть звёзды: без них поле
    ``rating`` не имеет смысла и игнорируется. Названия и порядок уровней
    берутся из уже загруженных агрегатов по коду уровня; уровней, которых в
    агрегатах нет, — код вместо названия и порядок по размеру группы (staff)."""
    if not parsed.ratings:
        return
    has_stars = conn.execute(
        "SELECT COUNT(*) FROM fact WHERE star_received IS NOT NULL"
    ).fetchone()[0]
    if not has_stars:
        return
    person_ids = {
        row["person_key"]: row["person_id"]
        for row in conn.execute("SELECT person_id, person_key FROM person")
    }
    known: dict[str, tuple[str | None, int | None]] = {}
    for row in conn.execute(
        "SELECT level, MIN(level_name) AS level_name, MIN(level_order) AS level_order "
        "FROM peer_aggregate GROUP BY level"
    ):
        known[str(row["level"]).strip().casefold()] = (row["level_name"], row["level_order"])
    # Уровни без агрегатов — после известных, от меньшей группы к большей.
    unknown_sizes: dict[str, int] = {}
    for r in parsed.ratings:
        if r.level.casefold() not in known:
            size = r.staff if r.staff is not None else 0
            unknown_sizes[r.level] = max(unknown_sizes.get(r.level, 0), size)
    base = max((o for _, o in known.values() if o is not None), default=0)
    unknown_order = {
        level: base + i + 1
        for i, level in enumerate(sorted(unknown_sizes, key=lambda lv: (unknown_sizes[lv], lv)))
    }
    for r in parsed.ratings:
        person_id = person_ids.get(r.person_key)
        if person_id is None:
            continue
        name, order = known.get(r.level.casefold(), (None, None))
        conn.execute(
            "INSERT OR IGNORE INTO rating (person_id, level, level_name, level_order, "
            "year, quarter, place, staff) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                person_id,
                r.level,
                name or r.level,
                order if order is not None else unknown_order.get(r.level),
                r.year,
                r.quarter,
                r.place,
                r.staff,
            ),
        )


_SERIES_VIEW = """
CREATE VIEW v_series AS
SELECT v.*,
       LAG(v.fact) OVER w AS prev_fact,
       LAG(v.date) OVER w AS prev_date,
       FIRST_VALUE(v.fact) OVER w AS first_fact
FROM v_fact v
WINDOW w AS (PARTITION BY v.person_key, v.metric, COALESCE(v.element, '')
             ORDER BY v.period_idx)
"""


@dataclass
class MetricRef:
    """Результат резолва имени метрики."""

    metric_id: int
    name: str
    path: str | None = None


class RunDb:
    """Собранная база одного запуска: соединение + справочные операции."""

    def __init__(self, conn: sqlite3.Connection, report: BuildReport) -> None:
        self.conn = conn
        self.report = report
        # Защита text2sql, если она установлена на это соединение. Наши
        # собственные записи (карта отклонений) идут через writable().
        self.guard: Any = None

    @contextmanager
    def writable(self) -> Iterator[None]:
        """Временно снимает read-only защиту для НАШИХ записей, не для SQL модели."""
        guard = self.guard
        if guard is None:
            yield
            return
        with guard.writable():
            yield

    @property
    def has_stars(self) -> bool:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM fact WHERE star_received IS NOT NULL"
        ).fetchone()
        return bool(row[0])

    @property
    def has_aggregates(self) -> bool:
        return bool(self.conn.execute("SELECT COUNT(*) FROM peer_aggregate").fetchone()[0])

    @property
    def has_ratings(self) -> bool:
        """Рейтинг по звёздам есть только вместе со звёздами (см. _insert_ratings)."""
        return bool(self.conn.execute("SELECT COUNT(*) FROM rating").fetchone()[0])

    def resolve_metric(self, text: Any) -> MetricRef | None:
        """Имя метрики → каталог. Лестница: точное совпадение нормализованного
        имени → подстрока → нечёткое сходство ≥ порога."""
        norm = loader.norm_text(text)
        if not norm:
            return None
        rows = [
            (r["metric_id"], r["name"], r["name_norm"], r["path"])
            for r in self.conn.execute("SELECT metric_id, name, name_norm, path FROM metric")
        ]
        for mid, name, name_norm, path in rows:
            if name_norm == norm:
                return MetricRef(mid, name, path)
        contains = [r for r in rows if norm in r[2] or r[2] in norm]
        if len(contains) == 1:
            mid, name, _, path = contains[0]
            return MetricRef(mid, name, path)
        best: tuple[float, tuple] | None = None
        for row in rows:
            score = similarity_ratio(norm, row[2])
            if score >= _FUZZY_THRESHOLD and (best is None or score > best[0]):
                best = (score, row)
        if best is not None:
            mid, name, _, path = best[1]
            return MetricRef(mid, name, path)
        return None

    def resolve_element(self, person_key: str, metric: str, text: Any) -> str | None:
        """Имя разреза → точное значение из данных.

        Разрезов бывают сотни, и модель почти всегда называет их неточно:
        принимаем точное совпадение, регистр и подстроку.
        """
        norm = loader.norm_text(text)
        if not norm:
            return None
        rows = [
            r["element"]
            for r in self.conn.execute(
                "SELECT DISTINCT element FROM v_fact WHERE person_key = ? "
                "AND ru_lower(metric) = ru_lower(?) AND element IS NOT NULL",
                (person_key, metric),
            )
        ]
        for value in rows:
            if loader.norm_text(value) == norm:
                return value
        contains = [v for v in rows if norm in loader.norm_text(v)]
        if len(contains) == 1:
            return contains[0]
        best: tuple[float, str] | None = None
        for value in rows:
            score = similarity_ratio(norm, loader.norm_text(value))
            if score >= _FUZZY_THRESHOLD and (best is None or score > best[0]):
                best = (score, value)
        return best[1] if best else None

    def resolve_person(self, text: Any) -> str | None:
        """Текст → person_key: табельный/ключ → ФИО (подстрока) → 'я'/руководитель."""
        norm = loader.norm_text(text)
        if not norm:
            return None
        rows = [
            (r["person_key"], loader.norm_text(r["fio"]), r["is_me"])
            for r in self.conn.execute("SELECT person_key, fio, is_me FROM person")
        ]
        for key, fio_norm, _ in rows:
            if loader.norm_text(key) == norm:
                return key
        matches = [key for key, fio_norm, _ in rows if fio_norm and norm in fio_norm]
        if len(matches) == 1:
            return matches[0]
        return None


def build_run_db(
    raw_dataset: Any,
    raw_aggregates: Any = None,
    *,
    knowledge: dict[str, Any] | None = None,
    deviations: list[dict[str, Any]] | None = None,
    compute: bool = True,
) -> RunDb:
    """JSON датасета (+ агрегаты peer-групп) → готовая к запросам база.

    ``compute=False`` отдаёт базу без производных полей — нужно только тестам
    паритета, которые сверяют расчёт формул отдельно от подавления процентов.
    """
    parsed = loader.parse_dataset(raw_dataset if isinstance(raw_dataset, dict) else {})
    report = BuildReport(
        skipped_empty_leaves=parsed.report.skipped_empty_leaves,
        description_conflicts=parsed.report.description_conflicts,
        warnings=list(parsed.report.warnings),
    )
    conn = _connect()
    _insert_dataset(conn, parsed, report)
    # Вид показателя нужен ДО расчёта аналитики: у рангов и вкладов
    # относительные проценты подавляются (см. analytics.apply_metric_kinds).
    _apply_guessed_kinds(conn, parsed)
    _insert_aggregates(conn, loader.parse_aggregates(raw_aggregates), report)
    # После агрегатов: оттуда берутся названия уровней рейтинга.
    _insert_ratings(conn, parsed)
    if sqlite3.sqlite_version_info >= (3, 25):
        conn.executescript(_SERIES_VIEW)
    else:  # pragma: no cover — зависит от сборки SQLite на проме
        report.warnings.append("SQLite < 3.25: вью v_series недоступна")
    conn.commit()
    db = RunDb(conn, report)
    if compute:
        # Готовая база = данные + производные поля + подавление процентов у
        # показателей, которым проценты не идут (ранги, вклады).
        analytics_mod.compute_analytics(conn)
        analytics_mod.apply_kinds_from_catalog(conn)
    return db


__all__ = ["BuildReport", "MetricRef", "RunDb", "build_run_db", "schema_doc", "sql_examples"]
