"""In-memory SQLite: схема, загрузка плоских строк и параметрические запросы.

Наружу (для инструментов агента) отдаются только типизированные методы —
агент никогда не пишет SQL. Внутренние модули (analytics) используют .conn.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from .loader import ROW_FIELDS

_ANALYTICS_FIELDS: tuple[str, ...] = (
    "plan_dev_abs",
    "plan_dev_pct",
    "plan_status",
    "benchmark_dev_abs",
    "benchmark_dev_pct",
    "benchmark_status",
    "wow_change_abs",
    "wow_change_pct",
    "wow_status",
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

_GROUP_BY_COLUMNS = {
    "person": "m.person_fio",
    "element": "m.element",
    "date": "m.date",
    "post": "m.person_post",
}


class SqliteStore:
    """Хранилище метрик и производной аналитики в оперативной памяти."""

    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()
        self._element_info_cache: dict[str, dict[str, Any]] = {}

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE metrics (
                metric_uid          INTEGER PRIMARY KEY,
                parent_uid          INTEGER,
                depth               INTEGER,
                person_tabnum       INTEGER,
                person_fio          TEXT,
                person_post         TEXT,
                person_depart       TEXT,
                person_is_me        INTEGER,
                metric_id           TEXT,
                metric_name         TEXT,
                metric_description  TEXT,
                metric_type         TEXT,
                measure_type        TEXT,
                date                TEXT,
                calc_period         TEXT,
                fact                REAL,
                plan                REAL,
                benchmark           REAL,
                influent_percent    REAL,
                element             TEXT
            );

            CREATE TABLE metric_analytics (
                metric_uid       INTEGER PRIMARY KEY REFERENCES metrics(metric_uid),
                plan_dev_abs     REAL,
                plan_dev_pct     REAL,
                plan_status      TEXT,
                benchmark_dev_abs REAL,
                benchmark_dev_pct REAL,
                benchmark_status TEXT,
                wow_change_abs   REAL,
                wow_change_pct   REAL,
                wow_status       TEXT,
                trend            TEXT,
                trend_status     TEXT,
                peer_mean        REAL,
                peer_std         REAL,
                peer_count       INTEGER,
                peer_rank        INTEGER,
                peer_percentile  REAL,
                zscore           REAL,
                peer_status      TEXT,
                is_anomaly       INTEGER
            );

            CREATE TABLE metric_relations (
                source     TEXT,
                target     TEXT,
                relation   TEXT,
                strength   TEXT,
                rationale  TEXT
            );

            CREATE INDEX idx_rel_source ON metric_relations(source);
            CREATE INDEX idx_rel_target ON metric_relations(target);

            CREATE INDEX idx_metrics_name   ON metrics(metric_name);
            CREATE INDEX idx_metrics_date   ON metrics(date);
            CREATE INDEX idx_metrics_elem   ON metrics(element);
            CREATE INDEX idx_metrics_person ON metrics(person_tabnum);
            CREATE INDEX idx_metrics_parent ON metrics(parent_uid);
            """
        )

    def load(self, rows: list[dict[str, Any]]) -> int:
        cols = ", ".join(ROW_FIELDS)
        placeholders = ", ".join("?" for _ in ROW_FIELDS)
        self.conn.executemany(
            f"INSERT INTO metrics ({cols}) VALUES ({placeholders})",
            [tuple(r[f] for f in ROW_FIELDS) for r in rows],
        )
        self.conn.commit()
        return len(rows)

    @staticmethod
    def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
        return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def _person_clause(person: str | int | None) -> tuple[str, list[Any]]:
        if person is None or str(person).strip() == "":
            return "", []
        text = str(person).strip()
        if text.isdigit():
            return " AND m.person_tabnum = ?", [int(text)]
        return " AND m.person_fio LIKE ?", [f"%{text}%"]

    @staticmethod
    def _element_clause(
        element: str | None, aggregate_default: bool = False
    ) -> tuple[str, list[Any]]:
        if element is None or (isinstance(element, str) and element.strip() == ""):
            return (" AND m.element IS NULL", []) if aggregate_default else ("", [])
        return " AND m.element = ?", [element]

    def metric_type_of(self, name: str) -> str | None:
        cur = self.conn.execute(
            "SELECT metric_type FROM metrics WHERE metric_name = ? LIMIT 1", (name,)
        )
        row = cur.fetchone()
        return row["metric_type"] if row else None

    def _metric_element_info(self, name: str) -> dict[str, Any]:
        cached = self._element_info_cache.get(name)
        if cached is not None:
            return cached
        has_agg = (
            self.conn.execute(
                "SELECT 1 FROM metrics WHERE metric_name = ? "
                "AND element IS NULL LIMIT 1",
                (name,),
            ).fetchone()
            is not None
        )
        elements = [
            r["element"]
            for r in self.conn.execute(
                "SELECT DISTINCT element FROM metrics WHERE metric_name = ? "
                "AND element IS NOT NULL ORDER BY element",
                (name,),
            )
        ]
        info = {"has_aggregate": has_agg, "elements": elements}
        self._element_info_cache[name] = info
        return info

    def _has_aggregate_row(
        self, name: str, person: str | int | None = None, date: str | None = None
    ) -> bool:
        """Есть ли у метрики агрегатная строка (element IS NULL) В РАМКАХ scope.

        Наличие агрегата проверяется с учётом фильтра по человеку/дате, а НЕ
        глобально по имени метрики. У сотрудника метрика может быть представлена
        только разрезами (element), тогда как у руководителя по той же метрике
        есть агрегат — глобальный флаг дал бы ложное True и скрыл данные
        сотрудника (запрос ушёл бы в element IS NULL и вернул 0 строк).
        """
        where = "m.metric_name = ? AND m.element IS NULL"
        params: list[Any] = [name]
        pc, pp = self._person_clause(person)
        where += pc
        params += pp
        if date:
            where += " AND m.date = ?"
            params.append(date)
        return (
            self.conn.execute(
                f"SELECT 1 FROM metrics m WHERE {where} LIMIT 1", params
            ).fetchone()
            is not None
        )

    def _elements_for(self, name: str, person: str | int | None = None) -> list[str]:
        """Значения element метрики в рамках фильтра по человеку (для пометки
        'разрезы_вместо_агрегата')."""
        where = "m.metric_name = ? AND m.element IS NOT NULL"
        params: list[Any] = [name]
        pc, pp = self._person_clause(person)
        where += pc
        params += pp
        return [
            r["element"]
            for r in self.conn.execute(
                f"SELECT DISTINCT m.element FROM metrics m WHERE {where} "
                "ORDER BY m.element",
                params,
            )
        ]

    def row_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) c FROM metrics").fetchone()["c"]

    def analytics_row_count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) c FROM metric_analytics"
        ).fetchone()["c"]

    def schema_overview(self) -> dict[str, Any]:
        metrics = self._rows(
            self.conn.execute(
                "SELECT metric_name, metric_type, measure_type, COUNT(*) AS rows "
                "FROM metrics GROUP BY metric_name, metric_type, measure_type "
                "ORDER BY metric_name"
            )
        )
        for m in metrics:
            info = self._metric_element_info(m["metric_name"])
            m["has_aggregate"] = bool(info["has_aggregate"])
            m["elements"] = info["elements"]
        elements = [
            r["element"]
            for r in self.conn.execute(
                "SELECT DISTINCT element FROM metrics "
                "WHERE element IS NOT NULL ORDER BY element"
            )
        ]
        people = self._rows(
            self.conn.execute(
                "SELECT person_tabnum, person_fio, person_post, person_depart, "
                "MAX(person_is_me) AS person_is_me FROM metrics "
                "GROUP BY person_tabnum ORDER BY person_is_me DESC, person_fio"
            )
        )
        dates = [
            r["date"]
            for r in self.conn.execute(
                "SELECT DISTINCT date FROM metrics WHERE date IS NOT NULL ORDER BY date"
            )
        ]
        return {
            "metrics": metrics,
            "elements": elements,
            "people": people,
            "dates": dates,
            "total_metric_rows": self.row_count(),
        }

    def describe_metric(self, name: str) -> dict[str, Any] | None:
        cur = self.conn.execute(
            "SELECT metric_name, metric_description, metric_type, measure_type, "
            "calc_period FROM metrics WHERE metric_name = ? LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def distinct_metric_names(self) -> list[str]:
        return [
            r["metric_name"]
            for r in self.conn.execute(
                "SELECT DISTINCT metric_name FROM metrics "
                "WHERE metric_name IS NOT NULL ORDER BY metric_name"
            )
        ]

    def distinct_descriptions(self) -> list[tuple[str, str]]:
        rows = self.conn.execute(
            "SELECT metric_name, metric_description FROM metrics "
            "WHERE metric_description IS NOT NULL "
            "GROUP BY metric_name, metric_description"
        )
        return [(r["metric_name"], r["metric_description"]) for r in rows]

    def distinct_elements(self) -> list[str]:
        return [
            r["element"]
            for r in self.conn.execute(
                "SELECT DISTINCT element FROM metrics "
                "WHERE element IS NOT NULL ORDER BY element"
            )
        ]

    # --- Граф смысловых связей метрик (Блок D ТЗ) ---------------------------

    def relation_catalog(self) -> list[dict[str, Any]]:
        """Каталог метрик для LLM-вывода связей: имя, описание, тип, единица,
        имя родительской метрики и есть ли вручную заданный influent_percent.
        По одной записи на distinct-метрику (граф — на уровне имён)."""
        rows = self.conn.execute(
            "SELECT m.metric_name, m.metric_description, m.metric_type, "
            "m.measure_type, p.metric_name AS parent_name, "
            "MAX(m.influent_percent IS NOT NULL) AS has_influent "
            "FROM metrics m LEFT JOIN metrics p ON m.parent_uid = p.metric_uid "
            "WHERE m.metric_name IS NOT NULL "
            "GROUP BY m.metric_name ORDER BY m.metric_name"
        )
        return [
            {
                "metric_name": r["metric_name"],
                "metric_description": r["metric_description"],
                "metric_type": r["metric_type"],
                "measure_type": r["measure_type"],
                "parent": r["parent_name"],
                "has_influent": bool(r["has_influent"]),
            }
            for r in rows
        ]

    def load_relations(self, edges: list[dict[str, Any]]) -> int:
        """Загружает рёбра графа связей в таблицу metric_relations (идемпотентно)."""
        self.conn.execute("DELETE FROM metric_relations")
        self.conn.executemany(
            "INSERT INTO metric_relations (source, target, relation, strength, "
            "rationale) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    e.get("source"),
                    e.get("target"),
                    e.get("relation"),
                    e.get("strength"),
                    e.get("rationale"),
                )
                for e in (edges or [])
                if e.get("source") and e.get("target")
            ],
        )
        self.conn.commit()
        return len(edges or [])

    def related_metrics(self, name: str) -> list[dict[str, Any]]:
        """Рёбра графа, где метрика участвует как source или target.
        Сильные связи первыми ('высокая' → 'средняя' → 'низкая')."""
        rows = self.conn.execute(
            "SELECT source, target, relation, strength, rationale "
            "FROM metric_relations WHERE source = ? OR target = ? "
            "ORDER BY CASE strength WHEN 'высокая' THEN 0 WHEN 'средняя' THEN 1 "
            "ELSE 2 END, target",
            (name, name),
        )
        return [dict(r) for r in rows]

    def _relation_between(self, a: str, b: str) -> dict[str, Any] | None:
        """LLM-выведенная связь между двумя метриками (любое направление)."""
        row = self.conn.execute(
            "SELECT source, target, relation, strength, rationale "
            "FROM metric_relations "
            "WHERE (source = ? AND target = ?) OR (source = ? AND target = ?) "
            "ORDER BY CASE strength WHEN 'высокая' THEN 0 WHEN 'средняя' THEN 1 "
            "ELSE 2 END LIMIT 1",
            (a, b, b, a),
        ).fetchone()
        return dict(row) if row else None

    def list_people(
        self,
        role: str | None = None,
        post: str | None = None,
        depart: str | None = None,
        name_query: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if role == "me":
            clauses.append("person_is_me = 1")
        elif role == "employee":
            clauses.append("person_is_me = 0")
        if post:
            clauses.append("person_post = ?")
            params.append(post)
        if depart:
            clauses.append("person_depart = ?")
            params.append(depart)
        if name_query:
            clauses.append("person_fio LIKE ?")
            params.append(f"%{name_query}%")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return self._rows(
            self.conn.execute(
                "SELECT person_tabnum, person_fio, person_post, person_depart, "
                f"MAX(person_is_me) AS person_is_me FROM metrics{where} "
                "GROUP BY person_tabnum ORDER BY person_is_me DESC, person_fio",
                params,
            )
        )

    def _select_metrics(
        self,
        where: str,
        params: list[Any],
        order: str = "m.date, m.person_fio, m.element",
        limit: int = 60,
    ) -> dict[str, Any]:
        analytics_cols = ", ".join(f"a.{c}" for c in _ANALYTICS_FIELDS)
        sql = (
            "SELECT m.metric_uid, m.depth, m.person_tabnum, m.person_fio, "
            "m.person_post, m.person_is_me, m.metric_name, m.metric_type, "
            "m.measure_type, m.date, m.element, m.fact, m.plan, m.benchmark, "
            "m.influent_percent, "
            f"{analytics_cols} "
            "FROM metrics m LEFT JOIN metric_analytics a "
            "ON a.metric_uid = m.metric_uid "
            f"WHERE {where} ORDER BY {order} LIMIT ?"
        )
        rows = self._rows(self.conn.execute(sql, [*params, limit + 1]))
        truncated = len(rows) > limit
        return {"rows": rows[:limit], "count": min(len(rows), limit), "truncated": truncated}

    def get_metric(
        self,
        name: str,
        person: str | None = None,
        element: str | None = None,
        date: str | None = None,
        limit: int = 60,
    ) -> dict[str, Any]:
        where = "m.metric_name = ?"
        params: list[Any] = [name]
        pc, pp = self._person_clause(person)
        where += pc
        params += pp
        element_unspecified = element is None or (
            isinstance(element, str) and element.strip() == ""
        )
        fallback_elements: list[str] | None = None
        if element_unspecified:
            if self._has_aggregate_row(name, person=person, date=date):
                where += " AND m.element IS NULL"
            else:
                fallback_elements = self._elements_for(name, person=person)
        else:
            where += " AND m.element = ?"
            params.append(element)
        if date:
            where += " AND m.date = ?"
            params.append(date)
        result = self._select_metrics(where, params, limit=limit)
        if fallback_elements is not None:
            result["разрезы_вместо_агрегата"] = (
                "у метрики нет агрегатной строки; показаны все разрезы по element: "
                + ", ".join(fallback_elements)
            )
        return result

    def compare(
        self,
        name: str,
        person: str | None = None,
        element: str | None = None,
        dates: list[str] | None = None,
        limit: int = 80,
    ) -> dict[str, Any]:
        if person is None:
            return {
                "error": (
                    "Для динамики укажи person. Чтобы найти, у кого самый сильный "
                    "спад или рост по всем сотрудникам, используй find_flags "
                    "(kind='trend')."
                ),
                "rows": [],
                "count": 0,
            }
        where = "m.metric_name = ?"
        params: list[Any] = [name]
        pc, pp = self._person_clause(person)
        where += pc
        params += pp
        element_unspecified = element is None or (
            isinstance(element, str) and element.strip() == ""
        )
        fallback_elements: list[str] | None = None
        if element_unspecified:
            if self._has_aggregate_row(name, person=person):
                where += " AND m.element IS NULL"
            else:
                fallback_elements = self._elements_for(name, person=person)
        else:
            where += " AND m.element = ?"
            params.append(element)
        if dates:
            placeholders = ", ".join("?" for _ in dates)
            where += f" AND m.date IN ({placeholders})"
            params += list(dates)
        result = self._select_metrics(
            where, params, order="m.person_fio, m.element, m.date", limit=limit
        )
        if fallback_elements is not None:
            result["разрезы_вместо_агрегата"] = (
                "у метрики нет агрегатной строки; динамика показана по всем "
                "разрезам element: " + ", ".join(fallback_elements)
            )
        return result

    def rank(
        self,
        name: str,
        date: str,
        element: str | None = None,
        post: str | None = None,
        limit: int = 30,
    ) -> dict[str, Any]:
        where = "m.metric_name = ? AND m.date = ? AND m.person_is_me = 0"
        params: list[Any] = [name, date]
        if element is None or (isinstance(element, str) and element.strip() == ""):
            # Наличие агрегата для рейтинга проверяем по ранжируемой популяции
            # (сотрудники, person_is_me = 0) на эту дату, а не глобально по имени:
            # у руководителя может быть агрегат, а у сотрудников — только разрезы.
            has_agg = (
                self.conn.execute(
                    "SELECT 1 FROM metrics WHERE metric_name = ? AND date = ? "
                    "AND person_is_me = 0 AND element IS NULL LIMIT 1",
                    (name, date),
                ).fetchone()
                is not None
            )
            if not has_agg:
                return {
                    "error": (
                        f"У метрики '{name}' нет агрегатной строки — рейтинг "
                        "строится по одному конкретному разрезу. Передай "
                        "element и повтори."
                    ),
                    "rows": [],
                    "count": 0,
                    "metric_type": self.metric_type_of(name),
                    "elements": self._elements_for(name),
                }
            where += " AND m.element IS NULL"
        else:
            where += " AND m.element = ?"
            params.append(element)
        if post:
            where += " AND m.person_post = ?"
            params.append(post)
        result = self._select_metrics(
            where, params, order="a.peer_rank IS NULL, a.peer_rank", limit=limit
        )
        result["metric_type"] = self.metric_type_of(name)
        return result

    def aggregate(
        self,
        name: str,
        group_by: str,
        date: str | None = None,
        element: str | None = None,
    ) -> dict[str, Any]:
        column = _GROUP_BY_COLUMNS.get(group_by)
        if column is None:
            return {
                "error": f"group_by должен быть одним из {sorted(_GROUP_BY_COLUMNS)}",
                "groups": [],
            }
        where = "m.metric_name = ?"
        params: list[Any] = [name]
        if date:
            where += " AND m.date = ?"
            params.append(date)
        ec, ep = self._element_clause(element)
        where += ec
        params += ep
        rows = self._rows(
            self.conn.execute(
                f"SELECT {column} AS grp, COUNT(m.fact) AS n, "
                "AVG(m.fact) AS avg, MIN(m.fact) AS min, MAX(m.fact) AS max, "
                "SUM(m.fact) AS sum FROM metrics m "
                f"WHERE {where} GROUP BY {column} ORDER BY avg DESC",
                params,
            )
        )
        return {
            "metric": name,
            "metric_type": self.metric_type_of(name),
            "group_by": group_by,
            "groups": rows,
        }

    def metric_tree(
        self,
        name: str | None = None,
        person: str | None = None,
        date: str | None = None,
        max_levels: int = 1,
        limit: int = 80,
    ) -> dict[str, Any]:
        fallback_elements: list[str] | None = None
        if name is None:
            root_where = "depth = 1"
            root_params: list[Any] = []
        else:
            if self._has_aggregate_row(name, person=person):
                root_where = "metric_name = ? AND element IS NULL"
            else:
                root_where = "metric_name = ?"
                fallback_elements = self._elements_for(name, person=person)
            root_params = [name]
        pc, pp = self._person_clause(person)
        root_where += pc.replace("m.", "")
        root_params += pp
        if date:
            root_where += " AND date = ?"
            root_params.append(date)
        # Разворачиваем дерево не целиком, а на max_levels уровней от корня (по
        # умолчанию ОДИН: корень + прямые дети). lvl считается в рекурсивном CTE;
        # детей узла уровня (max_levels-1) дальше НЕ раскрываем — иначе на одну
        # верхнеуровневую метрику вываливалось бы всё поддерево. has_children
        # помечает, есть ли у узла собственные подкомпоненты (нераскрытые на
        # этом шаге), чтобы агент знал, где ещё можно копать, а где лист.
        sql = (
            "WITH RECURSIVE tree(metric_uid, lvl) AS ("
            f"  SELECT metric_uid, 0 FROM metrics WHERE {root_where}"
            "  UNION ALL"
            "  SELECT m.metric_uid, t.lvl + 1 FROM metrics m "
            "  JOIN tree t ON m.parent_uid = t.metric_uid"
            "  WHERE t.lvl < ?"
            ") "
            "SELECT m.metric_uid, m.parent_uid, m.depth, m.person_fio, "
            "m.metric_name, m.metric_type, m.measure_type, m.date, m.element, "
            "m.fact, m.plan, m.benchmark, m.influent_percent, "
            "a.plan_status, a.plan_dev_pct, a.benchmark_status, "
            "a.benchmark_dev_pct, a.trend, a.trend_status, "
            "a.wow_change_pct, a.wow_status, "
            "EXISTS(SELECT 1 FROM metrics c WHERE c.parent_uid = m.metric_uid) "
            "AS has_children "
            "FROM metrics m JOIN tree t ON m.metric_uid = t.metric_uid "
            "LEFT JOIN metric_analytics a ON a.metric_uid = m.metric_uid "
            "ORDER BY m.person_fio, m.depth, m.metric_uid LIMIT ?"
        )
        rows = self._rows(
            self.conn.execute(sql, [*root_params, max_levels, limit + 1])
        )
        shown = rows[:limit]
        self._annotate_tree_relations(shown)
        result: dict[str, Any] = {
            "rows": shown,
            "count": min(len(rows), limit),
            "truncated": len(rows) > limit,
            "levels_shown": max_levels,
        }
        if fallback_elements is not None:
            result["разрезы_вместо_агрегата"] = (
                "у метрики нет агрегатной строки; дерево развернуто от всех "
                "разрезов element: " + ", ".join(fallback_elements)
            )
        return result

    def _annotate_tree_relations(self, rows: list[dict[str, Any]]) -> None:
        """Запасной сигнал влияния для узлов БЕЗ influent_percent (Блок D ТЗ).

        Бизнес часто не проставляет influent_percent. Для таких узлов
        подмешиваем LLM-выведенную связь ребёнок↔родитель из metric_relations:
        influent_percent_missing=True и inferred_relation (или None, если в
        графе связи нет — значит граф спрошен, но пусто). Это ЭВРИСТИКА из
        текста, не точный вес."""
        uid_to_name = {
            r.get("metric_uid"): r.get("metric_name")
            for r in rows
            if r.get("metric_uid") is not None
        }
        for r in rows:
            if r.get("influent_percent") is not None:
                continue
            parent_uid = r.get("parent_uid")
            if parent_uid is None:
                continue
            child_name = r.get("metric_name")
            parent_name = uid_to_name.get(parent_uid)
            if parent_name is None:
                row = self.conn.execute(
                    "SELECT metric_name FROM metrics WHERE metric_uid = ? LIMIT 1",
                    (parent_uid,),
                ).fetchone()
                parent_name = row["metric_name"] if row else None
            r["influent_percent_missing"] = True
            r["inferred_relation"] = (
                self._relation_between(child_name, parent_name)
                if child_name and parent_name else None
            )

    def find_flags(
        self,
        kind: str,
        date: str | None = None,
        metric: str | None = None,
        element: str | None = None,
        limit: int = 25,
    ) -> dict[str, Any]:
        where = "1 = 1"
        params: list[Any] = []
        if kind == "anomaly":
            where += " AND a.is_anomaly = 1"
        elif kind == "below_plan":
            where += " AND a.plan_status = 'хуже_плана'"
        elif kind == "above_plan":
            where += " AND a.plan_status = 'лучше_плана'"
        elif kind == "trend":
            where += " AND a.trend IN ('рост', 'падение')"
        elif kind == "declining":
            where += " AND a.trend_status = 'ухудшение'"
        elif kind == "improving":
            where += " AND a.trend_status = 'улучшение'"
        else:
            return {
                "error": "kind должен быть 'anomaly', 'below_plan', "
                "'above_plan', 'trend', 'declining' или 'improving'",
                "rows": [],
            }
        if date:
            where += " AND m.date = ?"
            params.append(date)
        if metric:
            where += " AND m.metric_name = ?"
            params.append(metric)
        if element is not None and not (isinstance(element, str) and element.strip() == ""):
            where += " AND m.element = ?"
            params.append(element)
        order = {
            "anomaly": "ABS(a.zscore) DESC",
            "below_plan": "(a.plan_dev_pct IS NULL), ABS(a.plan_dev_pct) DESC",
            "above_plan": "(a.plan_dev_pct IS NULL), ABS(a.plan_dev_pct) DESC",
            "trend": "(a.wow_change_pct IS NULL), a.wow_change_pct ASC",
            "declining": "(a.wow_change_pct IS NULL), ABS(a.wow_change_pct) DESC",
            "improving": "(a.wow_change_pct IS NULL), ABS(a.wow_change_pct) DESC",
        }[kind]
        return self._select_metrics(where, params, order=order, limit=limit)
