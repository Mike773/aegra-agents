"""Документация схемы для системного промпта (вход text2sql).

Собирается из самой базы (список вью и колонок) плюс курируемый словарь
семантики колонок и примеры запросов. То, чего в датасете нет (звёзды,
агрегаты, rankings), не рекламируется — модель не должна звать несуществующее.
"""
from __future__ import annotations

from typing import Any

# Семантика колонок: что означает значение и как его читать. Только то, что
# нельзя угадать по имени, — остальное модель поймёт сама.
COLUMN_DOCS: dict[tuple[str, str], str] = {
    ("v_fact", "person_key"): "идентификатор человека (табельный или ФИО)",
    ("v_fact", "is_me"): "1 = руководитель, 0 = сотрудник",
    ("v_fact", "metric"): "название показателя (как в данных)",
    ("v_fact", "direction"): "'прямая' (больше = лучше) | 'обратная' (меньше = лучше)",
    ("v_fact", "kind"): "'уровень' | 'вклад' | 'индекс'; у не-'уровня' проценты подавлены",
    ("v_fact", "depth"): "уровень в дереве показателей ЭТОГО человека: 1 = верхний",
    ("v_fact", "path"): "путь по дереву этого человека: 'Родитель > Ребёнок'",
    ("v_fact", "element"): (
        "разрез показателя; NULL = собственный итог показателя. Смотри итог, "
        "разрезы — для декомпозиции. Разрезов у показателя бывают сотни"
    ),
    ("v_metric_person", "has_aggregate"): (
        "1 = у этого человека есть собственный итог показателя (строка с "
        "element IS NULL); 0 = показатель существует только разрезами"
    ),
    ("v_metric_person", "n_elements"): "сколько разрезов у этого человека",
    ("v_metric", "n_elements"): (
        "число разрезов по ВСЕМ людям; для конкретного человека бери v_metric_person"
    ),
    ("v_fact", "date"): "дата периода (ISO)",
    ("v_fact", "period_idx"): "порядковый номер даты в датасете (1 = самая ранняя)",
    ("v_fact", "is_last_of_series"): (
        "1 = последнее значение ИМЕННО ЭТОЙ серии (человек × показатель × разрез)"
    ),
    ("v_fact", "ex"): "процент выполнения плана",
    ("v_fact", "rr"): "run rate — ожидаемый результат к концу периода",
    ("v_fact", "plan_status"): "лучше_плана | в_плане | хуже_плана (уже с учётом direction)",
    ("v_fact", "benchmark_status"): "сравнение с средним топ-20% группы",
    ("v_fact", "pop_status"): "улучшение | стабильно | ухудшение к прошлому периоду",
    ("v_fact", "trend_status"): "вердикт тренда по всему ряду",
    ("v_fact", "peer_status"): "лучше_коллег | на_уровне_коллег | хуже_коллег (внутри команды)",
    ("v_fact", "peer_rank"): "место внутри загруженной команды (1 = лучший)",
    ("v_fact", "rel_status"): "личная динамика против динамики группы (детрендинг)",
    ("v_fact", "group_hit_rate"): "доля коллег группы, выполняющих план, %",
    ("v_fact", "plan_rigidity"): "жёсткий_план | обычный_план | мягкий_план",
    ("v_fact", "is_anomaly"): "1 = выброс относительно коллег (|z| ≥ порога)",
    ("v_fact", "star_received"): "результат звезды: 1 получена, 0 нет; NULL = не звезда",
    ("v_tree", "person_key"): "чьё дерево: у каждого человека состав показателей свой",
    ("v_tree", "share_pct"): "доля влияния ребёнка среди детей родителя, %",
    ("v_metric", "depth"): "уровень в СВОДНОМ дереве по всем людям; для человека бери v_fact/v_tree",
    ("v_star", "received"): "1 = звезда получена, 0 = нет (на последнюю дату)",
    ("v_star", "n_below_plan"): "сколько влияющих показателей звезды хуже плана",
    ("v_star", "metrics"): (
        "влияющие показатели одной строкой: имя, факт, план, выполнение %, вердикт"
    ),
    ("v_star_metric", "completion_pct"): "выполнение плана, % (факт / план)",
    ("v_rating", "level_name"): "название уровня рейтинга — только его и называй",
    ("v_rating", "level_order"): "1 = самая узкая группа (ГОСБ), дальше шире (ТБ, Сбер)",
    ("v_rating", "place"): "место сотрудника в рейтинге по звёздам, 1 = лучший",
    ("v_rating", "staff"): "сколько сотрудников в рейтинге этого уровня",
    ("v_rating", "is_latest"): "1 = последний квартал, за который у человека есть рейтинг",
    ("v_peer_latest", "level_order"): "1 = самая узкая группа сравнения",
    ("v_peer_latest", "level_name"): "человеческое название группы — только его и называй",
    ("v_deviation", "priority"): "приоритет отклонения: влияние × масштаб × управляемость",
    ("v_deviation", "source"): "'auto' — посчитано правилами, 'agent' — отмечено тобой",
}

_RULES = """ПРАВИЛА РАБОТЫ С ДАННЫМИ
- Запросы только на чтение: один SELECT (можно WITH), без изменения данных.
- Вердикты в колонках *_status посчитаны С УЧЁТОМ направления показателя —
  переноси их как есть, не выводи оценку заново из знака отклонения.
- Последние значения РАЗНЫХ показателей могут быть на РАЗНЫЕ даты: у дочернего
  показателя данные приходят позже родительского. Бери последнее значение серии
  (is_last_of_series = 1 или v_fact_latest) и всегда сверяй колонку date — не
  предполагай, что у всех показателей дата общая, и не отбрасывай показатель
  из-за того, что его дата старше.
- Дерево показателей (v_tree, path, depth) от дат не зависит вовсе. Дерево у
  КАЖДОГО человека своё — фильтруй v_tree по person_key, а depth/path бери из
  v_fact того человека, о ком говоришь.
- element IS NULL — СОБСТВЕННЫЙ ИТОГ показателя. Его и бери за значение
  показателя; разрезы нужны, чтобы объяснить, из чего итог сложился.
- Если у показателя нет строки с element IS NULL (v_metric_person.has_aggregate = 0),
  собственного итога у него НЕТ. Не суммируй разрезы и не выдавай сумму за факт:
  складывать проценты, средние и ранги бессмысленно. Говори про разрезы и
  честно отмечай, что общего значения у показателя нет.
- Разрезов бывают СОТНИ. Не перечисляй их и не тяни целиком: фильтруй по имени
  (LIKE) и всегда ставь LIMIT, либо бери верхушку через metric_card.
- Диалект — SQLite: таблицы DUAL нет (SELECT без FROM допустим), функций
  DATEDIFF / WEEKS_BETWEEN / DATE_ADD нет. Даты — строки ISO 'YYYY-MM-DD'; с ними
  работают date(), julianday(), strftime(). Разница в днях —
  julianday(d2) - julianday(d1); в неделях — делить на 7; сдвиг —
  date(d, '+7 days'), date(d, 'start of month', '+1 month', '-1 day').
  Считай даты и остатки до срока запросом, а не в уме. Оконные функции
  (LAG, SUM() OVER) доступны — приросты и накопления считай ими."""

# Примеры запросов: (что просит руководитель, SQL). Идут в GigaChat как
# few_shot_examples функции query_sql и текстом — в справку для отладки.
# {person} и {metric} подставляются из датасета.
SQL_EXAMPLES: list[tuple[str, str]] = [
    (
        "Показатели верхнего уровня, отстающие от плана (последние значения серий)",
        """SELECT metric, date, fact, plan, plan_dev_pct, pop_status
FROM v_fact_latest
WHERE person_key = '{person}' AND depth = 1 AND element IS NULL
  AND plan_status = 'хуже_плана'
ORDER BY ABS(plan_dev_pct) DESC;""",
    ),
    (
        "История одного показателя с изменением к прошлому периоду",
        """SELECT date, fact, plan, ex, rr, pop_change_pct, pop_status
FROM v_fact
WHERE person_key = '{person}' AND metric = '{metric}' AND element IS NULL
ORDER BY period_idx;""",
    ),
    (
        "Худшие разрезы показателя с учётом направления",
        """SELECT element, fact, plan, plan_dev_pct, plan_status
FROM v_fact_latest
WHERE person_key = '{person}' AND metric = '{metric}' AND element IS NOT NULL
ORDER BY CASE WHEN direction = 'обратная' THEN -fact ELSE fact END
LIMIT 10;""",
    ),
    (
        "Состав ветки дерева этого человека с весами влияния",
        """SELECT child, influent_percent, share_pct, child_depth
FROM v_tree WHERE person_key = '{person}' AND parent = '{metric}'
ORDER BY share_pct DESC;""",
    ),
    (
        "Найти разрез по части имени (разрезов сотни — фильтруй и ограничивай)",
        """SELECT element, date, fact, plan, plan_status
FROM v_fact_latest
WHERE person_key = '{person}' AND metric = '{metric}'
  AND element IS NOT NULL AND ru_lower(element) LIKE '%часть имени%'
LIMIT 20;""",
    ),
    (
        "Есть ли у показателей собственный итог и сколько у них разрезов",
        """SELECT metric, has_aggregate, n_elements
FROM v_metric_person WHERE person_key = '{person}' AND n_elements > 0;""",
    ),
    (
        "Сколько недель осталось от последней даты данных до срока (конец "
        "квартала) и сколько нужно в неделю, чтобы закрыть разрыв до цели 312 "
        "(срок и цель — числа из вопроса, подставляй литералами)",
        """WITH last AS (
  SELECT date, fact FROM v_fact_latest
  WHERE person_key = '{person}' AND metric = '{metric}' AND element IS NULL
)
SELECT date, fact,
       CAST((julianday('2026-06-30') - julianday(date)) / 7 AS INTEGER) AS weeks_left,
       ROUND((312 - fact) / ((julianday('2026-06-30') - julianday(date)) / 7), 2)
         AS need_per_week
FROM last;""",
    ),
    (
        "Прирост к прошлому периоду своими руками (например, для накопительных серий)",
        """SELECT date, fact, fact - LAG(fact) OVER (ORDER BY period_idx) AS fact_delta
FROM v_fact
WHERE person_key = '{person}' AND metric = '{metric}' AND element IS NULL
ORDER BY period_idx;""",
    ),
]


def _views(conn) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def _columns(conn, view: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({view})").fetchall()]


def build_schema_doc(db: Any, *, examples: bool = True) -> str:
    """Справка по схеме: вью с колонками, семантика, правила, примеры.

    ``examples=False`` — без текстовых примеров: инструмент query_sql отдаёт их
    модели структурно, через ``sql_examples``."""
    conn = db.conn
    views = [v for v in _views(conn) if _view_has_data(db, v)]

    lines = [
        "СХЕМА ДАННЫХ (SQLite, только чтение)",
        "Обращайся к представлениям ниже — в них уже склеены человек, показатель, "
        "период и готовые вердикты. Базовые таблицы служебные.",
        "",
    ]
    # Список колонок v_fact — полсотни имён; v_fact_latest и v_series
    # состоят из тех же колонок, и печатать их трижды незачем.
    base_cols = _columns(conn, "v_fact")
    for view in views:
        cols = _columns(conn, view)
        if not cols:
            continue
        extra = [c for c in cols if c not in base_cols]
        if view != "v_fact" and base_cols and len(cols) - len(extra) == len(base_cols):
            tail = f", плюс {', '.join(extra)}" if extra else ""
            lines.append(f"{view}: те же колонки, что v_fact{tail}")
        else:
            lines.append(f"{view}({', '.join(cols)})")
        for col in cols:
            doc = COLUMN_DOCS.get((view, col))
            if doc:
                lines.append(f"  - {col}: {doc}")
        lines.append("")

    lines.append(_RULES)
    dataset_rules = _dataset_rules(db)
    if dataset_rules:
        lines.append(dataset_rules)
    if examples:
        lines.append("")
        lines.append(_examples_block(conn))
    return "\n".join(lines).strip()


def _view_has_data(db: Any, view: str) -> bool:
    """Вью, за которыми в этом датасете нет данных, в справку не идут — модель
    не должна звать несуществующее."""
    conn = db.conn
    if view == "v_deviation":
        return _has_rows(conn, "deviation")
    if view in ("v_star", "v_star_metric"):
        return bool(db.has_stars)
    if view == "v_rating":
        return bool(db.has_ratings)
    if view == "v_peer_latest":
        return bool(db.has_aggregates) or _has_rows(conn, "ranking")
    return True


def _has_rows(conn, table: str) -> bool:
    return bool(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _dataset_rules(db: Any) -> str:
    """Подсказки про вью, которые есть не в каждом датасете. Состав данных
    (люди, периоды, число показателей) здесь не повторяется — он в блоке
    «СОСТАВ ДАННЫХ»."""
    conn = db.conn
    parts = []
    if db.has_aggregates:
        levels = conn.execute(
            "SELECT DISTINCT level_name FROM peer_aggregate "
            "WHERE level_name IS NOT NULL ORDER BY level_order"
        ).fetchall()
        if levels:
            parts.append(
                "- Группы сравнения (v_peer_latest, от узкой к широкой): "
                + ", ".join(r["level_name"] for r in levels)
                + "."
            )
        else:
            parts.append(
                "- Группы сравнения (v_peer_latest) есть, но без названий — "
                "называй их обезличенно."
            )
    if _has_rows(conn, "ranking"):
        parts.append("- Есть места в больших группах коллег (v_peer_latest.rank_raw).")
    if db.has_stars:
        parts.append(
            "- Есть звёзды: именные показатели без чисел (получена/не получена). "
            "v_star — строка на звезду с влияющими показателями одной строкой, "
            "v_star_metric — те же показатели построчно."
        )
    if db.has_ratings:
        parts.append(
            "- Есть рейтинг по звёздам (v_rating): место сотрудника среди коллег по "
            "уровням за квартал; уровни называй по level_name, место — вместе со staff."
        )
    return "\n".join(parts)


def _example_slots(conn) -> dict[str, str]:
    person = conn.execute(
        "SELECT person_key FROM person ORDER BY is_me, person_id LIMIT 1"
    ).fetchone()
    metric = conn.execute(
        "SELECT name FROM metric WHERE depth = 1 ORDER BY metric_id LIMIT 1"
    ).fetchone()
    return {
        "person": person["person_key"] if person else "…",
        "metric": metric["name"] if metric else "…",
    }


def _examples_block(conn) -> str:
    slots = _example_slots(conn)
    parts = ["ПРИМЕРЫ ЗАПРОСОВ"]
    for request, sql in SQL_EXAMPLES:
        parts.append(f"-- {request}:\n{sql.format(**slots)}")
    return "\n\n".join(parts)


def sql_examples(db: Any) -> list[dict[str, Any]]:
    """Примеры для few_shot_examples функции query_sql: запрос руководителя →
    аргументы вызова, с подставленными человеком и показателем из датасета."""
    slots = _example_slots(db.conn)
    return [
        {"request": request, "params": {"sql": sql.format(**slots), "purpose": request}}
        for request, sql in SQL_EXAMPLES
    ]


__all__ = ["COLUMN_DOCS", "SQL_EXAMPLES", "build_schema_doc", "sql_examples"]
