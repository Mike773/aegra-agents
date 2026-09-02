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
    ("v_fact", "depth"): "уровень в дереве показателей: 1 = верхний",
    ("v_fact", "path"): "путь по дереву: 'Родитель > Ребёнок'",
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
    ("v_tree", "share_pct"): "доля влияния ребёнка среди детей родителя, %",
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
- Дерево показателей (v_tree, path, depth) от дат не зависит вовсе.
- element IS NULL — СОБСТВЕННЫЙ ИТОГ показателя. Его и бери за значение
  показателя; разрезы нужны, чтобы объяснить, из чего итог сложился.
- Если у показателя нет строки с element IS NULL (v_metric_person.has_aggregate = 0),
  собственного итога у него НЕТ. Не суммируй разрезы и не выдавай сумму за факт:
  складывать проценты, средние и ранги бессмысленно. Говори про разрезы и
  честно отмечай, что общего значения у показателя нет.
- Разрезов бывают СОТНИ. Не перечисляй их и не тяни целиком: фильтруй по имени
  (LIKE) и всегда ставь LIMIT, либо бери верхушку через metric_card."""

_EXAMPLES = """ПРИМЕРЫ ЗАПРОСОВ
-- Показатели верхнего уровня, отстающие от плана (последние значения серий):
SELECT metric, date, fact, plan, plan_dev_pct, pop_status
FROM v_fact_latest
WHERE person_key = '{person}' AND depth = 1 AND element IS NULL
  AND plan_status = 'хуже_плана'
ORDER BY ABS(plan_dev_pct) DESC;

-- История одного показателя с изменением к прошлому периоду:
SELECT date, fact, plan, ex, rr, pop_change_pct, pop_status
FROM v_fact
WHERE person_key = '{person}' AND metric = '{metric}' AND element IS NULL
ORDER BY period_idx;

-- Худшие разрезы показателя с учётом направления:
SELECT element, fact, plan, plan_dev_pct, plan_status
FROM v_fact_latest
WHERE person_key = '{person}' AND metric = '{metric}' AND element IS NOT NULL
ORDER BY CASE WHEN direction = 'обратная' THEN -fact ELSE fact END
LIMIT 10;

-- Состав ветки дерева с весами влияния:
SELECT child, influent_percent, share_pct, child_depth
FROM v_tree WHERE parent = '{metric}' ORDER BY share_pct DESC;

-- Найти разрез по части имени (разрезов сотни — фильтруй и ограничивай):
SELECT element, date, fact, plan, plan_status
FROM v_fact_latest
WHERE person_key = '{person}' AND metric = '{metric}'
  AND element IS NOT NULL AND ru_lower(element) LIKE '%часть имени%'
LIMIT 20;

-- Есть ли у показателей собственный итог и сколько у них разрезов:
SELECT metric, has_aggregate, n_elements
FROM v_metric_person WHERE person_key = '{person}' AND n_elements > 0;"""


def _views(conn) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def _columns(conn, view: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({view})").fetchall()]


def build_schema_doc(db: Any) -> str:
    """Справка по схеме: вью с колонками, семантика, факты датасета, примеры."""
    conn = db.conn
    views = [v for v in _views(conn) if v != "v_deviation" or _has_rows(conn, "deviation")]

    lines = [
        "СХЕМА ДАННЫХ (SQLite, только чтение)",
        "Обращайся к представлениям ниже — в них уже склеены человек, показатель, "
        "период и готовые вердикты. Базовые таблицы служебные.",
        "",
    ]
    for view in views:
        cols = _columns(conn, view)
        if not cols:
            continue
        lines.append(f"{view}({', '.join(cols)})")
        for col in cols:
            doc = COLUMN_DOCS.get((view, col))
            if doc:
                lines.append(f"  - {col}: {doc}")
        lines.append("")

    lines.append(_facts_block(db))
    lines.append("")
    lines.append(_RULES)
    lines.append("")
    lines.append(_examples_block(conn))
    return "\n".join(lines).strip()


def _has_rows(conn, table: str) -> bool:
    return bool(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _facts_block(db: Any) -> str:
    conn = db.conn
    people = conn.execute(
        "SELECT person_key, fio, is_me FROM person ORDER BY is_me DESC, person_id"
    ).fetchall()
    dates = conn.execute("SELECT date FROM period ORDER BY period_idx").fetchall()
    n_metrics = conn.execute("SELECT COUNT(*) FROM metric").fetchone()[0]
    max_depth = conn.execute("SELECT MAX(depth) FROM metric").fetchone()[0] or 1

    who = ", ".join(
        f"{r['fio'] or r['person_key']} ({r['person_key']}"
        + (", руководитель)" if r["is_me"] else ")")
        for r in people
    )
    parts = [
        "ЧТО В ЭТОЙ БАЗЕ",
        f"- Люди: {who or 'нет'}.",
        f"- Показателей в каталоге: {n_metrics}, глубина дерева: {max_depth}.",
    ]
    if dates:
        parts.append(
            f"- Периоды: {len(dates)}, от {dates[0]['date']} до {dates[-1]['date']} "
            "(у разных показателей последняя дата может отличаться)."
        )
    levels = conn.execute(
        "SELECT DISTINCT level_name, level_order FROM peer_aggregate "
        "WHERE level_name IS NOT NULL ORDER BY level_order"
    ).fetchall()
    if levels:
        parts.append(
            "- Группы сравнения (от узкой к широкой): "
            + ", ".join(r["level_name"] for r in levels)
            + "."
        )
    elif db.has_aggregates:
        parts.append("- Группы сравнения есть, но без названий — называй их обезличенно.")
    if _has_rows(conn, "ranking"):
        parts.append("- Есть места в больших группах коллег (v_peer_latest.rank_raw).")
    if db.has_stars:
        parts.append(
            "- Есть звёзды: именные показатели без чисел (получена/не получена), "
            "их влияющие показатели — в v_star."
        )
    return "\n".join(parts)


def _examples_block(conn) -> str:
    person = conn.execute(
        "SELECT person_key FROM person ORDER BY is_me, person_id LIMIT 1"
    ).fetchone()
    metric = conn.execute(
        "SELECT name FROM metric WHERE depth = 1 ORDER BY metric_id LIMIT 1"
    ).fetchone()
    return _EXAMPLES.format(
        person=person["person_key"] if person else "…",
        metric=metric["name"] if metric else "…",
    )


__all__ = ["COLUMN_DOCS", "build_schema_doc"]
