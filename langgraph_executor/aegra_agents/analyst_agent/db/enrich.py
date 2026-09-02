"""Блоки системного промпта, посчитанные SQL-статистикой.

Датасет на проме — до сотни показателей верхнего уровня и пять уровней вглубь,
поэтому в промпт идёт не сырьё, а выжимка: состав данных, скорборд верхнего
уровня, каталог показателей и пробелы. Каждый блок имеет бюджет символов; то,
чего в датасете нет (звёзды, группы сравнения, разрезы), не упоминается вовсе —
модель не должна звать несуществующее.
"""
from __future__ import annotations

from typing import Any

from .sqlrunner import QueryResult, render_kv, run_template

# Бюджеты блоков (символы). Сумма с запасом укладывается в общий бюджет промпта.
BUDGET_PROFILE = 600
BUDGET_SCOREBOARD = 4500
BUDGET_ZONES = 400
BUDGET_PEER = 700
BUDGET_STARS = 800
BUDGET_GAPS = 600
BUDGET_CATALOG = 14000

# Сколько показателей скорборда печатать полными строками; остальные — счётчиками.
_SCOREBOARD_FULL_ROWS = 30


def _rows(res: QueryResult) -> list[dict[str, Any]]:
    return [dict(zip(res.columns, row)) for row in res.rows]


def _num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{round(value, digits):g}"
    return str(value)


def _pct(value: Any) -> str:
    return "—" if value is None else f"{value:+.1f} %"


def _verdict(value: Any) -> str:
    """Вердикт как слово: подчёркивания в промпте читаются плохо."""
    return "" if not value else str(value).replace("_", " ")


def _plural_metrics(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return "метрика"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return "метрики"
    return "метрик"


def dataset_profile_block(db: Any) -> str:
    """«Состав данных»: люди, показатели, периоды, разрезы, группы, звёзды."""
    res = run_template(db.conn, "enrich_profile")
    if res.error or not res.rows:
        return ""
    p = _rows(res)[0]
    people = db.conn.execute(
        "SELECT fio, person_key, is_me FROM person ORDER BY is_me DESC, person_id"
    ).fetchall()
    who = ", ".join(
        f"{r['fio'] or r['person_key']}" + (" (руководитель)" if r["is_me"] else "")
        for r in people[:12]
    )
    if len(people) > 12:
        who += f" и ещё {len(people) - 12}"

    lines = ["СОСТАВ ДАННЫХ", f"- В анализе: {who}."]
    lines.append(
        f"- Показателей: {p['n_metrics']}, из них верхнего уровня {p['n_roots']}; "
        f"глубина дерева {p['max_depth']}; с планом {p['n_with_plan']}."
    )
    if p["n_periods"]:
        lines.append(
            f"- Периоды ({p['calc_periods'] or 'без указания'}): {p['n_periods']}, "
            f"с {p['first_date']} по {p['last_date']}. У разных показателей "
            "последняя дата может отличаться — смотри дату у самого показателя."
        )
    if p["n_elements"]:
        lines.append(f"- Разрезов: {p['n_elements']}.")
    levels = db.conn.execute(
        "SELECT DISTINCT level_name, total_objects FROM peer_aggregate "
        "WHERE level_name IS NOT NULL ORDER BY level_order"
    ).fetchall()
    if levels:
        named = ", ".join(
            f"{r['level_name']}" + (f" ({r['total_objects']} чел.)" if r["total_objects"] else "")
            for r in levels
        )
        lines.append(f"- Группы сравнения от узкой к широкой: {named}.")
    if db.has_stars:
        n_stars = db.conn.execute("SELECT COUNT(*) FROM metric WHERE is_star = 1").fetchone()[0]
        lines.append(f"- Звёзд (именных показателей без чисел): {n_stars}.")
    return "\n".join(lines)[:BUDGET_PROFILE]


def zones_block(db: Any, *, person_key: str) -> str:
    """Счётчики зон внимания по всему дереву."""
    res = run_template(db.conn, "enrich_zones", person_key=person_key)
    text = render_kv(res)
    if not text.strip():
        return ""
    labels = {
        "n_below_plan": "хуже плана",
        "n_declining": "ухудшаются",
        "n_anomaly": "аномалии",
        "n_peer_worse": "хуже коллег",
        "n_group_worse": "хуже группы",
        "n_above_plan": "лучше плана",
        "n_improving": "улучшаются",
    }
    row = _rows(res)[0]
    parts = [f"{labels[k]} — {row[k]}" for k in labels if row.get(k)]
    if not parts:
        return ""
    return ("ЗОНЫ ВНИМАНИЯ (верхний уровень): " + "; ".join(parts) + ".")[:BUDGET_ZONES]


def scoreboard_block(db: Any, *, person_key: str, max_depth: int = 1) -> str:
    """Скорборд: последнее значение КАЖДОЙ серии со своей датой."""
    res = run_template(
        db.conn, "enrich_scoreboard", max_rows=400,
        person_key=person_key, max_depth=max_depth, row_limit=400,
    )
    rows = _rows(res)
    if not rows:
        return ""
    lines = ["ПОКАЗАТЕЛИ ВЕРХНЕГО УРОВНЯ (последнее значение каждого, со своей датой)"]
    for r in rows[:_SCOREBOARD_FULL_ROWS]:
        bits = [f"{r['metric']} ({r['date']}): факт {_num(r['fact'])}"]
        if r["unit"]:
            bits[0] += f" {r['unit']}"
        if r["plan"] is not None:
            bits.append(f"план {_num(r['plan'])}")
        if r["ex"] is not None:
            bits.append(f"выполнение {_num(r['ex'])} %")
        if r["plan_status"]:
            v = _verdict(r["plan_status"])
            bits.append(v + (f" на {abs(r['plan_dev_pct']):.1f} %" if r["plan_dev_pct"] else ""))
        if r["pop_status"]:
            v = _verdict(r["pop_status"])
            bits.append(
                v + (f" ({_pct(r['pop_change_pct'])} к прошлому)" if r["pop_change_pct"] else "")
            )
        if r["trend_status"]:
            bits.append(f"тренд — {_verdict(r['trend_status'])}")
        if r["rel_status"]:
            bits.append(_verdict(r["rel_status"]))
        tail = []
        if r["n_children"]:
            tail.append(f"детей {r['n_children']}")
        if r["n_elements"]:
            tail.append(f"разрезов {r['n_elements']}")
        line = "- " + ", ".join(bits) + (f" [{'; '.join(tail)}]" if tail else "")
        lines.append(line)

    rest = rows[_SCOREBOARD_FULL_ROWS:]
    if rest:
        by_zone: dict[str, list[str]] = {}
        for r in rest:
            by_zone.setdefault(r["zone"], []).append(r["metric"])
        chunks = []
        for zone, names in by_zone.items():
            shown = ", ".join(names[:6])
            more = f" и ещё {len(names) - 6}" if len(names) > 6 else ""
            chunks.append(f"{zone} — {len(names)} ({shown}{more})")
        lines.append(f"- Остальные {len(rest)}: " + "; ".join(chunks) + ".")
    return _trim("\n".join(lines), BUDGET_SCOREBOARD)


def catalog_block(db: Any, *, budget_chars: int = BUDGET_CATALOG) -> str:
    """Каталог: все показатели 1-го и 2-го уровня.

    У показателя 2-го уровня печатаются счётчики потомков по слоям
    («10 метрик 3-го, 20 4-го») — имена глубже второго уровня в промпт не идут,
    их модель достаёт через SQL или карточку показателя.
    """
    res = run_template(db.conn, "enrich_catalog", max_rows=1000, max_level=2, row_limit=1000)
    rows = _rows(res)
    if not rows:
        return ""
    counts = run_template(
        db.conn, "enrich_deeper_counts", max_rows=5000, max_level=2, row_limit=5000
    )
    deeper: dict[str, list[tuple[int, int]]] = {}
    for r in _rows(counts):
        deeper.setdefault(r["ancestor"], []).append((r["descendant_depth"], r["n"]))

    children: dict[str, list[dict[str, Any]]] = {}
    roots: list[dict[str, Any]] = []
    for r in rows:
        if r["depth"] == 1:
            roots.append(r)
        else:
            children.setdefault(r["parent_name"], []).append(r)

    def describe(r: dict[str, Any]) -> str:
        tags = []
        if r["unit"]:
            tags.append(str(r["unit"]))
        if r["direction"]:
            tags.append(str(r["direction"]))
        tags.append("план есть" if r["has_plan"] else "плана нет")
        if r["n_elements"]:
            tags.append(f"разрезов {r['n_elements']}")
        if r["kind"] and r["kind"] != "уровень":
            tags.append(str(r["kind"]))
        if r["influent_percent"]:
            tags.append(f"влияние {_num(r['influent_percent'], 0)} %")
        return f"{r['name']} [{', '.join(tags)}]"

    def deeper_text(name: str) -> str:
        levels = sorted(deeper.get(name, []))
        if not levels:
            return ""
        parts = [f"{n} {_plural_metrics(n)} {depth}-го" for depth, n in levels]
        return f" ({', '.join(parts)})"

    lines = [
        "КАТАЛОГ ПОКАЗАТЕЛЕЙ (уровни 1-2; в скобках у второго уровня — сколько "
        "показателей лежит ниже по слоям, их имена доступны через SQL)"
    ]
    for root in roots:
        lines.append(f"- {describe(root)}{deeper_text(root['name']) if not children.get(root['name']) else ''}")
        for child in children.get(root["name"], []):
            lines.append(f"  - {describe(child)}{deeper_text(child['name'])}")
    return _trim("\n".join(lines), budget_chars)


def peer_block(db: Any, *, person_key: str) -> str:
    """Позиция в группах сравнения — только при наличии агрегатов."""
    res = run_template(db.conn, "enrich_peer", person_key=person_key)
    rows = _rows(res)
    if not rows:
        return ""
    by_level: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_level.setdefault(r["level_name"], []).append(r)
    lines = ["СРАВНЕНИЕ С КОЛЛЕГАМИ (от узкой группы к широкой)"]
    for level, items in by_level.items():
        total = items[0]["total_objects"]
        head = f"- {level}" + (f" ({total} чел.)" if total else "") + ": "
        bits = []
        for r in items[:4]:
            piece = f"{r['metric']} — среднее {_num(r['mean_fact'])}"
            if r["top20_mean_fact"] is not None:
                piece += f", у сильнейших {_num(r['top20_mean_fact'])}"
            if r["hit_rate"] is not None:
                piece += f", план выполняет {_num(r['hit_rate'], 0)} % коллег"
            if r["rank_raw"]:
                piece += f", место {r['rank_raw']}"
            bits.append(piece)
        more = f" и ещё {len(items) - 4} показателей" if len(items) > 4 else ""
        lines.append(head + "; ".join(bits) + more + ".")
    return _trim("\n".join(lines), BUDGET_PEER)


def stars_block(db: Any, *, person_key: str | None = None) -> str:
    """Звёзды и их влияющие показатели — только если звёзды есть."""
    if not db.has_stars:
        return ""
    if person_key is None:
        row = db.conn.execute(
            "SELECT person_key FROM person ORDER BY is_me, person_id LIMIT 1"
        ).fetchone()
        person_key = row["person_key"] if row else None
    if person_key is None:
        return ""
    rows = _rows(run_template(db.conn, "enrich_stars", person_key=person_key))
    if not rows:
        return ""
    by_star: dict[str, list[dict[str, Any]]] = {}
    received: dict[str, Any] = {}
    dates: dict[str, Any] = {}
    for r in rows:
        by_star.setdefault(r["star"], []).append(r)
        received[r["star"]] = r["received"]
        dates[r["star"]] = r["star_date"]
    lines = ["ЗВЁЗДЫ"]
    for star, items in by_star.items():
        status = "получена" if received[star] else "не получена"
        lines.append(f"- {star} ({dates[star]}): {status}.")
        missed = [
            f"{r['child']} — факт {_num(r['child_fact'])} при плане {_num(r['child_plan'])}"
            for r in items
            if r["child"] and r["child_plan_status"] == "хуже_плана"
        ]
        if missed:
            lines.append("  ниже плана: " + "; ".join(missed[:5]) + ".")
    return _trim("\n".join(lines), BUDGET_STARS)


def gaps_block(db: Any, *, person_key: str) -> str:
    """Пробелы данных: без плана, единственный период, нет справки из wiki."""
    res = run_template(db.conn, "enrich_gaps", person_key=person_key)
    rows = _rows(res)
    if not rows:
        return ""
    r = rows[0]
    parts = []
    if r["n_no_plan"]:
        parts.append(f"без плана — {r['n_no_plan']} ({r['names_no_plan']})")
    if r["n_single_period"]:
        parts.append(
            f"только один период — {r['n_single_period']} ({r['names_single_period']})"
        )
    if not parts:
        return ""
    return _trim("ПРОБЕЛЫ ДАННЫХ: " + "; ".join(parts) + ".", BUDGET_GAPS)


def _trim(text: str, budget: int) -> str:
    """Обрезает по границе строки, чтобы не рвать её посередине."""
    if len(text) <= budget:
        return text
    lines = text.splitlines()
    out: list[str] = []
    size = 0
    for line in lines:
        if size + len(line) + 1 > budget:
            break
        out.append(line)
        size += len(line) + 1
    return "\n".join(out)


def enrichment_block(db: Any, *, person_key: str, budget_chars: int = 9000) -> str:
    """Сводный блок обогащения: профиль, зоны, скорборд, коллеги, звёзды, пробелы.

    При нехватке бюджета режется с хвоста приоритета: сначала пробелы, потом
    звёзды/коллеги, в последнюю очередь — скорборд.
    """
    blocks = [
        dataset_profile_block(db),
        zones_block(db, person_key=person_key),
        scoreboard_block(db, person_key=person_key),
        peer_block(db, person_key=person_key),
        stars_block(db, person_key=person_key),
        gaps_block(db, person_key=person_key),
    ]
    blocks = [b for b in blocks if b.strip()]
    text = "\n\n".join(blocks)
    while len(text) > budget_chars and len(blocks) > 1:
        blocks.pop()
        text = "\n\n".join(blocks)
    return _trim(text, budget_chars)


__all__ = [
    "catalog_block",
    "dataset_profile_block",
    "enrichment_block",
    "gaps_block",
    "peer_block",
    "scoreboard_block",
    "stars_block",
    "zones_block",
]
