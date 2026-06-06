"""Типизированные инструменты агента-аналитика.

Агент НИКОГДА не пишет SQL — он только вызывает эти инструменты с параметрами.
Каждый инструмент внутри выполняет параметрический запрос к in-memory SQLite
(метрики + производная аналитика) либо семантический поиск по in-memory индексу
эмбеддингов (EmbeddingIndex).

Выдача намеренно компактная: пустые поля убираются, числа округляются —
контекст чат-модели ограничен.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.tools import StructuredTool

from . import analytics
from .sqlite_store import SqliteStore
from .store_cache import EmbeddingIndex

def _blank_to_none(value: Any) -> Any:
    """Пустая/пробельная строка → None.

    Некоторые модели присылают аргумент как "" вместо опускания, если считают
    фильтр ненужным. Без нормализации это превратится в WHERE col = '' и молча
    даст 0 строк.
    """
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _dump(obj: Any) -> str:
    """JSON-фолбэк: используется только в _safe при ошибке рендера. Основная
    выдача инструментов — человекочитаемый рендер ниже."""
    return json.dumps(obj, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------- #
# Человекочитаемый рендер для GigaChat: Markdown-таблицы и фразовые списки вместо
# сырого JSON. Русские самоописательные метки, единицы инлайн, без дублирующих
# сырых чисел (trend 'рост/падение' и сырой zscore выкинуты — их смысл уже в
# вердиктах trend_status/peer_status). Каждый рендер вызывается через _safe:
# при любой ошибке отдаём прежний JSON, чтобы баг рендера не ломал инструмент.
# --------------------------------------------------------------------------- #
def _unit(measure_type: Any) -> str | None:
    u = (measure_type or "").strip() if isinstance(measure_type, str) else ""
    return u or None


def _fmt_num(value: Any, unit: str | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        value = round(value, 2)
        if value == int(value):
            value = int(value)
    s = str(value)
    return f"{s} {unit}" if unit else s


def _pct(value: Any) -> str:
    return "" if value is None else f"{_fmt_num(value)} %"


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ") if value not in (None, "") else ""


def _md_table(rows: list[dict[str, Any]], columns: list[tuple[str, Any]]) -> str:
    """Markdown-таблица. columns: список (заголовок, fn(row)->значение). Столбцы,
    пустые во всех строках, выкидываются — один набор колонок подходит всем
    row-инструментам (у compare отпадут peer-колонки, у rank — динамика и т.п.)."""
    if not rows:
        return ""
    cells = [[fn(r) for (_, fn) in columns] for r in rows]
    keep = [
        i
        for i in range(len(columns))
        if any(cells[r][i] not in (None, "") for r in range(len(rows)))
    ]
    if not keep:
        return ""
    headers = [columns[i][0] for i in keep]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in range(len(rows)):
        out.append("| " + " | ".join(_md_cell(cells[r][i]) for i in keep) + " |")
    return "\n".join(out)


def _flat_columns() -> list[tuple[str, Any]]:
    def plain(key: str) -> Any:
        return lambda r: "" if r.get(key) is None else str(r.get(key))

    def fact_with_unit(r: dict[str, Any]) -> str:
        return _fmt_num(r.get("fact"), _unit(r.get("measure_type")))

    def rank_cell(r: dict[str, Any]) -> str:
        if r.get("peer_rank") is None:
            return ""
        pc = r.get("peer_count")
        return f"{r['peer_rank']}/{pc}" if pc else str(r["peer_rank"])

    return [
        ("сотрудник", plain("person_fio")),
        ("метрика", plain("metric_name")),
        ("разрез", plain("element")),
        ("период", plain("date")),
        ("факт", fact_with_unit),
        ("план_статус", plain("plan_status")),
        ("откл_от_плана_%", lambda r: _pct(r.get("plan_dev_pct"))),
        ("бенчмарк_статус", plain("benchmark_status")),
        ("динамика", plain("pop_status")),
        ("изм_к_прошлому_%", lambda r: _pct(r.get("pop_change_pct"))),
        ("тренд", plain("trend_status")),
        ("ранг_среди_коллег", rank_cell),
        ("vs_коллеги", plain("peer_status")),
        ("аномалия", lambda r: "да" if r.get("is_anomaly") else ""),
    ]


def _meta_lines(result: dict[str, Any]) -> list[str]:
    """Не-табличные пометки результата (усечение, разрезы вместо агрегата)."""
    lines: list[str] = []
    rows = result.get("rows", [])
    if result.get("разрезы_вместо_агрегата"):
        lines.append(str(result["разрезы_вместо_агрегата"]))
    if result.get("truncated"):
        shown = result.get("count", len(rows))
        lines.append(
            f"(показаны первые {shown} строк — самые значимые, по убыванию; это НЕ "
            "полное число случаев. Сузь фильтрами metric/date/element.)"
        )
    elif rows:
        lines.append(f"Строк: {len(rows)}.")
    return lines


def _render_rows(result: dict[str, Any]) -> str:
    if result.get("error"):
        return _render_error(result)
    rows = result.get("rows", [])
    meta = _meta_lines(result)
    if not rows:
        return "\n".join(meta) or "Ничего не найдено по заданным фильтрам."
    table = _md_table(rows, _flat_columns())
    return "\n".join(meta + ([table] if table else []))


def _render_error(result: dict[str, Any]) -> str:
    parts = [f"Ошибка: {result.get('error')}"]
    if result.get("hint"):
        parts.append(f"Подсказка: {result['hint']}")
    for k in ("available_elements", "разрезы"):
        if result.get(k):
            parts.append(f"{k}: {', '.join(map(str, result[k]))}")
    return "\n".join(parts)


def _render_tree(result: dict[str, Any]) -> str:
    if result.get("error"):
        return _render_error(result)
    rows = result.get("rows", [])
    if not rows:
        return "Дерево пустое."
    lines: list[str] = []
    if result.get("разрезы_вместо_агрегата"):
        lines.append(str(result["разрезы_вместо_агрегата"]))
    current_person = object()
    multi_person = len({r.get("person_fio") for r in rows}) > 1
    for r in rows:
        if multi_person and r.get("person_fio") != current_person:
            current_person = r.get("person_fio")
            lines.append(f"Сотрудник: {current_person}")
        depth = r.get("depth") or 1
        indent = "  " * (depth - 1)
        label = r.get("metric_name")
        if r.get("element"):
            label = f"{label} [{r['element']}]"
        parts = [f"{label}: {_fmt_num(r.get('fact'), _unit(r.get('measure_type')))}"]
        if r.get("influent_percent") is not None:
            parts.append(f"влияние {_fmt_num(r['influent_percent'])}%")
        elif r.get("influent_percent_missing"):
            rel = r.get("inferred_relation")
            parts.append("влияние н/д" + (f" (связь: {rel})" if rel else ""))
        if r.get("plan_status"):
            dev = r.get("plan_dev_pct")
            parts.append(
                f"план: {r['plan_status']}"
                + (f" ({_fmt_num(dev)}%)" if dev is not None else "")
            )
        if r.get("pop_status"):
            parts.append(f"динамика: {r['pop_status']}")
        lines.append(f"{indent}- " + " | ".join(parts))
    return "\n".join(lines)


def _render_schema(ov: dict[str, Any]) -> str:
    lines = ["Состав датасета:"]
    lines.append(f"- Периоды: {', '.join(ov.get('dates') or [])}")
    metrics = ov.get("metrics") or []
    mparts = []
    for m in metrics:
        tag = "agg+" if m.get("has_aggregate", True) else "agg-"
        u = _unit(m.get("measure_type")) or "—"
        elems = m.get("elements") or []
        e = f" {{разрезы: {', '.join(elems)}}}" if elems else ""
        mparts.append(f"{m['metric_name']} ({m.get('metric_type')}, {u}) [{tag}]{e}")
    lines.append(f"- Метрики ({len(metrics)}): " + "; ".join(mparts))
    if ov.get("elements"):
        lines.append(f"- Разрезы (element): {', '.join(ov['elements'])}")
    people = ov.get("people") or []
    managers = sum(1 for p in people if p.get("person_is_me"))
    lines.append(f"- Людей: {len(people)} ({managers} рук. + {len(people) - managers} сотр.)")
    return "\n".join(lines)


def _render_describe(d: dict[str, Any]) -> str:
    if d.get("error"):
        return _render_error(d)
    dir_hint = "меньше=лучше" if d.get("metric_type") == "обратная" else "больше=лучше"
    return (
        f"Метрика «{d.get('metric_name')}»: "
        f"{d.get('metric_description') or 'описание отсутствует'}\n"
        f"Тип: {d.get('metric_type')} ({dir_hint}). "
        f"Единица: {_unit(d.get('measure_type')) or '—'}. "
        f"Период расчёта: {d.get('calc_period') or '—'}."
    )


def _render_people(people: list[dict[str, Any]]) -> str:
    if not people:
        return "Людей по фильтрам не найдено."
    cols = [
        ("сотрудник", lambda r: r.get("person_fio") or ""),
        ("должность", lambda r: r.get("person_post") or ""),
        ("подразделение", lambda r: r.get("person_depart") or ""),
        ("роль", lambda r: "руководитель" if r.get("person_is_me") else "сотрудник"),
    ]
    return f"Людей: {len(people)}.\n" + _md_table(people, cols)


def _render_resolve(result: dict[str, Any]) -> str:
    if result.get("error"):
        return _render_error(result)
    kind = result.get("kind")
    matches = result.get("matches") or []
    if not matches:
        return f"Совпадений ({kind}) не найдено."
    if kind == "person":
        return _render_people(matches)
    lines = [f"Кандидаты ({kind}), по убыванию похожести:"]
    for i, m in enumerate(matches, 1):
        name = m.get("canonical") or m.get("content")
        sim = m.get("similarity")
        lines.append(f"{i}. {name}" + (f" (похожесть {sim})" if sim is not None else ""))
    return "\n".join(lines)


def _render_related(metric: str, edges: list[dict[str, Any]]) -> str:
    if not edges:
        return f"Связанных метрик для «{metric}» в графе нет."
    cols = [
        ("источник", lambda r: r.get("source") or ""),
        ("связь", lambda r: r.get("relation") or ""),
        ("цель", lambda r: r.get("target") or ""),
        ("сила", lambda r: r.get("strength") or ""),
        ("обоснование", lambda r: r.get("rationale") or ""),
    ]
    return (
        f"Связи метрики «{metric}» (ЭВРИСТИКА из названий/описаний, "
        "не из значений — помечай как предположительные):\n" + _md_table(edges, cols)
    )


def _render_aggregate(result: dict[str, Any]) -> str:
    if result.get("error"):
        return _render_error(result)
    groups = result.get("groups") or []
    head = (
        f"Агрегация «{result.get('metric')}» по {result.get('group_by')} "
        f"({len(groups)} групп):"
    )
    if not groups:
        return head + "\nпусто."
    cols = [
        ("группа", lambda r: r.get("grp") or ""),
        ("avg", lambda r: _fmt_num(r.get("avg"))),
        ("min", lambda r: _fmt_num(r.get("min"))),
        ("max", lambda r: _fmt_num(r.get("max"))),
        ("sum", lambda r: _fmt_num(r.get("sum"))),
        ("кол-во", lambda r: _fmt_num(r.get("n") if r.get("n") is not None else r.get("count"))),
    ]
    return head + "\n" + _md_table(groups, cols)


def _render_summary(s: dict[str, Any]) -> str:
    scope = s.get("scope") or {}
    lines = [
        "Сводка по датасету:",
        f"- Охват: {scope.get('people')} чел ({scope.get('employees')} сотр.), "
        f"метрик {scope.get('metric_types')}, периоды {', '.join(scope.get('dates') or [])}.",
        f"- Последний период: {s.get('latest_date')}.",
    ]
    by_metric = s.get("by_metric_latest") or []
    if by_metric:
        cols = [
            ("метрика", lambda r: r.get("metric") or ""),
            ("тип", lambda r: r.get("metric_type") or ""),
            ("сред_факт", lambda r: _fmt_num(r.get("avg_fact"))),
            ("хуже_плана", lambda r: _fmt_num(r.get("below_plan"))),
            ("аномалий", lambda r: _fmt_num(r.get("anomalies"))),
        ]
        lines.append("Ключевые метрики на последнем периоде:")
        lines.append(_md_table(by_metric, cols))
    anomalies = s.get("top_anomalies_latest") or []
    if anomalies:
        cols = [
            ("сотрудник", lambda r: r.get("person_fio") or ""),
            ("метрика", lambda r: r.get("metric_name") or ""),
            ("разрез", lambda r: r.get("element") or ""),
            ("факт", lambda r: _fmt_num(r.get("fact"))),
            ("z-score", lambda r: _fmt_num(r.get("zscore"))),
        ]
        lines.append("Топ аномалий:")
        lines.append(_md_table(anomalies, cols))
    tc = s.get("trend_counts_level1") or {}
    if tc:
        lines.append(
            "Динамика метрик 1-го уровня: "
            + ", ".join(f"{k}: {v}" for k, v in tc.items())
        )
    return "\n".join(lines)


def _safe(render: Any, result: Any) -> str:
    """Рендер с безопасным fallback на JSON при любой ошибке/пустом выводе."""
    try:
        out = render(result)
        return out if out else _dump(result)
    except Exception:
        return _dump(result)


def build_tools(
    store: SqliteStore,
    index: EmbeddingIndex,
    embed_query: Callable[[str], list[float]],
) -> list[StructuredTool]:
    """Собирает инструменты, замкнутые на конкретные хранилища.

    Семантический поиск (resolve_entity) идёт по in-memory индексу эмбеддингов,
    уже загруженному только для текущего направления — данные других
    направлений в него не попадают.
    """

    def _unknown_metric(metric: str) -> str | None:
        if store.metric_type_of(metric) is not None:
            return None
        return _render_error(
            {
                "error": f"Метрика '{metric}' не найдена. Здесь нужно ТОЧНОЕ "
                "название метрики — не человек, не продукт, не произвольный текст.",
                "hint": "человека передавай в person, продукт — в element; "
                "точные названия метрик смотри в schema_overview или подбери "
                "через resolve_entity(kind='metric'). Если фильтр по метрике не "
                "нужен — просто не передавай этот аргумент.",
            }
        )

    def _unknown_person(person: str | None) -> str | None:
        if person is None or str(person).strip() == "":
            return None
        text = str(person).strip()
        people = store.list_people()
        if text.isdigit():
            found = any(str(p["person_tabnum"]) == text for p in people)
        else:
            needle = text.lower()
            found = any(needle in (p["person_fio"] or "").lower() for p in people)
        if found:
            return None
        return _render_error(
            {
                "error": f"Человек '{person}' не найден. Здесь нужно ТОЧНОЕ ФИО "
                "(или его часть) либо табельный номер сотрудника — не метрика, "
                "не продукт, не произвольный текст.",
                "hint": "список людей смотри в list_people, неточное имя "
                "разрешай через resolve_entity(kind='person'). Если фильтр по "
                "человеку не нужен — просто не передавай аргумент person.",
            }
        )

    def schema_overview() -> str:
        """Обзор загруженного датасета: метрики с их типами и единицами, значения
        element (продукты/разрезы), люди и диапазон дат. Семантику метрик не
        предполагай — смотри по факту."""
        return _safe(_render_schema, store.schema_overview())

    def resolve_entity(text: str, kind: str) -> str:
        """Разрешает нечёткую формулировку в каноничное имя сущности.
        kind: 'metric' — название метрики (поиск по названиям и описаниям),
        'element' — значение поля element (продукт/разрез), 'person' — сотрудник.
        Используй, когда метрика/продукт/человек названы неточно или описательно."""
        kind = (kind or "").strip().lower()
        if kind == "person":
            matches = store.list_people(name_query=text)[:10]
            return _safe(_render_resolve, {"kind": "person", "matches": matches})
        if kind == "metric":
            search_kinds = ["metric_name", "metric_description"]
        elif kind == "element":
            search_kinds = ["element"]
        else:
            return _render_error(
                {"error": "kind должен быть 'metric', 'element' или 'person'"}
            )
        vector = embed_query(text)
        matches = index.search(vector, kinds=search_kinds, top_k=5)
        return _safe(_render_resolve, {"kind": kind, "matches": matches})

    def describe_metric(metric: str) -> str:
        """Описание метрики, её тип ('прямая' — чем больше, тем лучше; 'обратная' —
        чем меньше, тем лучше), единица измерения и период расчёта. Вызывай перед
        интерпретацией значений: направление метрики критично."""
        result = store.describe_metric(metric)
        if result is None:
            return _render_error(
                {"error": f"Метрика '{metric}' не найдена", "hint": "используй resolve_entity"}
            )
        return _safe(_render_describe, result)

    def get_metric(
        metric: str,
        person: str | None = None,
        element: str | None = None,
        date: str | None = None,
    ) -> str:
        """Значения метрики (fact/plan/benchmark) + аналитика по строке.
        Динамику/позицию читай по ВЕРДИКТАМ (trend_status/pop_status/peer_status),
        а не по знаку: для 'обратной' метрики рост значения = ухудшение.
        person — ФИО/табельный; date — YYYY-MM-DD; element не указан = агрегат
        (для agg--метрик вернутся все разрезы с пометкой
        'разрезы_вместо_агрегата')."""
        person = _blank_to_none(person)
        element = _blank_to_none(element)
        date = _blank_to_none(date)
        unknown = _unknown_metric(metric) or _unknown_person(person)
        if unknown:
            return unknown
        return _safe(
            _render_rows,
            store.get_metric(metric, person=person, element=element, date=date),
        )

    def compare(
        metric: str,
        person: str | None = None,
        element: str | None = None,
        dates: list[str] | None = None,
    ) -> str:
        """Динамика метрики по периодам (pop_change_pct, trend + вердикты
        pop_status/trend_status) для ОДНОГО человека (person обязателен). Оценивай
        по *_status, а не по знаку. element не указан = агрегат (agg--метрики →
        все разрезы). Чтобы найти, у кого сильнее спад/рост по всем — find_flags."""
        person = _blank_to_none(person)
        element = _blank_to_none(element)
        unknown = _unknown_metric(metric) or _unknown_person(person)
        if unknown:
            return unknown
        return _safe(
            _render_rows,
            store.compare(metric, person=person, element=element, dates=dates),
        )

    def rank(
        metric: str,
        date: str,
        element: str | None = None,
        post: str | None = None,
    ) -> str:
        """Рейтинг сотрудников по метрике на конкретный период. Направление уже
        учтено: peer_rank=1 — лучший. element не указан = агрегат по сотруднику;
        post — фильтр по должности. Если у метрики нет агрегата (agg- в
        составе датасета), тула вернёт error со списком доступных element —
        передай element и повтори."""
        element = _blank_to_none(element)
        post = _blank_to_none(post)
        unknown = _unknown_metric(metric)
        if unknown:
            return unknown
        return _safe(_render_rows, store.rank(metric, date, element=element, post=post))

    def aggregate(
        metric: str,
        group_by: str,
        date: str | None = None,
        element: str | None = None,
    ) -> str:
        """Агрегация значений метрики (avg/min/max/sum/count) по группам.
        group_by: 'person' | 'element' | 'date' | 'post'."""
        date = _blank_to_none(date)
        element = _blank_to_none(element)
        unknown = _unknown_metric(metric)
        if unknown:
            return unknown
        return _safe(
            _render_aggregate, store.aggregate(metric, group_by, date=date, element=element)
        )

    def metric_tree(
        metric: str | None = None,
        person: str | None = None,
        date: str | None = None,
    ) -> str:
        """Иерархия метрики со всеми дочерними child_metrics и аналитикой по
        каждому узлу (plan_status, trend_status, pop_status, influent_percent и
        др.). ОДИН вызов раскладывает метрику на компоненты — не дёргай get_metric
        по каждому. Задавай metric, person и date (иначе строк много). agg--метрика
        → корнями станут её разрезы (пометка 'разрезы_вместо_агрегата')."""
        metric = _blank_to_none(metric)
        person = _blank_to_none(person)
        date = _blank_to_none(date)
        unknown = (
            _unknown_metric(metric) if metric is not None else None
        ) or _unknown_person(person)
        if unknown:
            return unknown
        return _safe(
            _render_tree, store.metric_tree(name=metric, person=person, date=date)
        )

    def list_people(
        role: str | None = None,
        post: str | None = None,
        depart: str | None = None,
        name_query: str | None = None,
    ) -> str:
        """Список людей в датасете. role: 'me' (руководитель) | 'employee'.
        name_query — подстрока ФИО для поиска."""
        role = _blank_to_none(role)
        post = _blank_to_none(post)
        depart = _blank_to_none(depart)
        name_query = _blank_to_none(name_query)
        return _safe(
            _render_people,
            store.list_people(role=role, post=post, depart=depart, name_query=name_query),
        )

    def find_flags(
        kind: str,
        date: str | None = None,
        metric: str | None = None,
        element: str | None = None,
    ) -> str:
        """Предрассчитанные проблемные/заметные строки, отсортированы по силе
        (первая — самая значимая). kind:
        'anomaly' — выбросы (|z-score| выше порога);
        'below_plan' — хуже плана (с учётом направления); 'above_plan' — лучше плана;
        'declining' — динамика ухудшилась (trend_status='ухудшение'); 'improving' —
        улучшилась; 'trend' — любое движение значения без оценки хорошо/плохо.
        «просела/упала/ухудшилась динамика» = 'declining', «хуже плана/отстаёт» =
        'below_plan' (это РАЗНЫЕ вопросы). Фокусируй фильтрами metric/date/element."""
        date = _blank_to_none(date)
        metric = _blank_to_none(metric)
        element = _blank_to_none(element)
        if metric and store.metric_type_of(metric) is None:
            metric = None
        return _safe(
            _render_rows, store.find_flags(kind, date=date, metric=metric, element=element)
        )

    def analytics_summary() -> str:
        """Стартовая детерминированная сводка: охват датасета, средние по ключевым
        метрикам на последнем периоде, топ аномалий, счётчики трендов."""
        return _safe(_render_summary, analytics.build_summary(store))

    def related_metrics(metric: str) -> str:
        """Связанные по СМЫСЛУ метрики (граф выведен LLM из названий/описаний, не
        из значений). Возвращает рёбра с relation ('опережающая→запаздывающая'/
        'компонент'/'смежная'/'влияет_на'), strength и rationale. Полезно, когда у
        дочерней метрики нет influent_percent. Это ЭВРИСТИКА — помечай связи как
        предположительные, не как факт."""
        metric = _blank_to_none(metric)
        if metric is None:
            return _render_error(
                {"error": "related_metrics требует точное название метрики"}
            )
        unknown = _unknown_metric(metric)
        if unknown:
            return unknown
        edges = store.related_metrics(metric)
        return _safe(lambda e: _render_related(metric, e), edges)

    specs = [
        (schema_overview, "schema_overview"),
        (resolve_entity, "resolve_entity"),
        (describe_metric, "describe_metric"),
        (get_metric, "get_metric"),
        (compare, "compare"),
        (rank, "rank"),
        (aggregate, "aggregate"),
        (metric_tree, "metric_tree"),
        (list_people, "list_people"),
        (find_flags, "find_flags"),
        (analytics_summary, "analytics_summary"),
        (related_metrics, "related_metrics"),
    ]

    return [
        StructuredTool.from_function(func=func, name=name, description=func.__doc__)
        for func, name in specs
    ]
