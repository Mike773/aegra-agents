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

_ROW_KEYS = (
    "person_fio",
    "metric_name",
    "metric_type",
    "date",
    "element",
    "fact",
    "plan",
    "benchmark",
    "plan_status",
    "plan_dev_pct",
    "benchmark_status",
    "wow_change_pct",
    "wow_status",
    "trend",
    "trend_status",
    "peer_rank",
    "peer_count",
    "zscore",
    "peer_status",
    "is_anomaly",
)


def _clean(value: Any) -> Any:
    return round(value, 2) if isinstance(value, float) else value


# JSON/Python-репрезентации «значение не задано», которые модель присылает
# строкой вместо опускания аргумента.
_UNSET_TOKENS = {"", "null", "none", "nil"}


def _blank_to_none(value: Any) -> Any:
    """Пустая/пробельная строка или строковый литерал null/none/nil → None.

    Некоторые модели присылают аргумент как "" или "null" вместо опускания, если
    считают фильтр ненужным. Без нормализации это превратится в WHERE col = 'null'
    и молча даст 0 строк. Набор токенов узкий (null/none/nil) — это не реальные
    имена метрик/продуктов/людей в домене, так что валидные значения не заденет.
    """
    if isinstance(value, str) and value.strip().casefold() in _UNSET_TOKENS:
        return None
    return value


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: _clean(row[k]) for k in _ROW_KEYS if row.get(k) is not None}


def _strip(row: dict[str, Any]) -> dict[str, Any]:
    return {k: _clean(v) for k, v in row.items() if v is not None}


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _pack(result: dict[str, Any], curated: bool = True) -> str:
    packed = {k: v for k, v in result.items() if k != "rows"}
    transform = _compact_row if curated else _strip
    packed["rows"] = [transform(r) for r in result.get("rows", [])]
    if packed.pop("truncated", False):
        # Выборка усечена по лимиту строк. Голый count тут равен лимиту
        # пагинации — модель путала его с «числом случаев». Заменяем явной
        # пометкой, чтобы это нельзя было принять за итоговый счёт.
        shown = packed.pop("count", len(packed["rows"]))
        packed["выборка"] = (
            f"показаны первые {shown} строк (самые значимые); в данных есть и "
            "другие — это НЕ полное число случаев. Чтобы охватить остальное, "
            "сузь запрос фильтрами metric/date/element."
        )
    return _dump(packed)


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
        return _dump(
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
        return _dump(
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
        return _dump(store.schema_overview())

    def resolve_entity(text: str, kind: str) -> str:
        """Разрешает нечёткую формулировку в каноничное имя сущности.
        kind: 'metric' — название метрики (поиск по названиям и описаниям),
        'element' — значение поля element (продукт/разрез), 'person' — сотрудник.
        Используй, когда метрика/продукт/человек названы неточно или описательно."""
        kind = (kind or "").strip().lower()
        if kind == "person":
            return _dump(
                {"kind": "person", "matches": store.list_people(name_query=text)[:10]}
            )
        if kind == "metric":
            search_kinds = ["metric_name", "metric_description"]
        elif kind == "element":
            search_kinds = ["element"]
        else:
            return _dump({"error": "kind должен быть 'metric', 'element' или 'person'"})
        vector = embed_query(text)
        return _dump(
            {
                "kind": kind,
                "matches": index.search(vector, kinds=search_kinds, top_k=5),
            }
        )

    def describe_metric(metric: str) -> str:
        """Описание метрики, её тип ('прямая' — чем больше, тем лучше; 'обратная' —
        чем меньше, тем лучше), единица измерения и период расчёта. Вызывай перед
        интерпретацией значений: направление метрики критично."""
        result = store.describe_metric(metric)
        if result is None:
            return _dump(
                {"error": f"Метрика '{metric}' не найдена", "hint": "используй resolve_entity"}
            )
        return _dump(result)

    def get_metric(
        metric: str,
        person: str | None = None,
        element: str | None = None,
        date: str | None = None,
    ) -> str:
        """Значения метрики (fact/plan/benchmark) плюс производная аналитика:
        статусы отклонений, динамика, тренд, peer-ранг, флаг аномалии.
        Динамику и позицию читай по ВЕРДИКТАМ с учётом направления:
        trend_status / wow_status ('улучшение'/'ухудшение'/'стабильно') и
        peer_status ('лучше_коллег'/'хуже_коллег'). Сырые trend ('рост'/'падение'),
        wow_change_pct и знак zscore — это направление ЗНАЧЕНИЯ, а не «хорошо/плохо»:
        для 'обратной' метрики рост значения = ухудшение.
        person — ФИО (или часть) либо табельный номер; date — неделя (YYYY-MM-DD).
        element НЕ указан = агрегат по метрике; чтобы получить конкретный
        продукт/разрез — задай element явно. Если у метрики нет агрегатной
        строки (см. agg- в составе датасета), вернутся все строки разрезов с
        пометкой 'разрезы_вместо_агрегата' — используй их как есть либо
        повтори с конкретным element."""
        person = _blank_to_none(person)
        element = _blank_to_none(element)
        date = _blank_to_none(date)
        unknown = _unknown_metric(metric) or _unknown_person(person)
        if unknown:
            return unknown
        return _pack(store.get_metric(metric, person=person, element=element, date=date))

    def compare(
        metric: str,
        person: str | None = None,
        element: str | None = None,
        dates: list[str] | None = None,
    ) -> str:
        """Динамика метрики по неделям (поля wow_change_pct и trend, плюс
        вердикты с учётом направления wow_status и trend_status —
        'улучшение'/'ухудшение'/'стабильно') для одного человека. Оценивай по
        *_status, а не по знаку wow_change: для 'обратной' метрики рост значения =
        ухудшение. person ОБЯЗАТЕЛЕН. element не указан = агрегат. Если у метрики
        нет агрегата (agg-), вернётся динамика по всем разрезам с пометкой
        'разрезы_вместо_агрегата'. Чтобы найти, у кого сильнее всего спад/рост
        по всем сотрудникам, используй find_flags."""
        person = _blank_to_none(person)
        element = _blank_to_none(element)
        unknown = _unknown_metric(metric) or _unknown_person(person)
        if unknown:
            return unknown
        return _pack(store.compare(metric, person=person, element=element, dates=dates))

    def rank(
        metric: str,
        date: str,
        element: str | None = None,
        post: str | None = None,
    ) -> str:
        """Рейтинг сотрудников по метрике на конкретную неделю. Направление уже
        учтено: peer_rank=1 — лучший. element не указан = агрегат по сотруднику;
        post — фильтр по должности. Если у метрики нет агрегата (agg- в
        составе датасета), тула вернёт error со списком доступных element —
        передай element и повтори."""
        element = _blank_to_none(element)
        post = _blank_to_none(post)
        unknown = _unknown_metric(metric)
        if unknown:
            return unknown
        return _pack(store.rank(metric, date, element=element, post=post))

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
        result = store.aggregate(metric, group_by, date=date, element=element)
        if "groups" in result:
            result = dict(result)
            result["groups"] = [_strip(g) for g in result["groups"]]
        return _dump(result)

    def metric_tree(
        metric: str | None = None,
        person: str | None = None,
        date: str | None = None,
        max_levels: int = 1,
    ) -> str:
        """ОДИН уровень иерархии метрики: метрика metric (или метрики верхнего
        уровня) и её ПРЯМЫЕ дети-компоненты с аналитикой по каждому узлу
        (plan_status, plan_dev_pct, benchmark_status, benchmark_dev_pct, trend,
        trend_status, wow_change_pct, wow_status, influent_percent). У каждого
        узла есть has_children: 1 — у него есть собственные подкомпоненты (можно
        копать глубже), 0 — это лист (глубже некуда). Динамику оценивай по
        trend_status/wow_status ('улучшение'/'ухудшение'), а не по сырому trend.
        Чтобы спуститься на следующий уровень, вызови metric_tree ПОВТОРНО на
        нужном ребёнке (обычно — с наибольшим отклонением и has_children=1), и
        так уровень за уровнем до листа. По умолчанию возвращается ОДИН уровень —
        НЕ разворачивай всё дерево сразу. Для разбора состава задавай metric и
        person (и date — иначе строк много). Если у метрики нет агрегата (agg- в
        составе датасета), корнями становятся все её разрезы — в результате будет
        пометка 'разрезы_вместо_агрегата'."""
        metric = _blank_to_none(metric)
        person = _blank_to_none(person)
        date = _blank_to_none(date)
        unknown = (
            _unknown_metric(metric) if metric is not None else None
        ) or _unknown_person(person)
        if unknown:
            return unknown
        # Клампим глубину: по умолчанию один уровень, максимум 3 — защита от
        # «дай всё дерево» (ровно то, что делало metric_tree бесполезным на
        # одной верхнеуровневой метрике). Спуск глубже — повторными вызовами.
        max_levels = max(1, min(int(max_levels or 1), 3))
        return _pack(
            store.metric_tree(
                name=metric, person=person, date=date, max_levels=max_levels
            ),
            curated=False,
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
        return _dump(
            {
                "people": store.list_people(
                    role=role, post=post, depart=depart, name_query=name_query
                )
            }
        )

    def find_flags(
        kind: str,
        date: str | None = None,
        metric: str | None = None,
        element: str | None = None,
    ) -> str:
        """Выборка предрассчитанных проблемных/заметных строк, ОТСОРТИРОВАННАЯ по
        силе: первая строка — самая значимая.
        kind: 'anomaly' — статистические выбросы (|z-score| выше порога);
        'below_plan' — факт ХУЖЕ ПЛАНА с учётом направления метрики (проблемные
        места); 'above_plan' — факт ЛУЧШЕ ПЛАНА (сильные стороны), первая строка —
        самое сильное перевыполнение;
        'declining' — ДИНАМИКА УХУДШИЛАСЬ с учётом направления (trend_status=
        'ухудшение'; для 'обратной' метрики это рост значения), первая строка —
        самое сильное ухудшение; 'improving' — динамика УЛУЧШИЛАСЬ
        (trend_status='улучшение'); 'trend' — любое движение ЗНАЧЕНИЯ (рост ИЛИ
        падение) без учёта хорошо/плохо — бери его, только когда нужно именно
        направление значения, а не вердикт.
        ВАЖНО: «просела / упала / снизилась / ухудшилась динамика» — это
        kind='declining', «выросла / улучшилась динамика» — kind='improving'
        (оба учитывают направление метрики). А «хуже плана / не выполняет план /
        отстаёт» — это kind='below_plan' (отклонение от плана). Это РАЗНЫЕ вопросы:
        метрика бывает хуже плана, но с улучшающейся динамикой, и наоборот.
        Чтобы сфокусировать выдачу, задавай metric (и date). Фильтры
        date/metric/element опциональны."""
        date = _blank_to_none(date)
        metric = _blank_to_none(metric)
        element = _blank_to_none(element)
        if metric and store.metric_type_of(metric) is None:
            metric = None
        return _pack(store.find_flags(kind, date=date, metric=metric, element=element))

    def analytics_summary() -> str:
        """Стартовая детерминированная сводка: охват датасета, средние по ключевым
        метрикам на последней неделе, топ аномалий, счётчики трендов."""
        return _dump(analytics.build_summary(store))

    def related_metrics(metric: str) -> str:
        """Связанные по СМЫСЛУ метрики (граф выведен LLM из названий и описаний,
        НЕ из значений). Для метрики metric возвращает рёбра, где она участвует:
        relation — тип связи ('опережающая→запаздывающая' / 'компонент' /
        'смежная' / 'влияет_на'), strength — сила ('низкая'/'средняя'/'высокая'),
        rationale — краткое обоснование. Используй, чтобы понять, на какие метрики
        данная влияет или от каких зависит — ОСОБЕННО когда у дочерней метрики не
        задан influent_percent (его вес влияния на родителя). Это ЭВРИСТИКА из
        текста, а не точный вес: не подавай связи как факт, помечай как
        предположительные."""
        metric = _blank_to_none(metric)
        if metric is None:
            return _dump({"error": "related_metrics требует точное название метрики"})
        unknown = _unknown_metric(metric)
        if unknown:
            return unknown
        edges = store.related_metrics(metric)
        return _dump({
            "metric": metric,
            "relations": [_strip(e) for e in edges],
            "count": len(edges),
        })

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
