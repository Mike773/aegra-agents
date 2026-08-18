"""Загрузка входного JSON и разворачивание дерева метрик в плоские строки.

Структура входа фиксирована для всех доменов (колл-центр, разработчики и т.д.):
    {"me": <person>, "employees": [<person>, ...]}
    person  = {tabnum, fio, post, depart, metrics: [<metric>, ...]}
    metric  = {id, metric_name, metric_description, metric_type, measure_type,
               date, calc_period, fact, plan, benchmark, [ex], [rr],
               [influent_percent], element, [rankings],
               [star_received], [is_star_metric],
               child_metrics: [<metric>, ...]}

``star_received`` — бинарный результат «метрика получена / не получена»; он
ЗАМЕНЯЕТ ``fact``: числа у такой метрики нет вовсе (как и плана с бенчмарком).
``is_star_metric`` — метрика влияет на получение звезды; НЕЗАВИСИМО от
``star_received`` стоит и на обычных числовых метриках. Оба поля опциональны:
их отсутствие (None) — не то же самое, что False, см. ``_normalize_flag``.
    rankings = [{rank: "458 из 500", level: <код уровня>, percentile}, ...] —
                 место сотрудника в peer-группе уровня.

Отдельный вход — batch-агрегаты peer-групп (``load_aggregates_obj``):
    [{"dataset": {"level": <код уровня>, [level_name], "metrics": [<agg_metric>]}}]
    agg_metric = {metric_id, metric_name, aggregates: {dt, calc_period,
                  mean_fact, mean_plan, mean_ex, median, hit_rate,
                  top20_mean_fact, iqr, cv, total_objects,
                  history: [<те же поля за прошлые периоды>]},
                  children_metrics: [<agg_metric>, ...]}

``level`` — служебный код группы сравнения (наборы кодов инстанс-специфичны),
``level_name`` — её человеческое название («по офису»), единственный источник
имени для выдачи. Поле опционально: без него группа описывается обезличенно,
код в выдачу не попадает никогда.

Названия конкретных метрик НЕ хардкодятся — обходим то, что есть в JSON.
"""
from __future__ import annotations

import re
from typing import Any

ROW_FIELDS: tuple[str, ...] = (
    "metric_uid",
    "parent_uid",
    "depth",
    "person_tabnum",
    "person_fio",
    "person_post",
    "person_depart",
    "person_is_me",
    "person_key",
    "metric_id",
    "metric_name",
    "metric_description",
    "metric_type",
    "measure_type",
    "date",
    "calc_period",
    "fact",
    "plan",
    "benchmark",
    "ex",
    "rr",
    "influent_percent",
    "element",
    # Бинарный результат «метрика получена / не получена». ЗАМЕНЯЕТ fact: числа у
    # такой метрики нет. NULL = поля не было (обычная числовая метрика).
    "star_received",
    # Метрика влияет на получение звезды. Независима от star_received: бывает и на
    # обычных числовых метриках (там влияние идёт через выполнение плана).
    "is_star_metric",
)


def _is_empty_fact(value: Any) -> bool:
    """Факт «пустой» = None или пустая строка. Ноль (0/0.0) — ВАЛИДНЫЙ факт и
    пустым не считается (иначе потеряли бы законные нулевые значения)."""
    if value is None:
        return True
    return isinstance(value, str) and value.strip() == ""


_TRUE_TOKENS = frozenset({"true", "1", "да", "yes", "y"})
_FALSE_TOKENS = frozenset({"false", "0", "нет", "no", "n"})


def _normalize_flag(value: Any) -> int | None:
    """Тристейт булева поля → 1 / 0 / None.

    None означает «поля не было» (или пришёл null) и НЕ равно False: на этом
    различии держится обратная совместимость — датасет без звёздных полей обязан
    вести себя ровно как раньше, поэтому «флага нет» нельзя схлопывать в 0.
    Источники присылают bool, 0/1 и строки в разном регистре — как с metric_type.
    """
    if value is None:
        return None
    # bool — подкласс int, проверяем его первым, иначе True уйдёт в ветку чисел.
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value else 0
    if isinstance(value, str):
        norm = value.strip().casefold()
        if norm in _TRUE_TOKENS:
            return 1
        if norm in _FALSE_TOKENS:
            return 0
    return None


def _has_binary_result(node: dict[str, Any]) -> bool:
    """Бинарная («звёздная») метрика приходит БЕЗ числового факта: её результат
    лежит в star_received. Такой лист нельзя выкидывать по правилу «нет факта —
    нет строки», иначе метрика молча исчезнет из датасета.

    Спасаем строку строго по star_received, а НЕ по is_star_metric: последний
    стоит и на обычных числовых метриках, а числовой лист с пустым фактом — это
    ровно тот случай, ради которого правило отбрасывания и написано."""
    return _normalize_flag(node.get("star_received")) is not None


def _normalize_metric_type(value: Any) -> Any:
    """Приводит направление метрики к каноничным 'прямая'/'обратная'.

    Источники присылают разные формы и регистр ('Обратный', 'ОБРАТНАЯ',
    'прямой'), а весь расчёт вердиктов и подсказки LLM сравнивают строго с
    'обратная' (analytics.py: `metric_type != "обратная"`). Без нормализации
    'Обратный' не матчится и обратная метрика считается как прямая. Неизвестные
    значения отдаём как есть (трактуются как 'прямая', прежнее поведение)."""
    if not isinstance(value, str):
        return value
    norm = value.strip().casefold()
    if norm.startswith("обратн"):
        return "обратная"
    if norm.startswith("прям"):
        return "прямая"
    return value


def _walk(
    metrics: list[dict[str, Any]],
    person: dict[str, Any],
    rows: list[dict[str, Any]],
    counter: list[int],
    parent_uid: int | None,
    depth: int,
) -> None:
    for node in metrics:
        children = node.get("child_metrics") or []
        # Строку с пустым фактом в базу не грузим: в ряду по датам пустая точка
        # затеняет последнюю реальную (analytics берёт prev = непосредственно
        # предыдущий период), и сравнение период-к-периоду не считается. Отбрасываем
        # только ЛИСТ без факта — узел-агрегат с детьми сохраняем, чтобы не потерять
        # реальные разрезы под ним (на практике пустой факт всегда у листьев).
        # Исключение — бинарная метрика: у неё факта нет по определению.
        if (
            not children
            and _is_empty_fact(node.get("fact"))
            and not _has_binary_result(node)
        ):
            continue
        uid = counter[0]
        counter[0] += 1
        rankings = node.get("rankings")
        rows.append(
            {
                "metric_uid": uid,
                "parent_uid": parent_uid,
                "depth": depth,
                "person_tabnum": person.get("tabnum"),
                "person_fio": person.get("fio"),
                "person_post": person.get("post"),
                "person_depart": person.get("depart"),
                "person_is_me": 1 if person.get("_is_me") else 0,
                "person_key": person.get("_key"),
                "metric_id": node.get("id"),
                "metric_name": node.get("metric_name"),
                "metric_description": node.get("metric_description"),
                "metric_type": _normalize_metric_type(node.get("metric_type")),
                "measure_type": node.get("measure_type"),
                "date": node.get("date"),
                "calc_period": node.get("calc_period"),
                "fact": node.get("fact"),
                "plan": node.get("plan"),
                "benchmark": node.get("benchmark"),
                "ex": node.get("ex"),
                "rr": node.get("rr"),
                "influent_percent": node.get("influent_percent"),
                "element": node.get("element"),
                "star_received": _normalize_flag(node.get("star_received")),
                "is_star_metric": _normalize_flag(node.get("is_star_metric")),
                # Не в ROW_FIELDS: в таблицу metrics не идёт, SqliteStore.load
                # разложит по отдельной metric_rankings.
                "rankings": rankings if isinstance(rankings, list) else None,
            }
        )
        if children:
            _walk(children, person, rows, counter, uid, depth + 1)


def _people(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Персоны датасета (me + employees) с каноническим ключом ``_key``.

    Ключ — для ВНУТРЕННЕЙ идентичности: табельный (как текст), иначе ФИО, иначе
    индекс. Нужен, т.к. на проде tabnum может приходить null — тогда без фолбэка
    разные люди схлопывались бы по NULL person_tabnum."""
    people: list[dict[str, Any]] = []
    me = data.get("me")
    if me is not None:
        people.append({**me, "_is_me": True})
    for emp in data.get("employees", []) or []:
        people.append({**emp, "_is_me": False})

    for i, person in enumerate(people):
        tab = person.get("tabnum")
        fio = person.get("fio")
        person["_key"] = str(tab) if tab is not None else (fio if fio else f"person{i}")
    return people


def load_dataset_obj(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Разворачивает уже распарсенный JSON-объект в плоский список строк."""
    rows: list[dict[str, Any]] = []
    counter = [1]
    for person in _people(data):
        _walk(person.get("metrics", []) or [], person, rows, counter, None, 1)
    return rows


def load_person_aggregates_obj(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Связь «персона → её предагрегаты» из индивидуального датасета.

    ``aggregates_ids`` лежит у персоны рядом с ``metrics``; ключи персон те же,
    что в строках ``load_dataset_obj`` (person_key). Пары уникальны. Старый
    формат без поля → ``[]`` — привязка опциональна."""
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for person in _people(data):
        for agg_id in person.get("aggregates_ids") or []:
            if agg_id is None:
                continue
            text = str(agg_id).strip()
            if not text or (person["_key"], text) in seen:
                continue
            seen.add((person["_key"], text))
            pairs.append({"person_key": person["_key"], "aggregate_id": text})
    return pairs


_RANK_RE = re.compile(r"(\d+)\D+(\d+)")


def parse_rank(raw: Any) -> tuple[int | None, int | None]:
    """Разбирает строку ранга «458 из 500» → (458, 500).

    Формат разделителя не фиксируем (regex «число, не-числа, число»).
    Нераспознанное → (None, None); исходная строка сохраняется в rank_raw."""
    if not isinstance(raw, str):
        return None, None
    m = _RANK_RE.search(raw)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


AGG_ROW_FIELDS: tuple[str, ...] = (
    "node_uid",
    "parent_node_uid",
    "depth",
    "level",
    "level_name",
    "aggregate_id",
    "metric_id",
    "metric_name",
    "dt",
    "calc_period",
    "is_current",
    "mean_fact",
    "mean_plan",
    "mean_ex",
    "median",
    "hit_rate",
    "top20_mean_fact",
    "iqr",
    "cv",
    "total_objects",
)

_AGG_VALUE_KEYS = (
    "mean_fact",
    "mean_plan",
    "mean_ex",
    "median",
    "hit_rate",
    "top20_mean_fact",
    "iqr",
    "cv",
    "total_objects",
)


def _walk_aggregates(
    metrics: list[Any],
    level: Any,
    level_name: Any,
    aggregate_id: Any,
    rows: list[dict[str, Any]],
    counter: list[int],
    parent_uid: int | None,
    depth: int,
) -> None:
    for node in metrics or []:
        if not isinstance(node, dict):
            continue
        uid = counter[0]
        counter[0] += 1
        agg = node.get("aggregates") or {}
        base = {
            "node_uid": uid,
            "parent_node_uid": parent_uid,
            "depth": depth,
            "level": level,
            "level_name": level_name,
            "aggregate_id": aggregate_id,
            "metric_id": node.get("metric_id"),
            "metric_name": node.get("metric_name"),
        }

        def _row(src: dict[str, Any], is_current: int) -> dict[str, Any]:
            return {
                **base,
                "dt": src.get("dt"),
                "calc_period": src.get("calc_period"),
                "is_current": is_current,
                **{k: src.get(k) for k in _AGG_VALUE_KEYS},
            }

        if isinstance(agg, dict) and agg:
            rows.append(_row(agg, 1))
            for h in agg.get("history") or []:
                if isinstance(h, dict):
                    rows.append(_row(h, 0))
        # В этом payload дети приходят как "children_metrics" (в основном
        # датасете — "child_metrics"); принимаем оба на всякий случай.
        children = node.get("children_metrics") or node.get("child_metrics") or []
        _walk_aggregates(
            children, level, level_name, aggregate_id, rows, counter, uid, depth + 1
        )


def load_aggregates_obj(data: Any) -> list[dict[str, Any]]:
    """Разворачивает batch-агрегаты peer-групп в плоские строки.

    Вход: список ``{"dataset": {"level", "metrics": [...]}}`` (см. докстринг
    модуля); запись может нести пометку ``aggregate_id`` — id предагрегата,
    которым её загрузили (оркестратор помечает при по-штучной загрузке). Все
    срезы одного узла метрики (текущий ``aggregates`` + записи ``history``)
    делят ``node_uid``; текущий помечен ``is_current=1``. Битый или пустой
    вход → ``[]`` — агрегаты опциональны и не фатальны.
    """
    rows: list[dict[str, Any]] = []
    counter = [1]
    if not isinstance(data, list):
        return rows
    for entry in data:
        ds = entry.get("dataset") if isinstance(entry, dict) else None
        if isinstance(ds, dict):
            _walk_aggregates(
                ds.get("metrics") or [], ds.get("level"), ds.get("level_name"),
                entry.get("aggregate_id"), rows, counter, None, 1,
            )
    return rows
