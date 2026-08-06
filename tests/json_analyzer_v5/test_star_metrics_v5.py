"""json_analyzer_v5: звёздные метрики (star_received / is_star_metric).

Два НЕЗАВИСИМЫХ опциональных поля:
- ``star_received`` — бинарный результат «метрика получена / не получена»; он
  ЗАМЕНЯЕТ ``fact``, числа у такой метрики нет вовсе;
- ``is_star_metric`` — метрика влияет на получение звезды; стоит и на обычных
  числовых метриках.

Ключевое требование — обратная совместимость: нет полей во входе → ни новых
колонок, ни нового текста в промптах, ни нового инструмента (см.
``test_nothing_changes_without_star_fields``).
"""
from langgraph_executor.aegra_agents.json_analyzer_v5 import analytics
from langgraph_executor.aegra_agents.json_analyzer_v5.agent_base import format_facts
from langgraph_executor.aegra_agents.json_analyzer_v5.agent_classic import (
    compose_system_prompt,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.analytics import compute_analytics
from langgraph_executor.aegra_agents.json_analyzer_v5.loader import load_dataset_obj
from langgraph_executor.aegra_agents.json_analyzer_v5.prompts import SYSTEM_PROMPT_RULES
from langgraph_executor.aegra_agents.json_analyzer_v5.sqlite_store import SqliteStore
from langgraph_executor.aegra_agents.json_analyzer_v5.store_cache import EmbeddingIndex
from langgraph_executor.aegra_agents.json_analyzer_v5.tools import (
    _render_describe,
    _render_overview,
    _render_rows,
    _render_star_status,
    _render_tree,
    build_tools,
)

DATE = "2026-07-30"


def _node(name, fact=None, children=None, **extra):
    """Обычный числовой узел; **extra — звёздные и прочие опциональные поля."""
    node = {
        "id": f"{name}-id", "metric_name": name, "metric_type": "прямая",
        "measure_type": "рубль", "date": DATE, "calc_period": "Месяц",
        "fact": fact, "plan": fact, "benchmark": None, "element": None,
        "child_metrics": children or [],
    }
    node.update(extra)
    return node


def _binary(name, received, **extra):
    """Бинарный звёздный лист: без fact/plan и БЕЗ metric_type/measure_type —
    ровно так, как приходит с продакшена."""
    node = {
        "id": f"{name}-id", "metric_name": name, "date": DATE,
        "calc_period": "Месяц", "element": None,
        "star_received": received, "is_star_metric": True,
        "child_metrics": [],
    }
    node.update(extra)
    return node


def _rows(metrics):
    return load_dataset_obj({"me": {"fio": "Босс", "metrics": metrics}, "employees": []})


def _store(metrics):
    store = SqliteStore()
    store.load(_rows(metrics))
    compute_analytics(store)
    return store


def _tool_names(store):
    return {t.name for t in build_tools(store, EmbeddingIndex([]), lambda q: [0.0])}


# --------------------------------------------------------------------------- #
# Loader: тристейт и правило отбрасывания листа
# --------------------------------------------------------------------------- #

def test_binary_leaf_survives_empty_fact():
    """Бинарный лист без факта обязан попасть в базу: результат — в star_received."""
    rows = _rows([_binary("Обучение", False)])
    assert len(rows) == 1
    assert rows[0]["star_received"] == 0
    assert rows[0]["fact"] is None


def test_binary_leaf_without_star_still_skipped():
    """Страж изменённого правила: лист без факта и БЕЗ звёздных полей по-прежнему
    отбрасывается — иначе пустая точка затенит последнюю реальную в ряду дат."""
    assert _rows([_node("Продажи", None)]) == []


def test_absent_flags_are_none():
    """Отсутствие поля ≠ False: на этом различии держится вся совместимость."""
    row = _rows([_node("Продажи", 100)])[0]
    assert row["star_received"] is None
    assert row["is_star_metric"] is None


def test_false_is_not_none():
    row = _rows([_node("Продажи", 100, star_received=False, is_star_metric=False)])[0]
    assert row["star_received"] == 0
    assert row["is_star_metric"] == 0


def test_flags_are_independent():
    """is_star_metric живёт и на числовой метрике — её факт при этом сохраняется."""
    row = _rows([_node("Продажи", 100, is_star_metric=True)])[0]
    assert row["is_star_metric"] == 1
    assert row["star_received"] is None
    assert row["fact"] == 100


def test_string_booleans_normalized():
    rows = _rows([
        _node("Продажи", 100, is_star_metric="true"),
        _binary("Обучение", "нет"),
    ])
    by_name = {r["metric_name"]: r for r in rows}
    assert by_name["Продажи"]["is_star_metric"] == 1
    assert by_name["Обучение"]["star_received"] == 0


# --------------------------------------------------------------------------- #
# Обратная совместимость: без звёздных полей всё как раньше
# --------------------------------------------------------------------------- #

def test_nothing_changes_without_star_fields():
    """Одним кейсом по всем четырём поверхностям, где звезда могла бы протечь."""
    store = _store([_node("Продажи", 100)])
    overview = store.schema_overview()

    rendered = _render_rows(store.get_metric("Продажи"))
    assert "метрика получена" not in rendered
    assert "влияет на звезду" not in rendered

    assert "звезд" not in format_facts(overview).casefold()
    assert compose_system_prompt(overview) == (
        SYSTEM_PROMPT_RULES + "\n\n" + format_facts(overview)
    )
    assert "star_status" not in _tool_names(store)
    assert store.star_presence() == {
        "star_binary_rows": 0, "star_received_rows": 0, "star_metric_rows": 0,
    }


def test_situation_overview_has_no_star_key_without_star_data():
    store = _store([_node("Продажи", 100, children=[_node("Лиды", 40)])])
    assert "star" not in analytics.build_situation_overview(store)


# --------------------------------------------------------------------------- #
# Путь с данными
# --------------------------------------------------------------------------- #

def test_star_columns_present_when_star_data():
    store = _store([
        _node("Продажи", 100, is_star_metric=True),
        _binary("Обучение", False),
    ])
    assert "влияет на звезду" in _render_rows(store.get_metric("Продажи"))
    rendered = _render_rows(store.get_metric("Обучение"))
    assert "метрика получена" in rendered
    assert "нет" in rendered


def test_metric_tree_binary_line():
    """Без спец-ветки узел печатался бы как «Обучение: » с пустым значением."""
    store = _store([_node("Продажи", 100, children=[_binary("Обучение", False)])])
    rendered = _render_tree(store.metric_tree("Продажи"))
    assert "Обучение: НЕ получена" in rendered
    assert "Обучение: |" not in rendered


def test_star_status_tool_registered_and_renders():
    store = _store([
        _node("Продажи", 100, is_star_metric=True),
        _binary("Обучение", False),
        _binary("Наставничество", True),
    ])
    assert "star_status" in _tool_names(store)
    rendered = _render_star_status(store.star_status())
    assert "НЕ получены: Обучение" in rendered
    assert "Получены: Наставничество" in rendered
    assert "Звезда заработана: нет" in rendered
    assert "Продажи" in rendered  # числовая звёздная метрика — таблицей


def test_star_earned_when_all_received():
    store = _store([_binary("Обучение", True), _binary("Наставничество", True)])
    assert "Звезда заработана: да" in _render_star_status(store.star_status())


def test_star_status_uses_latest_period():
    """Бинарный показатель приходит КАЖДЫЙ период. Без выбора последнего одна и та
    же метрика попала бы разом в «получены» и «не получены», а вердикт звезды
    считался бы по устаревшему периоду."""
    metrics = [
        {"id": "x", "metric_name": "Обучение", "date": d, "calc_period": "неделя",
         "element": None, "star_received": received, "is_star_metric": True,
         "child_metrics": []}
        for d, received in (("2026-05-04", False), ("2026-05-11", True))
    ]
    store = _store(metrics)
    result = store.star_status()
    assert result["date"] == "2026-05-11"
    rendered = _render_star_status(result)
    assert "НЕ получены" not in rendered
    assert "Звезда заработана: да" in rendered
    assert "за период 2026-05-11" in rendered


def test_star_status_explicit_date_wins():
    metrics = [
        {"id": "x", "metric_name": "Обучение", "date": d, "calc_period": "неделя",
         "element": None, "star_received": received, "is_star_metric": True,
         "child_metrics": []}
        for d, received in (("2026-05-04", False), ("2026-05-11", True))
    ]
    store = _store(metrics)
    assert "Звезда заработана: нет" in _render_star_status(
        store.star_status(date="2026-05-04")
    )


def test_find_flags_star_kinds():
    store = _store([
        _node("Продажи", fact=50, is_star_metric=True),
        _binary("Обучение", False),
        _binary("Наставничество", True),
    ])
    # План у «Продажи» = факт, поэтому 'в плане' — star_at_risk пуст.
    missed = store.find_flags("star_missed")["rows"]
    assert [r["metric_name"] for r in missed] == ["Обучение"]
    received = store.find_flags("star_received")["rows"]
    assert [r["metric_name"] for r in received] == ["Наставничество"]


def test_find_flags_star_at_risk():
    store = SqliteStore()
    store.load(_rows([_node("Продажи", fact=50, plan=100, is_star_metric=True)]))
    compute_analytics(store)
    rows = store.find_flags("star_at_risk")["rows"]
    assert [r["metric_name"] for r in rows] == ["Продажи"]


def test_compute_analytics_survives_binary_rows():
    """Смешанный датасет: бинарная строка не роняет расчёт и не получает вердиктов."""
    store = _store([_node("Продажи", 100), _binary("Обучение", False)])
    row = store.get_metric("Обучение")["rows"][0]
    for field in ("plan_status", "pop_status", "trend_status", "peer_status"):
        assert row[field] is None
    assert not row["is_anomaly"]


def test_situation_overview_excludes_binary():
    store = _store([
        _node("Продажи", 100, children=[_node("Лиды", 40)]),
        _binary("Обучение", False),
    ])
    result = analytics.build_situation_overview(store)
    zones = (result["problems"] or []) + result["positives"] + result["stable"]
    assert "Обучение" not in {z.get("metric") for z in zones}
    assert result["star"]["missed"] == [{"metric": "Обучение", "element": None}]
    rendered = _render_overview(result)
    assert "Звезда:" in rendered
    assert "НЕ получены: Обучение" in rendered


def test_numeric_guards_on_binary_metric():
    store = _store([_binary("Обучение", False)])
    for result in (
        store.aggregate("Обучение", "person"),
        store.rank("Обучение", DATE),
        analytics.rank_elements(store, "Обучение"),
    ):
        assert "бинарная" in result["error"]
        assert "star_status" in result["hint"]


def test_binary_metric_without_metric_type_is_found():
    """Страж правки metric_exists: бинарная метрика приходит без metric_type, и
    прежняя проверка «metric_type_of is not None» объявляла бы её ненайденной."""
    store = _store([_binary("Обучение", False)])
    assert store.metric_exists("Обучение")
    assert store.metric_type_of("Обучение") is None
    tools = {t.name: t for t in build_tools(store, EmbeddingIndex([]), lambda q: [0.0])}
    assert "не найдена" not in tools["get_metric"].func(metric="Обучение")


def test_describe_metric_binary_and_star():
    store = _store([
        _binary("Обучение", False),
        _node("Продажи", 100, is_star_metric=True),
    ])
    binary = _render_describe(store.describe_metric("Обучение"))
    assert "БИНАРНАЯ" in binary
    assert "меньше=лучше" not in binary and "больше=лучше" not in binary
    assert "Влияет на получение звезды." in binary
    assert "Влияет на получение звезды." in _render_describe(
        store.describe_metric("Продажи")
    )


def test_schema_overview_star_counters_and_prompt_block():
    store = _store([
        _node("Продажи", 100, is_star_metric=True),
        _binary("Обучение", False),
        _binary("Наставничество", True),
    ])
    overview = store.schema_overview()
    assert overview["star_binary_rows"] == 2
    assert overview["star_received_rows"] == 1
    assert overview["star_metric_rows"] == 3  # is_star_metric стоит и на бинарных

    facts = format_facts(overview)
    assert "ЗВЁЗДНЫЕ данные" in facts
    assert "бинарная" in facts
    assert "14. ЗВЕЗДА" in compose_system_prompt(overview)
