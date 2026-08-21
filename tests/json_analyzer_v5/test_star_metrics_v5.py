"""json_analyzer_v5: звёзды (star_received / is_star_metric).

Семантика:
- ЗВЕЗДА — узел дерева с непустым ``star_received`` (получена / НЕ получена).
  Числа (fact/plan) у неё нет; ``is_star_metric`` на ней ничего не значит.
  Звёзд у сотрудника несколько, с разными именами, каждая независима.
- Влияющие на звезду показатели — её ``child_metrics`` с ``is_star_metric=true``:
  обычные числовые метрики с фактом, планом и историей. Все даты ребёнка лежат
  под одним узлом звезды (так отдаёт прод-конвертер agent_dataset.py).

Ключевое требование — обратная совместимость: нет полей во входе → ни новых
колонок, ни нового текста в промптах, ни нового инструмента (см.
``test_nothing_changes_without_star_fields``).
"""
import json
import os

from langgraph_executor.aegra_agents.json_analyzer_v5 import analytics
from langgraph_executor.aegra_agents.json_analyzer_v5.agent_base import format_facts
from langgraph_executor.aegra_agents.json_analyzer_v5.agent_classic import (
    compose_system_prompt,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.analytics import compute_analytics
from langgraph_executor.aegra_agents.json_analyzer_v5.loader import (
    has_star_fields,
    load_dataset_obj,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.prompts import SYSTEM_PROMPT_RULES
from langgraph_executor.aegra_agents.json_analyzer_v5.sqlite_store import SqliteStore
from langgraph_executor.aegra_agents.json_analyzer_v5.store_cache import EmbeddingIndex
from langgraph_executor.aegra_agents.json_analyzer_v5.tools import (
    _render_describe,
    _render_overview,
    _render_rows,
    _render_schema,
    _render_star_status,
    _render_tree,
    build_tools,
)

PREV = "2026-07-23"
DATE = "2026-07-30"
SAMPLE = os.path.join(
    os.path.dirname(__file__), "..", "..", "samples_v2", "sample_star.json"
)


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


def _child(name, series, plan, metric_type="прямая", unit="%", star=True):
    """Дочерний показатель звезды: по узлу на каждую дату серии, все — сиблинги
    в child_metrics звезды (как отдаёт прод-конвертер)."""
    out = []
    for dt, fact in series:
        node = {
            "id": f"{name}-id", "metric_name": name, "metric_type": metric_type,
            "measure_type": unit, "date": dt, "calc_period": "неделя",
            "fact": fact, "plan": plan, "benchmark": None, "element": None,
            "child_metrics": [],
        }
        if star:
            node["is_star_metric"] = True
        out.append(node)
    return out


def _star(name, received, children=None, date=DATE, **extra):
    """Узел звезды: без fact/plan и БЕЗ metric_type/measure_type — как на проде."""
    node = {
        "id": f"{name}-id", "metric_name": name, "date": date,
        "calc_period": "месяц", "element": None,
        "star_received": received, "is_star_metric": True,
        "child_metrics": children or [],
    }
    node.update(extra)
    return node


def _quality(received=False):
    """«Звезда качества»: CSI хуже плана, FCR лучше плана, «Оценки» — без флага."""
    return _star(
        "Звезда качества", received,
        _child("CSI", [(PREV, 4.2), (DATE, 4.1)], 4.5, unit="балл")
        + _child("FCR", [(PREV, 77.0), (DATE, 78.0)], 75.0)
        + _child("Оценки", [(PREV, 39), (DATE, 44)], None, unit="шт", star=False),
    )


def _sales(received=True):
    return _star(
        "Звезда продаж", received,
        _child("Конверсия", [(PREV, 12.8), (DATE, 13.4)], 12.0),
    )


def _rows(metrics):
    return load_dataset_obj({"me": {"fio": "Босс", "metrics": metrics}, "employees": []})


def _store(metrics):
    store = SqliteStore()
    store.load(_rows(metrics))
    compute_analytics(store)
    return store


def _tool_names(store):
    return {t.name for t in build_tools(store, EmbeddingIndex([]), lambda q: [0.0])}


def _by_name(rows):
    return {r["metric_name"]: r for r in rows}


# --------------------------------------------------------------------------- #
# Loader: тристейт и правило отбрасывания листа
# --------------------------------------------------------------------------- #

def test_star_leaf_survives_empty_fact():
    """Звезда без детей и без факта обязана попасть в базу: результат — в
    star_received."""
    rows = _rows([_star("Звезда качества", False)])
    assert len(rows) == 1
    assert rows[0]["star_received"] == 0
    assert rows[0]["fact"] is None


def test_leaf_without_star_still_skipped():
    """Страж правила: лист без факта и БЕЗ star_received по-прежнему отбрасывается
    — иначе пустая точка затенит последнюю реальную в ряду дат."""
    assert _rows([_node("Продажи", None)]) == []
    assert _rows([_node("Продажи", None, is_star_metric=True)]) == []


def test_absent_flags_are_none():
    """Отсутствие поля ≠ False: на этом различии держится вся совместимость."""
    row = _rows([_node("Продажи", 100)])[0]
    assert row["star_received"] is None
    assert row["is_star_metric"] is None


def test_false_is_not_none():
    row = _rows([_node("Продажи", 100, star_received=False, is_star_metric=False)])[0]
    assert row["star_received"] == 0
    assert row["is_star_metric"] == 0


def test_string_booleans_normalized():
    rows = _rows([
        _node("Продажи", 100, is_star_metric="true"),
        _star("Звезда", "нет"),
    ])
    by = _by_name(rows)
    assert by["Продажи"]["is_star_metric"] == 1
    assert by["Звезда"]["star_received"] == 0


def test_star_children_linked_by_parent_uid():
    """Все даты ребёнка — под одним uid звезды; у звезды depth=1, у детей 2."""
    rows = _rows([_quality()])
    star = next(r for r in rows if r["star_received"] is not None)
    kids = [r for r in rows if r["metric_name"] == "CSI"]
    assert len(kids) == 2
    assert {k["parent_uid"] for k in kids} == {star["metric_uid"]}
    assert star["depth"] == 1 and all(k["depth"] == 2 for k in kids)


def test_has_star_fields_predicate():
    assert has_star_fields({"star_received": 0})
    assert has_star_fields({"is_star_metric": 1})
    assert not has_star_fields({"star_received": None, "is_star_metric": None})
    assert not has_star_fields({"is_star_metric": 0})


# --------------------------------------------------------------------------- #
# Обратная совместимость: без звёздных полей всё как раньше
# --------------------------------------------------------------------------- #

def test_nothing_changes_without_star_fields():
    """Одним кейсом по всем поверхностям, где звезда могла бы протечь."""
    store = _store([_node("Продажи", 100, children=[_node("Лиды", 40)])])
    overview = store.schema_overview()

    rendered = _render_rows(store.get_metric("Продажи"))
    assert "звезд" not in rendered.casefold()
    assert "звезд" not in _render_tree(store.metric_tree("Продажи")).casefold()
    assert "звезд" not in _render_schema(overview).casefold()

    assert "звезд" not in format_facts(overview).casefold()
    assert compose_system_prompt(overview) == (
        SYSTEM_PROMPT_RULES + "\n\n" + format_facts(overview)
    )
    assert "star_status" not in _tool_names(store)
    assert store.star_presence() == {
        "star_binary_rows": 0, "star_received_rows": 0, "star_metric_rows": 0,
        "star_names": [],
    }


def test_situation_overview_has_no_star_key_without_star_data():
    store = _store([_node("Продажи", 100, children=[_node("Лиды", 40)])])
    assert "star" not in analytics.build_situation_overview(store)


# --------------------------------------------------------------------------- #
# Счётчики и состав датасета
# --------------------------------------------------------------------------- #

def test_star_presence_counts_children_not_stars():
    store = _store([_quality(), _sales()])
    presence = store.star_presence()
    assert presence["star_binary_rows"] == 2
    assert presence["star_received_rows"] == 1
    # Флаг на самой звезде ничего не значит — считаем только влияющие строки.
    assert presence["star_metric_rows"] == 6  # CSI×2 + FCR×2 + Конверсия×2
    assert presence["star_names"] == ["Звезда качества", "Звезда продаж"]


def test_schema_overview_tags_and_prompt_block():
    store = _store([_quality(), _sales(), _node("Продажи", 100)])
    overview = store.schema_overview()
    by = _by_name(overview["metrics"])
    assert by["Звезда качества"]["is_star"] == 1
    assert by["Звезда качества"]["is_star_metric"] == 0
    assert by["CSI"]["is_star_metric"] == 1
    assert by["CSI"]["star_of"] == "Звезда качества"
    assert by["Оценки"]["is_star_metric"] == 0
    assert by["Продажи"]["star_of"] is None

    schema = _render_schema(overview)
    assert "Звезда качества (" in schema and ", звезда]" in schema
    assert "влияет на звезду «Звезда качества»" in schema

    facts = format_facts(overview)
    assert "ЗВЁЗДЫ" in facts
    assert "«Звезда качества», «Звезда продаж»" in facts
    assert "бинарн" not in facts.casefold()
    assert "14. ЗВЁЗДЫ" in compose_system_prompt(overview)


# --------------------------------------------------------------------------- #
# star_status: по звёздам
# --------------------------------------------------------------------------- #

def test_star_status_per_star_structure():
    store = _store([_quality(), _sales()])
    result = store.star_status()
    assert "star_status" in _tool_names(store)
    stars = result["stars"]
    assert [s["star"] for s in stars] == ["Звезда качества", "Звезда продаж"]
    quality, sales = stars
    assert quality["received"] is False and sales["received"] is True
    kids = _by_name(quality["children"])
    assert kids["CSI"]["plan_status"] == "хуже_плана"
    assert kids["CSI"]["star_name"] == "Звезда качества"
    assert kids["CSI"]["is_star_metric"] == 1
    assert kids["Оценки"]["is_star_metric"] is None
    # Отстающие влияющие — первыми, без флага — последними.
    assert [c["metric_name"] for c in quality["children"]] == ["CSI", "FCR", "Оценки"]
    assert _by_name(sales["children"])["Конверсия"]["plan_status"] == "лучше_плана"


def test_star_status_children_latest_period_only():
    store = _store([_quality()])
    quality = store.star_status()["stars"][0]
    assert {c["date"] for c in quality["children"]} == {DATE}
    assert quality["children_date"] == DATE
    assert quality["date"] == DATE


def test_star_status_explicit_date_wins():
    """Явная дата: и звезда, и дети за неё. Звезда есть только на последней дате,
    значит, за прошлую — звёзд нет, но сам вызов не падает."""
    store = _store([_quality()])
    result = store.star_status(date=PREV)
    assert result["date"] == PREV
    assert result["stars"] == []
    assert "Звёзд" in _render_star_status(result)


def test_star_status_child_flag_from_history():
    """У исторических строк ребёнка флага может не быть (прод-конвертер копирует
    его из history): влияние определяем по серии, а не построчно."""
    kids = _child("CSI", [(PREV, 4.2)], 4.5, star=False) + _child("CSI", [(DATE, 4.1)], 4.5)
    store = _store([_star("Звезда качества", False, kids)])
    child = store.star_status()["stars"][0]["children"][0]
    assert child["date"] == DATE and child["is_star_metric"] == 1
    # И наоборот: флаг только на истории — ребёнок всё равно влияющий.
    kids = _child("CSI", [(PREV, 4.2)], 4.5) + _child("CSI", [(DATE, 4.1)], 4.5, star=False)
    store = _store([_star("Звезда качества", False, kids)])
    assert store.star_status()["stars"][0]["children"][0]["is_star_metric"] == 1


def test_render_star_status_text():
    store = _store([_quality(), _sales()])
    text = _render_star_status(store.star_status())
    assert f"Звёзды сотрудника Босс (за {DATE}):" in text
    assert (
        "- «Звезда качества» — НЕ получена. Влияющие показатели: "
        "CSI 4.1 балл при плане 4.5 балл (хуже плана на 8.89 %"
    ) in text
    assert "FCR 78 % при плане 75 % (лучше плана на 4 %" in text
    assert "Прочие показатели: Оценки 44 шт" in text
    assert "- «Звезда продаж» — получена. Влияющие показатели: Конверсия 13.4 %" in text
    assert text.index("Звезда качества") < text.index("Звезда продаж")
    assert "заработана" not in text
    assert "бинарн" not in text.casefold()


def test_star_status_falls_back_to_all_children_without_flags():
    kids = _child("CSI", [(DATE, 4.1)], 4.5, star=False)
    store = _store([_star("Звезда качества", False, kids)])
    text = _render_star_status(store.star_status())
    assert "Влияющие показатели: CSI 4.1 % при плане 4.5 %" in text
    assert "Прочие" not in text


def test_star_status_star_without_children():
    store = _store([_star("Звезда качества", False)])
    text = _render_star_status(store.star_status())
    assert "«Звезда качества» — НЕ получена. Влияющие показатели в данных не указаны." in text


def test_star_status_no_stars_message():
    store = _store([_node("Продажи", 100)])
    assert _render_star_status(store.star_status()) == "Звёзд в датасете нет."


def test_star_status_person_filter():
    data = {
        "me": None,
        "employees": [
            {"fio": "Иванов", "metrics": [_quality(False)]},
            {"fio": "Петров", "metrics": [_quality(True)]},
        ],
    }
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    compute_analytics(store)
    stars = store.star_status(person="Петров")["stars"]
    assert len(stars) == 1 and stars[0]["received"] is True
    assert stars[0]["person_fio"] == "Петров"


# --------------------------------------------------------------------------- #
# find_flags, get_metric, metric_tree, describe
# --------------------------------------------------------------------------- #

def test_find_flags_star_kinds():
    store = _store([_quality(), _sales()])
    missed = store.find_flags("star_missed")["rows"]
    assert [r["metric_name"] for r in missed] == ["Звезда качества"]
    received = store.find_flags("star_received")["rows"]
    assert [r["metric_name"] for r in received] == ["Звезда продаж"]
    at_risk = store.find_flags("star_at_risk", date=DATE)["rows"]
    assert [(r["metric_name"], r["star_name"]) for r in at_risk] == [
        ("CSI", "Звезда качества")
    ]


def test_star_column_in_rows_only_for_star_children():
    store = _store([_quality(), _node("Продажи", 100)])
    plain = _render_rows(store.get_metric("Продажи"))
    assert "| звезда |" not in plain
    child = _render_rows(store.get_metric("CSI"))
    assert "| звезда |" in child and "Звезда качества" in child
    assert "влияет на звезду" in child
    star = _render_rows(store.get_metric("Звезда качества"))
    assert "звезда получена" in star
    assert "влияет на звезду" not in star


def test_metric_tree_star_lines():
    store = _store([_quality()])
    rendered = _render_tree(store.metric_tree("Звезда качества"))
    assert "Звезда качества (звезда): НЕ получена" in rendered
    assert "Звезда качества (звезда): |" not in rendered
    assert "CSI: 4.1 балл | влияет на звезду «Звезда качества»" in rendered
    oценки = [ln for ln in rendered.splitlines() if "Оценки" in ln]
    assert oценки and all("влияет" not in ln for ln in oценки)


def test_numeric_guards_on_star_node():
    store = _store([_quality()])
    for result in (
        store.aggregate("Звезда качества", "person"),
        store.rank("Звезда качества", DATE),
        analytics.rank_elements(store, "Звезда качества"),
    ):
        assert "звезда" in result["error"]
        assert "star_status" in result["hint"]
    # Ребёнок звезды — обычная числовая метрика: звёздный guard на него не срабатывает.
    assert "звезда" not in store.rank("CSI", DATE).get("error", "")


def test_star_without_metric_type_is_found():
    store = _store([_star("Звезда качества", False)])
    assert store.metric_exists("Звезда качества")
    assert store.metric_type_of("Звезда качества") is None
    tools = {t.name: t for t in build_tools(store, EmbeddingIndex([]), lambda q: [0.0])}
    assert "не найдена" not in tools["get_metric"].func(metric="Звезда качества")


def test_describe_metric_star_and_child():
    store = _store([_quality()])
    star = _render_describe(store.describe_metric("Звезда качества"))
    assert "Вид: ЗВЕЗДА" in star
    assert "меньше=лучше" not in star and "больше=лучше" not in star
    assert "star_status" in star
    assert "Влияет на получение звезды" not in star
    child = _render_describe(store.describe_metric("CSI"))
    assert "Влияет на получение звезды «Звезда качества»." in child


def test_compute_analytics_survives_star_rows():
    store = _store([_quality()])
    row = store.get_metric("Звезда качества")["rows"][0]
    for field in ("plan_status", "pop_status", "trend_status", "peer_status"):
        assert row[field] is None
    assert not row["is_anomaly"]


# --------------------------------------------------------------------------- #
# situation_overview
# --------------------------------------------------------------------------- #

def test_situation_overview_star_section():
    store = _store([
        _node("Продажи", 100, children=[_node("Лиды", 40)]),
        _quality(), _sales(),
    ])
    result = analytics.build_situation_overview(store)
    zones = (result["problems"] or []) + result["positives"] + result["stable"]
    assert not {"Звезда качества", "Звезда продаж"} & {z.get("metric") for z in zones}
    stars = result["star"]["stars"]
    assert result["star"]["date"] == DATE
    assert [s["star"] for s in stars] == ["Звезда качества", "Звезда продаж"]
    quality = stars[0]
    assert quality["received"] is False
    assert [c["metric"] for c in quality["missed_children"]] == ["CSI"]
    assert [c["metric"] for c in quality["ok_children"]] == ["FCR"]
    assert [c["metric"] for c in quality["other_children"]] == ["Оценки"]
    rendered = _render_overview(result)
    assert f"Звёзды (за {DATE}):" in rendered
    assert "«Звезда качества» — НЕ получена. Ниже плана: CSI" in rendered
    assert "«Звезда продаж» — получена." in rendered
    assert "заработана" not in rendered


# --------------------------------------------------------------------------- #
# Реальный сэмпл
# --------------------------------------------------------------------------- #

def test_sample_star_json_shape():
    with open(SAMPLE, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = load_dataset_obj(data)
    stars = [r for r in rows if r["star_received"] is not None]
    assert sorted(r["metric_name"] for r in stars) == ["Звезда качества", "Звезда продаж"]
    store = SqliteStore()
    store.load(rows)
    compute_analytics(store)
    result = store.star_status()
    by = {s["star"]: s for s in result["stars"]}
    assert by["Звезда качества"]["received"] is False
    assert by["Звезда продаж"]["received"] is True
    missed = [c["metric_name"] for c in by["Звезда качества"]["children"]
              if c["plan_status"] == "хуже_плана"]
    assert sorted(missed) == ["CSI", "Повторные обращения"]


def test_star_rules_point_to_metric_tree_for_dynamics():
    """Динамики у звезды нет: на вопрос о ней правила отправляют к metric_tree по
    влияющим показателям (e2e: модель иначе ограничивалась последней датой)."""
    from langgraph_executor.aegra_agents.json_analyzer_v5.prompts import STAR_RULES
    assert "ДИНАМИКИ У ЗВЕЗДЫ НЕТ" in STAR_RULES
    assert "metric_tree(metric='X')" in STAR_RULES
