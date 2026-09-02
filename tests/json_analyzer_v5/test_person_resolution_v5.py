"""json_analyzer_v5: аргумент person инструментов — резолв роли/должности в ФИО.

На проме модель передавала в star_status person="руководитель" или
person="Начальник сектора" (должность) и получала «не найден» без списка людей.
Теперь: роль («руководитель», «начальник», «босс») → человек с person_is_me;
должность/подразделение → единственный подходящий человек; ФИО/табельный — как
раньше. Неоднозначность или промах — ошибка с перечнем людей датасета, а «Состав
датасета» называет ФИО с ролями заранее.
"""
from langgraph_executor.aegra_agents.json_analyzer_v5.agent_base import format_facts
from langgraph_executor.aegra_agents.json_analyzer_v5.analytics import compute_analytics
from langgraph_executor.aegra_agents.json_analyzer_v5.loader import load_dataset_obj
from langgraph_executor.aegra_agents.json_analyzer_v5.sqlite_store import SqliteStore
from langgraph_executor.aegra_agents.json_analyzer_v5.store_cache import EmbeddingIndex
from langgraph_executor.aegra_agents.json_analyzer_v5.tools import build_tools

DATE = "2026-07-30"


def _metric(name, fact, **extra):
    node = {
        "id": f"{name}-id", "metric_name": name, "metric_type": "прямая",
        "measure_type": "%", "date": DATE, "calc_period": "Месяц",
        "fact": fact, "plan": 10, "benchmark": None, "element": None,
        "child_metrics": [],
    }
    node.update(extra)
    return node


def _star(name, received, children):
    return {"id": f"{name}-id", "metric_name": name, "date": DATE,
            "calc_period": "месяц", "element": None, "star_received": received,
            "is_star_metric": True, "child_metrics": children}


def _store(employees=None, me=None):
    data = {
        "me": me if me is not None else {
            "tabnum": 1000, "fio": "Иванов Иван Иванович", "post": "Начальник сектора",
            "depart": "Сектор продаж", "metrics": [_metric("Продажи", 12)],
        },
        "employees": employees if employees is not None else [{
            "tabnum": 2000, "fio": "Петров Никита Сергеевич", "post": "Оператор",
            "depart": "Сектор продаж",
            "metrics": [_metric("Продажи", 8),
                        _star("Звезда качества", False,
                              [_metric("CSI", 4, plan=4.5, is_star_metric=True)])],
        }],
    }
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    compute_analytics(store)
    return store


def _tools(store):
    return {t.name: t for t in build_tools(store, EmbeddingIndex([]), lambda q: [0.0])}


# --- store.resolve_person -----------------------------------------------------

def test_resolve_by_fio_and_tabnum_unchanged():
    store = _store()
    assert store.resolve_person("Петров") == "Петров Никита Сергеевич"
    assert store.resolve_person("2000") == "Петров Никита Сергеевич"
    assert store.resolve_person(None) is None
    assert store.resolve_person("  ") is None


def test_resolve_role_word_to_manager():
    store = _store()
    for word in ("руководитель", "Руководитель", "начальник", "босс", "рук."):
        assert store.resolve_person(word) == "Иванов Иван Иванович", word


def test_resolve_by_post_and_depart_when_unique():
    store = _store()
    assert store.resolve_person("Начальник сектора") == "Иванов Иван Иванович"
    assert store.resolve_person("оператор") == "Петров Никита Сергеевич"
    # Подразделение общее для обоих — неоднозначно.
    assert store.resolve_person("Сектор продаж") is None


def test_resolve_ambiguous_or_unknown_is_none():
    store = _store(employees=[
        {"tabnum": 2000, "fio": "Петров Никита", "post": "Оператор", "metrics": [_metric("Продажи", 8)]},
        {"tabnum": 2001, "fio": "Сидоров Олег", "post": "Оператор", "metrics": [_metric("Продажи", 9)]},
    ])
    assert store.resolve_person("оператор") is None
    assert store.resolve_person("Продажи") is None


# --- инструменты --------------------------------------------------------------

def test_star_status_accepts_role_and_post():
    tools = _tools(_store())
    by_role = tools["star_status"].func(person="Оператор")
    assert "Звёзды сотрудника Петров Никита Сергеевич" in by_role
    # У руководителя звёзд нет — но аргумент принят, а не «не найден».
    by_manager = tools["star_status"].func(person="руководитель")
    assert "не найден" not in by_manager
    assert "Звёзд" in by_manager


def test_get_metric_accepts_post():
    tools = _tools(_store())
    out = tools["get_metric"].func(metric="Продажи", person="Начальник сектора")
    assert "Иванов Иван Иванович" in out
    assert "Петров" not in out


def test_unknown_person_error_lists_people():
    tools = _tools(_store())
    out = tools["get_metric"].func(metric="Продажи", person="Сектор продаж")
    assert "не найден" in out or "неоднозначн" in out
    assert "Иванов Иван Иванович" in out and "Петров Никита Сергеевич" in out
    assert "ФИО" in out


# --- состав датасета ----------------------------------------------------------

def test_format_facts_lists_people_with_roles():
    facts = format_facts(_store().schema_overview())
    assert "Иванов Иван Иванович" in facts and "Петров Никита Сергеевич" in facts
    assert "руководитель" in facts and "Начальник сектора" in facts
    assert "ФИО ДОСЛОВНО" in facts
