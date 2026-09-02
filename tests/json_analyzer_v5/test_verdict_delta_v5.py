"""json_analyzer_v5: вердикт нельзя оторвать от его величины, план виден значением.

Дефект, который здесь закрывается: в выдаче стояли ОТДЕЛЬНО «факт 22.99»,
«статус плана: лучше плана» и «отклонение от плана: 27.72 %», а самого значения
плана не было. Модель достраивала план из процента («план = 27.72») и писала
«22.99 ниже плана», хотя метрика план ПЕРЕВЫПОЛНЯЕТ. Второй источник переворота —
знак процента: он показывает направление ЗНАЧЕНИЯ, а не «хорошо/плохо» (у
'обратной' метрики плюс — это хуже плана).
"""
from langgraph_executor.aegra_agents.json_analyzer_v5.analytics import compute_analytics
from langgraph_executor.aegra_agents.json_analyzer_v5.loader import load_dataset_obj
from langgraph_executor.aegra_agents.json_analyzer_v5.sqlite_store import SqliteStore
from langgraph_executor.aegra_agents.json_analyzer_v5.tools import (
    _render_rows,
    _render_tree,
    _verdict_delta,
)

DATE = "2026-07-30"


def _node(name, fact, plan, metric_type="прямая", children=None):
    return {
        "id": f"{name}-id", "metric_name": name, "metric_type": metric_type,
        "measure_type": "у.е.", "date": DATE, "calc_period": "Месяц",
        "fact": fact, "plan": plan, "benchmark": None, "element": None,
        "child_metrics": children or [],
    }


def _store(metrics):
    data = {"me": {"fio": "Босс", "metrics": metrics}, "employees": []}
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    compute_analytics(store)
    return store


def test_plan_value_rendered_next_to_fact():
    """Значение плана есть в выдаче — достраивать его из процента не нужно."""
    store = _store([_node("Производительность", 22.99, 18.0)])
    rendered = _render_rows(store.get_metric("Производительность"))
    assert "план" in rendered
    assert "18 у.е." in rendered


def test_plan_verdict_carries_its_own_magnitude():
    """«лучше плана на 27.72 %» одной ячейкой, отдельного числа рядом нет."""
    store = _store([_node("Производительность", 22.99, 18.0)])
    rendered = _render_rows(store.get_metric("Производительность"))
    assert "лучше плана на 27.72 %" in rendered
    # Отдельной колонки с голым отклонением больше нет — её нечего перетолковать.
    assert "отклонение от плана" not in rendered


def test_inverse_metric_worse_than_plan_has_no_misleading_sign():
    """У 'обратной' метрики факт ВЫШЕ плана — это «хуже плана», без плюса в числе."""
    store = _store([_node("AHT", 446.4, 320.0, metric_type="обратная")])
    rendered = _render_rows(store.get_metric("AHT"))
    assert "хуже плана на 39.5 %" in rendered
    assert "хуже плана на +39.5" not in rendered


def test_in_plan_verdict_has_no_magnitude():
    """«в плане на 0.4 %» — бессмыслица: у нейтрального вердикта величины нет."""
    store = _store([_node("Adherence", 100.2, 100.0)])
    rendered = _render_rows(store.get_metric("Adherence"))
    assert "в плане" in rendered
    assert "в плане на" not in rendered


def test_tree_node_shows_plan_and_bound_verdict():
    store = _store([
        _node("Производительность", 22.99, 18.0, children=[_node("AHT", 446.4, 320.0)])
    ])
    rendered = _render_tree(store.metric_tree("Производительность"))
    assert "план 18 у.е." in rendered
    assert "лучше плана на 27.72 %" in rendered


def test_verdict_delta_unit_fallback_when_percent_is_absent():
    """Для знаковых/индексных метрик % обнулён — величину даём абсолютом с единицей."""
    assert _verdict_delta("хуже_плана", None, -3.0, "место") == "хуже плана на 3 место"
    assert _verdict_delta(None, 1.0, 1.0, None) == ""
