"""json_analyzer_v3: инструмент element_slice — кросс-метричный срез по одному
element (продукту/разрезу).

Корень бага, который он закрывает: element-разрез метрики хранится СИБЛИНГОМ её
агрегата (тот же parent_uid и depth), а не ребёнком. Поэтому обход дерева по
parent_uid (metric_tree) разрезы СОСЕДНИХ метрик уровня не находит, и агент
ложно отвечает «по этому продукту у других метрик данных нет». element_slice
делает плоский запрос по колонке element и видит разрезы ВСЕХ метрик сразу.
"""
from langgraph_executor.aegra_agents.json_analyzer_v3.analytics import compute_analytics
from langgraph_executor.aegra_agents.json_analyzer_v3.loader import load_dataset_obj
from langgraph_executor.aegra_agents.json_analyzer_v3.sqlite_store import SqliteStore
from langgraph_executor.aegra_agents.json_analyzer_v3.store_cache import EmbeddingIndex
from langgraph_executor.aegra_agents.json_analyzer_v3.tools import build_tools

DATE = "2026-05-11"


def _node(name, fact, element=None, plan=None, mtype="прямая", children=None):
    return {
        "id": f"{name}-{element}", "metric_name": name, "metric_type": mtype,
        "measure_type": "секунда", "date": DATE, "calc_period": "д",
        "fact": fact, "plan": plan, "benchmark": None, "element": element,
        "child_metrics": children or [],
    }


def _store():
    """Дерево, как в реальных данных: под одним родителем агрегат и его разрезы —
    сиблинги; одни и те же продукты (element) встречаются у нескольких метрик;
    «Влияние тематик» — agg-less (только разрезы, без агрегата)."""
    perf = _node("Производительность", 100, children=[
        _node("AHT", 300, plan=320, mtype="обратная"),                         # агрегат
        _node("AHT", 299, element="Банковские счета", plan=320, mtype="обратная"),
        _node("AHT", 310, element="Кредитование", plan=320, mtype="обратная"),
        _node("Доля переводов", 18, element="Банковские счета", plan=10),
        _node("Доля переводов", 22, element="Кредитование", plan=10),
        _node("Влияние тематик", 20, element="Банковские счета"),              # agg-less
        _node("Влияние тематик", 25, element="Кредитование"),
    ])
    quality = _node("Качество", 90, children=[
        _node("CSI", 4.5, element="Банковские счета", plan=4.0),
    ])
    data = {"me": {"fio": "Босс", "metrics": [perf, quality]}, "employees": []}
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    compute_analytics(store)
    return store


def _names(result):
    return {r["metric_name"] for r in result["rows"]}


def test_element_slice_returns_all_metrics_for_one_element():
    """Главный кейс: по продукту возвращаются ВСЕ метрики уровня, включая
    agg-less «Влияние тематик» и метрику из другой ветки (CSI)."""
    store = _store()
    res = store.element_slice("Банковские счета", date=DATE)
    assert res["count"] == 4
    assert _names(res) == {"AHT", "Доля переводов", "Влияние тематик", "CSI"}


def test_metric_tree_misses_sibling_segments_but_slice_finds_them():
    """Регрессия на корень бага: metric_tree(AHT) не показывает разрезы соседних
    метрик по тому же продукту, а element_slice — показывает."""
    store = _store()
    tree = store.metric_tree(name="AHT")
    tree_pairs = {(r["metric_name"], r.get("element")) for r in tree["rows"]}
    # В поддереве AHT нет разреза «Доля переводов [Банковские счета]» — он сиблинг.
    assert ("Доля переводов", "Банковские счета") not in tree_pairs
    # element_slice эту дыру закрывает.
    res = store.element_slice("Банковские счета", date=DATE)
    assert ("Доля переводов") in _names(res)


def test_element_slice_parent_scopes_to_one_branch():
    """parent ограничивает срез прямыми детьми указанной метрики-родителя."""
    store = _store()
    only_perf = store.element_slice("Банковские счета", date=DATE, parent="Производительность")
    assert _names(only_perf) == {"AHT", "Доля переводов", "Влияние тематик"}  # без CSI
    only_quality = store.element_slice("Банковские счета", date=DATE, parent="Качество")
    assert _names(only_quality) == {"CSI"}


def test_element_slice_tool_registered_and_renders_metric_column():
    store = _store()
    tools = build_tools(store, EmbeddingIndex([]), lambda q: [0.0])
    names = [t.name for t in tools]
    assert "element_slice" in names
    tool = [t for t in tools if t.name == "element_slice"][0]
    out = tool.invoke({"element": "Банковские счета", "date": DATE})
    assert "метрика" in out  # колонка имени метрики обязательна для кросс-метричного среза
    assert "AHT" in out and "Доля переводов" in out and "Влияние тематик" in out


def test_element_slice_requires_element():
    store = _store()
    tools = build_tools(store, EmbeddingIndex([]), lambda q: [0.0])
    tool = [t for t in tools if t.name == "element_slice"][0]
    out = tool.invoke({"element": ""})
    assert "element обязателен" in out
