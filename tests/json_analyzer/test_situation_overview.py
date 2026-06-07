"""Тесты универсального обзора ситуации build_situation_overview (без LLM/сети).

Гоняют детерминированный слой на синтетическом мини-датасете через
loader+SqliteStore+compute_analytics: классификация зон, рекурсивная причинная
цепочка произвольной глубины (вертикаль по influent_percent), разрезы-продукты,
деградация при одном уровне / без influent, и человекочитаемый рендер.
"""
from langgraph_executor.aegra_agents.json_analyzer.analytics import (
    _overview_signal,
    build_situation_overview,
    compute_analytics,
    rank_elements,
)
from langgraph_executor.aegra_agents.json_analyzer.loader import load_dataset_obj
from langgraph_executor.aegra_agents.json_analyzer.sqlite_store import SqliteStore
from langgraph_executor.aegra_agents.json_analyzer.tools import (
    _render_overview,
    _render_rank_elements,
)


def _m(name, mtype, measure, fact, plan, date, *, influent=None, element=None, children=None):
    node = {
        "id": f"{name}-{element or ''}-{date}",
        "metric_name": name,
        "metric_description": "",
        "metric_type": mtype,
        "measure_type": measure,
        "date": date,
        "calc_period": "день",
        "fact": fact,
        "plan": plan,
        "benchmark": None,
        "element": element,
    }
    if influent is not None:
        node["influent_percent"] = influent
    if children:
        node["child_metrics"] = children
    return node


def _tree(date, f):
    """Дерево метрик одного периода: Производительность(прямая) → AHT(90%) →
    ACD(60%) → ACD_sub(100%); AHT также имеет HOLD(30%) и разрезы-продукты Talk;
    + Качество (позитив) и Дисциплина (стабильно) как одноуровневые корни."""
    return [
        _m("Производительность", "прямая", "у.е.", f["perf"], 18, date, children=[
            _m("AHT", "обратная", "секунда", f["aht"], 180, date, influent=90, children=[
                _m("ACD", "обратная", "секунда", f["acd"], 90, date, influent=60, children=[
                    _m("ACD_sub", "обратная", "секунда", f["acdsub"], 40, date, influent=100, children=[
                        _m("ACD_leaf", "обратная", "секунда", f["acdleaf"], 20, date, influent=100),
                    ]),
                ]),
                _m("HOLD", "обратная", "секунда", f["hold"], 45, date, influent=30),
                _m("Talk", "обратная", "секунда", f["segX"], None, date, element="ПродуктX"),
                _m("Talk", "обратная", "секунда", f["segY"], None, date, element="ПродуктY"),
            ]),
            _m("Adherence", "прямая", "%", f["adher"], 80, date, influent=6),
        ]),
        _m("Качество", "прямая", "%", f["kach"], 90, date),
        _m("Дисциплина", "прямая", "%", f["disc"], 100, date),
    ]


def _store(employees_metrics):
    data = {
        "me": {"tabnum": 1, "fio": "Босс", "post": "рук", "depart": "Отдел", "metrics": []},
        "employees": [
            {"tabnum": 2, "fio": "Иванов", "post": "оператор", "depart": "Отдел",
             "metrics": employees_metrics},
        ],
    }
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    compute_analytics(store)
    return store


_D1 = {"perf": 20, "aht": 200, "acd": 100, "acdsub": 50, "acdleaf": 30, "hold": 50,
       "segX": 5, "segY": 10, "adher": 85, "kach": 95, "disc": 100}
_D2 = {"perf": 12, "aht": 300, "acd": 180, "acdsub": 120, "acdleaf": 90, "hold": 40,
       "segX": 20, "segY": 2, "adher": 95, "kach": 96, "disc": 100}


def _full_overview():
    store = _store(_tree("2026-05-04", _D1) + _tree("2026-05-11", _D2))
    return build_situation_overview(store)


def test_zones_classification():
    o = _full_overview()
    assert o["person_fio"] == "Иванов"
    assert o["date"] == "2026-05-11" and o["prev_date"] == "2026-05-04"
    assert o["single_level"] is False
    problem_names = {p["metric"] for p in o["problems"]}
    assert "Производительность" in problem_names
    assert "Качество" in {h["metric"] for h in o["positives"]}
    assert "Дисциплина" in {h["metric"] for h in o["stable"]}


def test_recursive_chain_depth_gt_two():
    o = _full_overview()
    perf = next(p for p in o["problems"] if p["metric"] == "Производительность")
    # вертикаль по бизнес-весу: AHT(90) — главный драйвер
    assert perf["drivers"][0]["metric"] == "AHT"
    assert perf["drivers"][0]["influent_percent"] == 90
    aht = perf["main_driver"]
    assert aht is not None and aht["metric"] == "AHT"
    acd = aht["main_driver"]
    assert acd is not None and acd["metric"] == "ACD"
    acd_sub = acd["main_driver"]
    assert acd_sub is not None and acd_sub["metric"] == "ACD_sub"  # 3-й переход (>2)
    # лист цепочки: ACD_leaf — драйвер ACD_sub (сам без детей → не разворачивается)
    assert acd_sub["drivers"][0]["metric"] == "ACD_leaf"
    assert acd_sub["main_driver"] is None


def test_by_segments_products_by_value():
    """Разрезы (продукты) под AHT ранжируются по ЗНАЧЕНИЮ с учётом направления:
    Talk обратная (выше=хуже) → ПродуктX (20) худший, ПродуктY (2) лучший."""
    o = _full_overview()
    perf = next(p for p in o["problems"] if p["metric"] == "Производительность")
    seg = perf["main_driver"]["by_segments"]  # разрезы под AHT
    assert seg is not None and seg["label"] == "Talk"
    assert seg["worst"][0]["element"] == "ПродуктX"  # худший по значению
    assert seg["best"][0]["element"] == "ПродуктY"   # лучший по значению


def test_rank_metric_excluded_from_drivers():
    """Сиблинги без influent (напр. РАНГ-метрики) не попадают в драйверы, когда
    у компонентов веса заданы."""
    o = _full_overview()
    perf = next(p for p in o["problems"] if p["metric"] == "Производительность")
    driver_names = {d["metric"] for d in perf["drivers"]}
    assert driver_names <= {"AHT", "Adherence"}  # только взвешенные компоненты


def test_single_level_dataset_degrades_gracefully():
    """Только одноуровневые метрики без детей: single_level=True, причинной
    цепочки нет, но зоны строятся."""
    flat = [
        _m("Конверсия", "прямая", "%", 70, 90, "2026-05-04"),
        _m("Конверсия", "прямая", "%", 60, 90, "2026-05-11"),
    ]
    o = build_situation_overview(_store(flat))
    assert o["single_level"] is True
    conv = next(p for p in o["problems"] if p["metric"] == "Конверсия")
    assert conv["drivers"] == []
    assert conv.get("main_driver") is None


def test_no_influent_uses_movement_heuristic():
    """Дети без influent_percent — равные «бизнес-веса»: ранжируем по величине
    изменения и помечаем как эвристику."""
    # Родитель с двумя детьми без influent, разная динамика.
    def node(date, parent_fact, a, b):
        return [_m("Итог", "прямая", "у.е.", parent_fact, 100, date, children=[
            _m("ДрайверA", "прямая", "у.е.", a, 50, date),
            _m("ДрайверB", "прямая", "у.е.", b, 50, date),
        ])]
    store = _store(node("2026-05-04", 100, 50, 50) + node("2026-05-11", 70, 48, 10))
    o = build_situation_overview(store)
    itog = next(p for p in o["problems"] if p["metric"] == "Итог")
    # ДрайверB изменился сильнее (50→10) — он первым в ранжировании по движению
    assert itog["drivers"][0]["metric"] == "ДрайверB"
    assert itog["note"] and "эвристик" in itog["note"]


def test_signal_skips_exact_zero_pop_change():
    """pop_change_pct=0.0 (ровно ноль) не должен блокировать ранжирование по
    отклонению от плана — иначе метрика молча получит силу 0."""
    assert _overview_signal({"pop_change_pct": 0.0, "plan_dev_pct": -15.5}) == 15.5
    assert _overview_signal({"pop_change_pct": -8.0, "plan_dev_pct": -15.5}) == 8.0
    assert _overview_signal({"pop_change_pct": 0.0, "plan_dev_pct": 0.0}) == 0.0
    assert _overview_signal({"pop_change_pct": None, "plan_dev_pct": None}) == 0.0


def test_missing_analytics_returns_note_not_stable():
    """Без compute_analytics зоны не строятся: честная пометка, а не молчаливый
    «stable» по NULL-вердиктам."""
    data = {
        "me": {"tabnum": 1, "fio": "Босс", "post": "рук", "depart": "О", "metrics": []},
        "employees": [{"tabnum": 2, "fio": "Иванов", "post": "оп", "depart": "О",
                       "metrics": _tree("2026-05-11", _D2)}],
    }
    store = SqliteStore()
    store.load(load_dataset_obj(data))  # НАМЕРЕННО без compute_analytics
    o = build_situation_overview(store)
    assert o["problems"] == [] and o["positives"] == [] and o["stable"] == []
    assert "не посчитана" in (o.get("note") or "")


def test_render_overview_phrased_with_numbers():
    o = _full_overview()
    out = _render_overview(o)
    assert "Проблемные зоны:" in out
    assert "Производительность" in out and "AHT" in out
    assert "вес 90%" in out          # бизнес-вес в драйвере
    assert "по разрезам" in out       # продукты Talk
    assert "Позитив:" in out and "Стабильно:" in out


# --- Устойчивость к null person_tabnum (прод: табельный не приходит) -----------

def _flat(name, fact, plan, date):
    return {"id": f"{name}{date}", "metric_name": name, "metric_type": "прямая",
            "measure_type": "у.е.", "date": date, "calc_period": "д", "fact": fact,
            "plan": plan, "benchmark": None, "element": None}


def _store_no_tabnum():
    """Руководитель и сотрудник БЕЗ tabnum (разные ФИО), одна метрика с разными
    значениями — моделирует прод, где табельный не приходит."""
    data = {
        "me": {"fio": "Босс", "post": "рук", "depart": "О", "metrics": [
            _flat("Производительность", 99, 18, "2026-05-04"),
            _flat("Производительность", 99, 18, "2026-05-11")]},
        "employees": [{"fio": "Иванов", "post": "оп", "depart": "О", "metrics": [
            _flat("Производительность", 20, 18, "2026-05-04"),
            _flat("Производительность", 12, 18, "2026-05-11")]}],
    }
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    compute_analytics(store)
    return store


def test_null_tabnum_people_not_collapsed():
    store = _store_no_tabnum()
    people = store.schema_overview()["people"]
    fios = sorted(p["person_fio"] for p in people)
    assert fios == ["Босс", "Иванов"]  # не схлопнулись в одного по NULL tabnum


def test_null_tabnum_series_not_mixed():
    store = _store_no_tabnum()
    row = store.conn.execute(
        "SELECT a.pop_change_pct FROM metrics m JOIN metric_analytics a "
        "ON a.metric_uid = m.metric_uid WHERE m.person_fio = 'Иванов' "
        "AND m.date = '2026-05-11' AND m.element IS NULL"
    ).fetchone()
    assert round(row["pop_change_pct"], 2) == -40.0  # (12-20)/20, не смешано с рук.


def test_null_tabnum_situation_overview_focus_by_fio():
    store = _store_no_tabnum()
    o_default = build_situation_overview(store)
    assert o_default.get("error") is None
    assert o_default["person_fio"] == "Иванов"  # единственный сотрудник, не «нет людей»
    assert any(p["metric"] == "Производительность" for p in o_default["problems"])
    o_by_fio = build_situation_overview(store, person="Иванов")
    assert o_by_fio["person_fio"] == "Иванов"


# --- rank_elements: лучшие/худшие разрезы по значению (без плана) --------------

def _influence_store(direction, facts, date="2026-05-11"):
    """Один сотрудник, метрика «Влияние» с element-разрезами БЕЗ плана."""
    metrics = [
        _m("Влияние", direction, "секунда", f, None, date, element=el)
        for el, f in facts.items()
    ]
    data = {"me": {"fio": "Босс", "metrics": []},
            "employees": [{"fio": "Иванов", "metrics": metrics}]}
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    compute_analytics(store)
    return store


def test_rank_elements_inverse_no_plan():
    s = _influence_store("обратная", {"A": 25, "B": -20, "C": 5})
    r = rank_elements(s, "Влияние")
    assert r.get("error") is None and r["count"] == 3
    assert r["worst"][0]["element"] == "A"   # обратная: выше=хуже
    assert r["best"][0]["element"] == "B"    # ниже=лучше


def test_rank_elements_direct_no_plan():
    s = _influence_store("прямая", {"A": 25, "B": -20, "C": 5})
    r = rank_elements(s, "Влияние")
    assert r["best"][0]["element"] == "A"    # прямая: выше=лучше
    assert r["worst"][0]["element"] == "B"


def test_rank_elements_render_phrased():
    out = _render_rank_elements({
        "metric": "Влияние", "metric_type": "обратная", "date": "2026-05-11",
        "person_fio": "Иванов", "count": 8, "top": 5,
        "worst": [{"element": "A", "fact": 25, "measure_type": "секунда"}],
        "best": [{"element": "B", "fact": -20, "measure_type": "секунда"}],
    })
    assert "Худшие" in out and "A 25" in out and "Лучшие" in out


def _overview_with_segments(seg_d1, seg_d2):
    """Производительность(прямая) ← AHT(обратная, 90%) ← Talk-продукты (обратная)."""
    def tree(date, perf, segs):
        return _m("Производительность", "прямая", "у.е.", perf, 18, date, children=[
            _m("AHT", "обратная", "секунда", 320, 180, date, influent=90, children=[
                _m("Talk", "обратная", "секунда", v, None, date, element=el)
                for el, v in segs.items()
            ]),
        ])
    data = {"me": {"fio": "Босс", "metrics": []},
            "employees": [{"fio": "Иванов", "metrics": [
                tree("2026-05-04", 20, seg_d1), tree("2026-05-11", 12, seg_d2)]}]}
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    compute_analytics(store)
    return store


def test_by_segments_worst_by_value_not_most_changed():
    """Регресс бага: худший разрез по ЗНАЧЕНИЮ (A=25, стабилен) всплывает первым,
    а не самый ИЗМЕНИВШИЙСЯ (C: 4→18)."""
    s = _overview_with_segments({"A": 24, "C": 4, "B": -20}, {"A": 25, "C": 18, "B": -20})
    o = build_situation_overview(s)
    perf = next(p for p in o["problems"] if p["metric"] == "Производительность")
    seg = perf["main_driver"]["by_segments"]
    assert seg["worst"][0]["element"] == "A"  # худший по значению, не C (самый изменившийся)
