"""json_analyzer_v3: строки с пустым фактом (fact=None/"") не грузятся в базу.

Зачем: пустая точка в ряду по датам затеняет последнюю РЕАЛЬНУЮ (analytics
берёт prev = непосредственно предыдущий период), и сравнение период-к-периоду
не считается. Отбрасывая пустые ЛИСТЬЯ при загрузке, ряд сшивается только по
реальным точкам — pop_change снова считается через «дыру» с пустым периодом.
Ноль (0/0.0) — валидный факт и НЕ отбрасывается.
"""
from langgraph_executor.aegra_agents.json_analyzer_v3.analytics import compute_analytics
from langgraph_executor.aegra_agents.json_analyzer_v3.loader import load_dataset_obj
from langgraph_executor.aegra_agents.json_analyzer_v3.sqlite_store import SqliteStore


def _node(name, fact, date, element=None, plan=None, mtype="прямая", children=None):
    return {
        "id": f"{name}-{element}-{date}", "metric_name": name, "metric_type": mtype,
        "measure_type": "секунда", "date": date, "calc_period": "д",
        "fact": fact, "plan": plan, "benchmark": None, "element": element,
        "child_metrics": children or [],
    }


def _rows(metrics_by_person_me):
    data = {"me": {"fio": "Босс", "metrics": metrics_by_person_me}, "employees": []}
    return load_dataset_obj(data)


def test_empty_fact_leaf_is_skipped():
    rows = _rows([
        _node("AHT", 300, "2026-05-11"),
        _node("AHT", None, "2026-05-12"),       # пустой факт — выкидываем
        _node("AHT", "", "2026-05-13"),         # пустая строка — тоже
    ])
    facts = [r["fact"] for r in rows if r["metric_name"] == "AHT"]
    assert facts == [300]


def test_zero_fact_is_kept():
    """Ноль — валидное значение, не «пусто»."""
    rows = _rows([_node("Доля переводов", 0, "2026-05-11")])
    assert len(rows) == 1
    assert rows[0]["fact"] == 0


def test_empty_fact_parent_with_children_is_kept():
    """Узел-агрегат без собственного факта, но с детьми, сохраняем — иначе
    потеряли бы реальные разрезы под ним."""
    rows = _rows([
        _node("Производительность", None, "2026-05-11", children=[
            _node("AHT", 300, "2026-05-11"),
        ]),
    ])
    names = {r["metric_name"] for r in rows}
    assert names == {"Производительность", "AHT"}  # оба на месте
    aht = [r for r in rows if r["metric_name"] == "AHT"][0]
    perf = [r for r in rows if r["metric_name"] == "Производительность"][0]
    assert aht["parent_uid"] == perf["metric_uid"]  # связь родитель-ребёнок цела


def test_pop_change_bridges_empty_period():
    """Главный кейс: между двумя реальными точками вклинился период с пустым
    фактом. После отбрасывания пустой строки pop_change считается напрямую
    между реальными точками (а не гасится пустым prev)."""
    store = SqliteStore()
    store.load(_rows([
        _node("AHT", 300, "2026-05-11", plan=320, mtype="обратная"),
        _node("AHT", None, "2026-05-12", plan=320, mtype="обратная"),  # дыра
        _node("AHT", 330, "2026-05-13", plan=320, mtype="обратная"),
    ]))
    compute_analytics(store)
    row = store.conn.execute(
        "SELECT a.pop_change_abs FROM metrics m JOIN metric_analytics a "
        "ON a.metric_uid = m.metric_uid WHERE m.date = '2026-05-13'"
    ).fetchone()
    assert row["pop_change_abs"] == 30  # 330 - 300, дыра 05-12 перешагнута
