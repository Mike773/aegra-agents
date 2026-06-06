"""Тесты человекочитаемого рендера выдачи инструментов (без LLM/БД).

Проверяют, что attribute_* рендерятся фразовым списком с «Главный драйвер»,
row-таблицы — Markdown с русскими заголовками/единицами и без сырых trend/zscore,
а _safe откатывается на JSON при ошибке рендера.
"""
from langgraph_executor.aegra_agents.json_analyzer_causal.tools import (
    _md_table,
    _render_attribution,
    _render_rows,
    _render_tree,
    _safe,
)


def test_attribution_causal_phrased_list():
    result = {
        "method": "causal",
        "target": "Производительность",
        "target_type": "прямая",
        "date_old": "2026-05-04",
        "date_new": "2026-05-11",
        "cohort_size": 25,
        "contributions": [
            {"node": "AHT", "share_pct": 43.9, "verdict": "улучшение"},
            {"node": "Adherence", "share_pct": 19.8, "verdict": "ухудшение"},
        ],
    }
    out = _render_attribution(result)
    assert "доля вклада 43.9%" in out
    assert "вердикт: улучшение" in out
    assert "Главный драйвер: AHT (43.9%, улучшение)." in out
    assert "каузальный" in out
    # русский рендер, не JSON-ключи
    assert "share_pct" not in out and '"node"' not in out


def test_attribution_algebraic_marks_heuristic():
    result = {
        "method": "algebraic",
        "target": "Производительность",
        "target_type": "прямая",
        "date_old": "2026-05-04",
        "date_new": "2026-05-11",
        "parent_old": 13.74,
        "parent_new": 12.05,
        "parent_delta": -1.69,
        "contributions": [
            {"node": "AHT", "share_pct": 56.5, "child_verdict": "улучшение"},
        ],
    }
    out = _render_attribution(result)
    assert "эвристический" in out
    assert "вердикт: улучшение" in out  # из child_verdict
    assert "Примечание" in out  # пометка про эвристику


def test_attribution_anomaly_self_mechanism():
    result = {
        "method": "anomaly",
        "target": "Производительность",
        "target_type": "прямая",
        "date": "2026-05-11",
        "person_tabnum": 6762565,
        "reference_cohort_size": 25,
        "contributions": [
            {"node": "(собственный механизм …)", "share_pct": 100.0,
             "is_self": True, "verdict": "—"},
            {"node": "AHT", "share_pct": 0.0, "is_self": False, "verdict": "нейтрально"},
        ],
    }
    out = _render_attribution(result)
    assert "объясняется детьми" in out


def test_md_table_drops_empty_columns_and_has_headers():
    rows = [{"a": "x", "b": None}, {"a": "y", "b": None}]
    cols = [("колонка_A", lambda r: r.get("a") or ""),
            ("колонка_B", lambda r: r.get("b") or "")]
    out = _md_table(rows, cols)
    assert "колонка_A" in out
    assert "колонка_B" not in out  # пустой столбец отсечён
    assert out.count("\n") >= 3  # заголовок + разделитель + 2 строки


def test_render_rows_units_inline_and_no_raw_direction():
    result = {
        "count": 1,
        "rows": [{
            "person_fio": "Иванов А.С.", "metric_name": "AHT",
            "measure_type": "секунда", "date": "2026-05-11", "fact": 352.6,
            "plan_status": "хуже_плана", "plan_dev_pct": 12.3,
            "pop_status": "ухудшение", "pop_change_pct": 5.1,
            "trend_status": "ухудшение",
            # сырые поля, которые НЕ должны попасть в таблицу:
            "trend": "рост", "zscore": 1.84, "metric_type": "обратная",
        }],
    }
    out = _render_rows(result)
    assert "352.6 секунда" in out  # единица инлайн
    assert "план_статус" in out and "динамика" in out
    assert "рост" not in out  # сырой trend выкинут
    assert "1.84" not in out  # сырой zscore выкинут


def test_render_tree_indented_with_element():
    result = {"rows": [
        {"depth": 1, "metric_name": "Производительность", "measure_type": "у.е.",
         "fact": 13.23, "plan_status": "хуже_плана", "plan_dev_pct": -26.5,
         "pop_status": "улучшение", "person_fio": "Козлов А.Н."},
        {"depth": 2, "metric_name": "Доля переводов", "element": "Кредитование",
         "measure_type": "%", "fact": 13.7, "influent_percent": 4.0,
         "pop_status": "ухудшение", "person_fio": "Козлов А.Н."},
    ]}
    out = _render_tree(result)
    assert "- Производительность: 13.23 у.е." in out
    assert "Доля переводов [Кредитование]" in out
    assert "влияние 4%" in out


def test_safe_falls_back_to_json_on_error():
    def boom(_):
        raise ValueError("render bug")
    out = _safe(boom, {"k": "v"})
    assert out == '{"k": "v"}'
