"""Тесты человекочитаемого рендера выдачи инструментов json_analyzer (без LLM/БД).

Проверяют, что row-выдачи рендерятся Markdown-таблицей с русскими заголовками,
единицами инлайн и без сырых trend/zscore, дерево — отступами, а _safe
откатывается на JSON при ошибке рендера.
"""
from langgraph_executor.aegra_agents.json_analyzer.tools import (
    _md_table,
    _render_rows,
    _render_tree,
    _safe,
)


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
