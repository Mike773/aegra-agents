"""describe_answer: подблок «Исходные данные» — фактура шагов пайплайна
(оргструктура, метрики, wiki, долгосрочная память) из персистентного стейта.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from langgraph_executor.aegra_agents.analytic_orchestrator_v4.nodes import (
    _describe_sources_block,
    _memory_source_line,
    _metrics_source_line,
    _wiki_source_line,
    _with_description,
    render_trace,
)

# me + один сотрудник; 2 верхнеуровневые метрики × 2 недели, у одной — ребёнок.
# Уникальных метрик по id — 3 (Производительность, РАНГ, дочерняя Обработка).
_METRICS = {
    "me": {"tabnum": "1", "fio": "Иванов Иван Иванович", "metrics": []},
    "employees": [
        {
            "tabnum": "4032085",
            "fio": "Петров Никита Сергеевич",
            "post": "Оператор",
            "metrics": [
                {
                    "id": "90022908", "metric_name": "Производительность",
                    "date": "2026-04-06", "calc_period": "неделя",
                    "fact": 25.4, "plan": 18.0,
                    "child_metrics": [
                        {"id": "90022999", "metric_name": "Обработка",
                         "date": "2026-04-06", "calc_period": "неделя",
                         "fact": 10.0, "influent_percent": 40.0},
                    ],
                },
                {
                    "id": "90022908", "metric_name": "Производительность",
                    "date": "2026-04-13", "calc_period": "неделя",
                    "fact": 20.1, "plan": 18.0, "child_metrics": [],
                },
                {
                    "id": "90022777", "metric_name": "РАНГ Производительность",
                    "date": "2026-04-06", "calc_period": "неделя",
                    "fact": 3.0, "child_metrics": [],
                },
                {
                    "id": "90022777", "metric_name": "РАНГ Производительность",
                    "date": "2026-04-13", "calc_period": "неделя",
                    "fact": 2.0, "child_metrics": [],
                },
            ],
        },
    ],
}


def test_metrics_source_line_counts_and_periods():
    line = _metrics_source_line(_METRICS)
    assert "всего уникальных метрик — 3" in line
    # Верхнеуровневые — только корни, без дочерней «Обработка».
    assert "верхнеуровневые: Производительность, РАНГ Производительность" in line
    assert "Обработка" not in line.split("верхнеуровневые:")[1].split(";")[0]
    assert "неделя — 2 (2026-04-06 … 2026-04-13)" in line


def test_sources_block_org_and_position():
    block = _describe_sources_block({"metrics": _METRICS, "position": "Оператор"})
    assert "руководитель — Иванов Иван Иванович" in block
    assert "в фокусе анализа — Петров Никита Сергеевич (позиция: Оператор)" in block
    assert block.startswith("**Исходные данные:**")


def test_sources_block_without_dataset():
    block = _describe_sources_block({"metrics_error": "сервис недоступен"})
    assert "Оргструктура: данные не загружены (сервис недоступен)." in block
    assert "Метрики: не загружены." in block


def test_wiki_source_line_variants():
    with_snippets = _wiki_source_line({"easyrag_snippets": [
        {"page_title": "Производительность", "section_title": "Методика расчёта",
         "similarity": 0.82, "body_md": "тело"},
        {"page_title": "Нормативы", "section_title": "Оператор",
         "similarity": 0.7, "body_md": "тело"},
    ]})
    assert "найдено фрагментов — 2" in with_snippets
    assert "«Производительность / Методика расчёта»" in with_snippets
    # Технические значения (similarity) и тело фрагмента в опись не попадают.
    assert "sim" not in with_snippets
    assert "0.82" not in with_snippets
    assert "тело" not in with_snippets

    assert "не запрашивалась" in _wiki_source_line({})
    # Текст исключения пользователю не показываем — только факт сбоя.
    err_line = _wiki_source_line({"easyrag_error": "OSError: connect failed"})
    assert "не удалось получить данные" in err_line
    assert "OSError" not in err_line
    empty = _wiki_source_line({"easyrag_query": "методика расчёта"})
    assert "«методика расчёта» ничего не найдено" in empty
    stubs = _wiki_source_line({
        "easyrag_query": "AHT",
        "easyrag_stub_pages": [{"slug": "aht", "title": "AHT"}],
    })
    assert "страницы-заглушки: AHT" in stubs


def test_memory_source_line_variants():
    # Заглушка «…отсутствует» — явная строка про пустую память, без её текста.
    stub = _memory_source_line({"memory_context": "Долгосрочная память отсутствует."})
    assert "отсутствует" in stub
    real = _memory_source_line({"memory_context": "Руководитель просил сводки кратко."})
    assert "Руководитель просил сводки кратко." in real
    err = _memory_source_line({"memory_error": "store down"})
    assert "не удалось загрузить" in err
    assert "store down" not in err


def test_render_trace_without_preamble_unchanged():
    trace = [{"stage": "route", "kind": "intent", "summary": "Определил тип."}]
    assert render_trace(trace) == (
        "---\n### Как я пришёл к выводу\n\n1. **Классификация запроса.** Определил тип."
    )
    assert render_trace([]) == ""


def test_with_description_flag_off_and_fallback():
    state = {
        "metrics": _METRICS,
        "position": "Оператор",
        "memory_context": "Долгосрочная память отсутствует.",
        "reasoning_trace": [
            {"stage": "route", "kind": "intent", "summary": "Определил тип."},
        ],
    }
    off = asyncio.run(_with_description("Ответ.", state, {"configurable": {}}))
    assert off == "Ответ."

    cfg = {"configurable": {"describe_answer": True}}
    on = asyncio.run(_with_description("Ответ.", state, cfg))
    assert on.startswith("Ответ.")
    assert "### Как я пришёл к выводу" in on
    assert "**Исходные данные:**" in on
    assert "Петров Никита Сергеевич" in on
    assert "1. **Классификация запроса.** Определил тип." in on
    # Исходные данные идут раньше шагов трассы.
    assert on.index("**Исходные данные:**") < on.index("1. **Классификация")
