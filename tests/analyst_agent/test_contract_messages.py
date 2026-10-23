"""Контракт сообщений и describe_answer — он должен совпадать с v4.

Прод-клиент уже завязан на теги orchestrator_step / orchestrator_final,
answer_html и раздел «Как я пришёл к выводу», поэтому поведение переносится
как есть.
"""
from __future__ import annotations

from _fixtures import make_dataset_obj, make_metric  # noqa: F401
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_executor.aegra_agents.analyst_agent.contract import describe, messages


def _cfg(**kw):
    return {"configurable": kw}


# --- шаговые сообщения ----------------------------------------------------

def test_step_update_tags_message():
    upd = messages.step_update(_cfg(), "📊 Посмотрел показатель")
    msg = upd["messages"][0]
    assert msg.additional_kwargs["orchestrator_step"] is True
    assert messages.is_step(msg) is True


def test_step_update_disabled_by_flag():
    assert messages.step_update(_cfg(emit_progress_messages=False), "шаг") == {}
    assert messages.step_update(_cfg(emit_progress_messages="false"), "шаг") == {}


def test_step_update_ignores_empty_text():
    assert messages.step_update(_cfg(), "   ") == {}


# --- итоговое сообщение ---------------------------------------------------

def test_final_message_html_by_default():
    msg = messages.final_message("**Итог** анализа", _cfg())
    assert msg.additional_kwargs["orchestrator_final"] is True
    assert "<strong>Итог</strong>" in msg.content
    assert msg.additional_kwargs["orchestrator_markdown"] == "**Итог** анализа"


def test_final_message_markdown_when_html_disabled():
    msg = messages.final_message("**Итог**", _cfg(answer_html=False))
    assert msg.content == "**Итог**"
    assert "orchestrator_markdown" not in msg.additional_kwargs


def test_final_message_renders_tables_with_borders():
    """Клиент показывает HTML как есть, без своих стилей: таблица без рамок
    сливается в текст, поэтому бордер ставим прямо в разметке."""
    md = "| a | b |\n| --- | --- |\n| 1 | 2 |\n\nтекст\n\n| c |\n| --- |\n| 3 |"
    msg = messages.final_message(md, _cfg())
    assert msg.content.count('<table border="1">') == 2
    assert "<table>" not in msg.content


# --- история для LLM ------------------------------------------------------

def test_history_drops_steps_and_uses_markdown_source():
    history = [
        HumanMessage(content="вопрос"),
        AIMessage(content="прогресс", additional_kwargs={"orchestrator_step": True}),
        AIMessage(
            content="<p>ответ</p>",
            additional_kwargs={"orchestrator_final": True, "orchestrator_markdown": "ответ"},
        ),
    ]
    out = messages.history_for_llm(history)
    assert [m.content for m in out] == ["вопрос", "ответ"]


def test_history_strips_trace_section():
    text = "Ответ по существу.\n\n---\n### Как я пришёл к выводу\n\nшаги…"
    out = messages.history_for_llm([AIMessage(content=text)])
    assert out[0].content == "Ответ по существу."


def test_last_user_text():
    history = [HumanMessage(content="первый"), AIMessage(content="ответ"),
               HumanMessage(content="второй")]
    assert messages.last_user_text({"messages": history}) == "второй"
    assert messages.last_user_text({"messages": []}) == ""


# --- describe_answer ------------------------------------------------------

def _state(**kw):
    base = {
        "messages": [], "reasoning_trace": [], "metrics": None,
        "position": "Аналитик", "memory_context": None,
    }
    base.update(kw)
    return base


def _dataset():
    star = make_metric("Звезда качества", date="2026-04-13", fact=None, star_received=1,
                       children=[make_metric("CSAT", date="2026-04-13", fact=4.6, plan=4.5,
                                             is_star_metric=True)])
    return make_dataset_obj(
        [
            make_metric("Продажи", date="2026-04-06", fact=10.0, plan=9.0),
            make_metric("Продажи", date="2026-04-13", fact=8.0, plan=9.0,
                        children=[make_metric("Звонки", date="2026-04-13", fact=30.0)]),
            make_metric("Продажи", date="2026-04-13", fact=3.0, element="Канал А"),
            star,
        ],
        me={"tabnum": 1, "fio": "Руководитель Р.Р.", "metrics": []},
    )


_AGGREGATES = [{"aggregate_id": None, "dataset": {
    "level": lv, "level_name": name, "metrics": [{"metric_name": "Продажи", "aggregates": {
        "dt": "2026-04-13", "median": 8.5, "history": [{"dt": "2026-04-06", "median": 8.0}]}}]}}
    for lv, name in (("gosb", "по ГОСБ"), ("tb", "по ТБ"))]

_TRACE = [
    {"stage": "load", "kind": "decision", "summary": "Собрал базу", "detail": {}},
    {"stage": "agent", "kind": "tool_call", "summary": "…", "detail": {
        "tool": "metric_card", "args": {"metric": "Продажи", "person": None, "depth": 2},
        "result_summary": "ПОКАЗАТЕЛЬ: Продажи\nОписание: …"}},
    {"stage": "agent", "kind": "tool_call", "summary": "…", "detail": {
        "tool": "query_sql",
        "args": {"sql": "SELECT metric, fact\nFROM v_fact_latest\nLIMIT 5",
                 "purpose": "последние значения"},
        "result_summary": "| metric | fact |\n| --- | --- |\n| Продажи | 8 |"}},
    {"stage": "agent", "kind": "error", "summary": "Модель не ответила", "detail": {}},
]


def test_describe_disabled_returns_text_unchanged():
    assert describe.with_description("ответ", _state(), _cfg()) == "ответ"


def test_describe_numbered_steps_at_the_end():
    state = _state(metrics=_dataset(), aggregates=_AGGREGATES, reasoning_trace=_TRACE)
    out = describe.with_description("ответ", state, _cfg(describe_answer=True))
    head, section = out.split("\n\n---\n", 1)
    assert head == "ответ"
    lines = section.splitlines()
    assert lines[0] == "### Как я пришёл к выводу"
    steps = [ln for ln in lines if ln[:1].isdigit()]
    assert steps == [
        "1. Структура: руководитель — Руководитель Р.Р.; сотрудников — 1: "
        "Иванов Иван Иванович; позиция — Аналитик.",
        "2. Данные: показателей — 3; верхнего уровня — 1: Продажи; разрезов — 1; "
        "значений — 6; периоды: неделя — 2 (2026-04-06 … 2026-04-13); "
        "звёзд — 1: Звезда качества (хотя бы раз получены — 1).",
        "3. Агрегаты групп сравнения: уровней — 2: по ГОСБ, по ТБ; показателей — 1; "
        "срезов — 4 (текущих — 2, история — 2).",
        "4. Долгосрочная память: сохранённого контекста прошлых диалогов нет.",
        "5. Инструмент `metric_card` (metric=Продажи, depth=2) → ПОКАЗАТЕЛЬ: Продажи",
        "6. Инструмент `query_sql` (цель — последние значения; SQL: "
        "`SELECT metric, fact FROM v_fact_latest LIMIT 5`) → | metric | fact |",
        "7. Сбой: Модель не ответила",
    ]


def test_describe_without_data_and_aggregates():
    steps = describe.describe_steps(
        _state(metrics_error="сервис недоступен", aggregates_error="таймаут",
               memory_context="Прошлый раз обсуждали AHT.")
    )
    assert steps == [
        "Структура: данные не загружены (сервис недоступен).",
        "Данные: показатели не загружены.",
        "Агрегаты групп сравнения: не загружены (таймаут).",
        "Долгосрочная память: Прошлый раз обсуждали AHT.",
    ]
    assert describe.describe_steps(_state())[2] == "Агрегаты групп сравнения: не пришли."


def test_describe_accepts_json_strings_and_caps_names():
    import json

    many = make_dataset_obj([make_metric(f"М{i}", fact=1.0) for i in range(15)])
    steps = describe.describe_steps(_state(metrics=json.dumps(many, ensure_ascii=False)))
    assert "верхнего уровня — 15: М0, М1" in steps[1]
    assert steps[1].count(", М") == 9 and "и ещё 5" in steps[1]
