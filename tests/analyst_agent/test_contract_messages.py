"""Контракт сообщений и describe_answer — он должен совпадать с v4.

Прод-клиент уже завязан на теги orchestrator_step / orchestrator_final,
answer_html и раздел «Как я пришёл к выводу», поэтому поведение переносится
как есть.
"""
from __future__ import annotations

import asyncio

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


def test_describe_disabled_returns_text_unchanged():
    out = asyncio.run(describe.with_description("ответ", _state(), _cfg()))
    assert out == "ответ"


def test_describe_appends_section_with_sources():
    state = _state(
        metrics=make_dataset_obj([make_metric("Продажи", fact=10.0, plan=9.0)]),
        reasoning_trace=[{"stage": "agent", "kind": "tool_call",
                          "summary": "query_sql(…) → 3 строки", "detail": {}}],
    )
    out = asyncio.run(describe.with_description("ответ", state, _cfg(describe_answer=True)))
    assert "### Как я пришёл к выводу" in out
    assert "**Исходные данные:**" in out
    assert "Метрики:" in out
    assert "query_sql" in out


def test_describe_uses_llm_narrative_when_available():
    class FakeLLM:
        def invoke(self, _messages):
            return AIMessage(content="Посмотрел продажи и сверил с планом.")

    state = _state(reasoning_trace=[{"stage": "agent", "kind": "tool_call",
                                     "summary": "query_sql(…)", "detail": {}}])
    out = asyncio.run(
        describe.with_description("ответ", state, _cfg(describe_answer=True), llm=FakeLLM())
    )
    assert "Посмотрел продажи и сверил с планом." in out


def test_describe_falls_back_when_llm_fails():
    class BrokenLLM:
        def invoke(self, _messages):
            raise RuntimeError("нет сети")

    state = _state(reasoning_trace=[{"stage": "agent", "kind": "tool_call",
                                     "summary": "query_sql(…) → 3 строки", "detail": {}}])
    out = asyncio.run(
        describe.with_description("ответ", state, _cfg(describe_answer=True), llm=BrokenLLM())
    )
    assert "query_sql" in out  # детерминированный список шагов


def test_sources_block_reports_missing_metrics():
    text = describe.describe_sources_block(_state(metrics_error="сервис недоступен"))
    assert "не загружены" in text


def test_sources_block_lists_wiki_hits():
    state = _state(easyrag_snippets=[
        {"page_title": "Методика", "section_title": "AHT", "similarity": 0.9}
    ])
    assert "«Методика / AHT»" in describe.describe_sources_block(state)


def test_render_trace_deterministic():
    trace = [
        {"stage": "agent", "kind": "tool_call", "summary": "первый шаг", "detail": {}},
        {"stage": "agent", "kind": "decision", "summary": "вывод", "detail": {}},
    ]
    text = describe.render_trace(trace)
    assert "1. **Метрики.** первый шаг" in text
    assert "2. **Вывод.** вывод" in text
