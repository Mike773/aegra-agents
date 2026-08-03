"""Формат ответа analytic_orchestrator_v2: HTML вместо markdown.

Вызывающая система рендерит ответ как HTML: markdown показывается «как есть», а
перенос строки без <br> теряется. Промпт этого требует, но полагаться на него
нельзя — итог хода приводится к HTML детерминированно. Здесь проверяем именно
детерминированную часть: конвертацию, лимит эмодзи и то, что узлы отдают HTML.

Импортируем только nodes.py/prompts.py (не graph.py) — GigaChat-креды/сеть не нужны.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langchain_core.messages import AIMessage, HumanMessage

from langgraph_executor.aegra_agents.analytic_orchestrator_v2.nodes import (
    _strip_html,
    _strip_trace_section,
    _to_html,
    _with_description,
    make_initial_analysis_node,
    make_respond_node,
    render_trace,
)
from langgraph_executor.aegra_agents.analytic_orchestrator_v2.prompts import (
    BUSINESS_SYSTEM_PROMPT,
    SAVE_INSIGHT_DONE,
    SAVE_INSIGHT_EMPTY,
    SAVE_INSIGHT_ERROR,
    SAVE_INSIGHT_NO_SOURCE,
)


class _AIMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content):
        self._content = content
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _AIMsg(self._content)


class FakeAnalyzer:
    """Подграф json_analyzer: отдаёт готовый разбор без сети и LLM."""

    async def ainvoke(self, payload):
        return {"answer": "Производительность 12 при плане 18.", "tool_steps": []}


def _dataset():
    return {
        "me": {"fio": "Босс Борис", "metrics": []},
        "employees": [{"fio": "Иванов Иван", "metrics": []}],
    }


# --- Конвертация markdown → HTML --------------------------------------------

def test_markdown_markup_becomes_html():
    out = _to_html(
        "### Резюме\n"
        "**Производительность** просела, *AHT* вырос.\n"
        "- первый пункт\n"
        "* второй пункт\n"
        "---\n"
        "Подробнее: [регламент](https://wiki.example/aht)"
    )
    assert "<b>Резюме</b>" in out
    assert "<b>Производительность</b>" in out
    assert "<i>AHT</i>" in out
    assert "• первый пункт" in out and "• второй пункт" in out
    assert '<a href="https://wiki.example/aht">регламент</a>' in out
    # Ни одного markdown-маркера не осталось.
    assert "**" not in out and "###" not in out and "---" not in out
    assert "[регламент]" not in out


def test_line_breaks_become_br():
    out = _to_html("Первая строка\nВторая строка\n\nНовый абзац")
    assert out == "Первая строка<br>Вторая строка<br><br>Новый абзац"
    # Переносов строк в готовом фрагменте не остаётся.
    assert "\n" not in out


def test_existing_html_survives_and_br_not_doubled():
    out = _to_html("<b>Резюме</b><br>\nAHT вырос<br />\nПлан не выполнен")
    assert out == "<b>Резюме</b><br>AHT вырос<br>План не выполнен"


def test_code_fence_is_dropped():
    out = _to_html("```html\n<b>Резюме</b>\n```")
    assert out == "<b>Резюме</b>"


def test_stray_lt_is_escaped_but_tags_kept():
    out = _to_html("<b>AHT</b> < 300 сек, отклонение <5%")
    assert "<b>AHT</b>" in out
    assert "&lt; 300" in out
    assert "&lt;5%" in out


# --- Эмодзи: не больше двух на ответ ----------------------------------------

def test_emoji_capped_at_two():
    out = _to_html("📉 Спад 🔥 и ещё ✅ плюс 🚀 и 📊 хвост")
    assert out.count("📉") == 1
    assert out.count("🔥") == 1
    for extra in ("✅", "🚀", "📊"):
        assert extra not in out
    # Текст вокруг не рвётся и двойных пробелов не остаётся.
    assert "Спад" in out and "хвост" in out
    assert "  " not in out


def test_zwj_emoji_counts_as_one():
    """Составной эмодзи (ZWJ) — один, иначе он один съедал бы весь лимит."""
    out = _to_html("👩‍💻 работает, 📈 растёт, ❗ третий")
    assert "👩‍💻" in out and "📈" in out
    assert "❗" not in out


def test_two_emoji_pass_through():
    text = "📉 Производительность просела, 📈 качество выросло"
    assert _to_html(text) == text


# --- Раздел трассы ----------------------------------------------------------

def test_trace_section_is_html():
    trace = [
        {"stage": "route", "kind": "intent", "summary": "Определил тип запроса."},
        {"stage": "json_analyzer", "kind": "tool_call",
         "summary": "compare(a<b) → 2 строки"},
    ]
    section = render_trace(trace)
    assert section.startswith("<b>Как я пришёл к выводу</b>")
    assert "<b>Классификация запроса.</b>" in section
    assert "**" not in section and "---" not in section
    assert "\n" not in section
    # «<» из аргументов инструмента не ломает разметку.
    assert "a&lt;b" in section


def test_trace_section_appended_and_stripped_back():
    state = {"reasoning_trace": [
        {"stage": "respond", "kind": "decision", "summary": "Сформировал ответ."},
    ]}
    config = {"configurable": {"describe_answer": True}}
    full = _with_description("Ответ агента.", state, config)
    assert full.startswith("Ответ агента.<br><br><b>Как я пришёл к выводу</b>")
    # В историю для LLM раздел не уезжает — вместе с отбивкой перед ним.
    assert _strip_trace_section(full) == "Ответ агента."


def test_strip_trace_section_handles_legacy_markdown():
    """В уже идущих диалогах история хранит старый markdown-заголовок."""
    legacy = "Ответ агента.\n\n---\n### Как я пришёл к выводу\n\n1. **Вывод.** Ок"
    assert _strip_trace_section(legacy) == "Ответ агента."


# --- Узлы отдают HTML -------------------------------------------------------

def test_respond_node_returns_html():
    llm = FakeLLM("**Резюме**\nAHT вырос на 22.5%\n\n- разберём переводы 🚀🚀🚀")
    out = make_respond_node(llm)(
        {"metrics": _dataset(), "metrics_summary": "AHT 300 сек.",
         "messages": [HumanMessage("что с AHT?")]},
        {"configurable": {}},
    )
    text = out["messages"][-1].content
    assert "<b>Резюме</b>" in text
    assert "<br>" in text and "\n" not in text
    assert "**" not in text
    assert text.count("🚀") <= 2


def test_initial_analysis_node_returns_html():
    llm = FakeLLM("## Резюме\nЕсть отклонения.\n\n**Производительность** 12 из 18.")
    node = make_initial_analysis_node(llm, FakeAnalyzer())
    out = asyncio.run(node(
        {"metrics": _dataset(), "direction_key": "dir-1",
         "briefing": "разбери метрики",
         "messages": [HumanMessage("разбери метрики")]},
        {"configurable": {}},
    ))
    text = out["messages"][-1].content
    assert "<b>Резюме</b>" in text and "<b>Производительность</b>" in text
    assert "\n" not in text and "**" not in text


# --- Статические тексты и промпт --------------------------------------------

def test_static_replies_are_html_without_markdown():
    for text in (SAVE_INSIGHT_DONE, SAVE_INSIGHT_EMPTY, SAVE_INSIGHT_ERROR,
                 SAVE_INSIGHT_NO_SOURCE):
        assert "*" not in text
        assert "\n" not in text
    assert "<b>копировать</b>" in SAVE_INSIGHT_DONE


def test_system_prompt_demands_html():
    assert "ТОЛЬКО HTML" in BUSINESS_SYSTEM_PROMPT
    assert "<br>" in BUSINESS_SYSTEM_PROMPT
    assert "не больше ДВУХ" in BUSINESS_SYSTEM_PROMPT


# --- Обратная конвертация для промптов --------------------------------------

def test_strip_html_gives_plain_text():
    plain = _strip_html("<b>Резюме</b><br>AHT &lt; 300 сек<br><br>Вывод")
    assert plain == "Резюме\nAHT < 300 сек\n\nВывод"


def test_history_answers_reach_classifier_without_tags():
    """Инсайт классифицируется по тексту без разметки — теги не попадут в сервис."""
    from langgraph_executor.aegra_agents.analytic_orchestrator_v2.nodes import (
        _gather_agent_answers,
    )

    text = _gather_agent_answers({"messages": [
        HumanMessage("что происходит?"),
        AIMessage("<b>Резюме</b><br>Производительность 12 при плане 18."),
    ]})
    assert "<b>" not in text
    assert "Резюме\nПроизводительность 12 при плане 18." in text
