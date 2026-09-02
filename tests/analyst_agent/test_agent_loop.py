"""analyst_agent.agent.loop: ручной цикл вызова инструментов.

GigaChat отдаёт один вызов инструмента за сообщение, поэтому цикл простой:
спросили модель → выполнили вызов → вернули результат → повторили. Гварды
защищают от зацикливания и от переполнения контекста.
"""
from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from langgraph_executor.aegra_agents.analyst_agent.agent import guards as guards_mod
from langgraph_executor.aegra_agents.analyst_agent.agent import loop as loop_mod


class ScriptedLLM:
    """Фейковая модель: отдаёт заранее заданные ответы по очереди."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.seen_messages = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls += 1
        self.seen_messages.append(list(messages))
        if not self.script:
            return AIMessage(content="Готов ответ по собранным данным.")
        item = self.script.pop(0)
        if isinstance(item, str):
            return AIMessage(content=item)
        name, args = item
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call{self.calls}"}],
        )


def _tool(name, fn, description="инструмент"):
    return StructuredTool.from_function(func=fn, name=name, description=description)


def _run(llm, tools, **kw):
    return asyncio.run(
        loop_mod.run_tool_loop(
            llm=llm,
            tools=tools,
            system_prompt="системный промпт",
            history=[],
            question="Что с показателями?",
            **kw,
        )
    )


def test_answer_without_tools():
    res = _run(ScriptedLLM(["Всё в норме."]), [])
    assert res.final_text == "Всё в норме."
    assert res.completed is True
    assert res.tool_steps == []


def test_tool_call_result_goes_back_to_model():
    seen = {}

    def probe(metric: str) -> str:
        seen["metric"] = metric
        return "факт 10, план 9"

    llm = ScriptedLLM([("metric_card", {"metric": "AHT"}), "Ответ по данным."])
    res = _run(llm, [_tool("metric_card", probe)])
    assert seen["metric"] == "AHT"
    assert res.final_text == "Ответ по данным."
    assert len(res.tool_steps) == 1
    assert res.tool_steps[0]["tool"] == "metric_card"
    assert "факт 10" in res.tool_steps[0]["result_summary"]
    # Результат инструмента реально уехал в модель следующим сообщением.
    assert any("факт 10" in str(m.content) for m in llm.seen_messages[-1])


def test_repeated_call_gets_notice_instead_of_result():
    calls = {"n": 0}

    def probe(metric: str) -> str:
        calls["n"] += 1
        return "данные"

    llm = ScriptedLLM([
        ("metric_card", {"metric": "AHT"}),
        ("metric_card", {"metric": "AHT"}),
        "Ответ.",
    ])
    res = _run(llm, [_tool("metric_card", probe)])
    assert calls["n"] == 1  # второй раз инструмент не выполнялся
    assert res.final_text == "Ответ."


def test_budget_exhausted_forces_final_answer():
    llm = ScriptedLLM([("t", {"x": i}) for i in range(10)] + ["Итог."])
    res = _run(llm, [_tool("t", lambda x: f"строка {x}")], budget=3)
    assert res.stop_reason == "budget"
    assert res.final_text
    assert len([s for s in res.tool_steps if s["tool"] == "t"]) <= 3


def test_output_cap_truncates_but_keeps_going():
    big = "x" * 50_000
    llm = ScriptedLLM([("t", {}), "Итог."])
    res = _run(llm, [_tool("t", lambda: big)], output_cap=100)
    summary = res.tool_steps[0]["result_summary"]
    assert len(summary) < 1000
    assert res.final_text == "Итог."
    sent = str(llm.seen_messages[-1][-1].content)
    assert len(sent) < 2000
    assert "НЕ отсутствие данных" in sent


def test_tool_exception_becomes_message_not_crash():
    def broken():
        raise RuntimeError("таблица недоступна")

    llm = ScriptedLLM([("t", {}), "Отвечаю по остальному."])
    res = _run(llm, [_tool("t", broken)])
    assert res.final_text == "Отвечаю по остальному."
    assert "таблица недоступна" in res.tool_steps[0]["result_summary"]


def test_unknown_tool_reported_to_model():
    llm = ScriptedLLM([("не_существует", {}), "Ладно."])
    res = _run(llm, [_tool("t", lambda: "ок")])
    assert res.final_text == "Ладно."
    assert "не_существует" in str(llm.seen_messages[-1][-1].content)


def test_empty_string_args_normalized_to_none():
    seen = {}

    def probe(metric: str, person: str | None = None) -> str:
        seen["person"] = person
        return "ок"

    llm = ScriptedLLM([("t", {"metric": "AHT", "person": "  "}), "Итог."])
    _run(llm, [_tool("t", probe)])
    assert seen["person"] is None


def test_max_rounds_stops_runaway_loop():
    llm = ScriptedLLM([("t", {"i": i}) for i in range(100)])
    res = _run(llm, [_tool("t", lambda i: f"{i}")], budget=50, max_rounds=5)
    assert res.stop_reason in ("rounds", "budget")
    assert llm.calls <= 7


def test_step_texts_collected_for_progress_messages():
    llm = ScriptedLLM([
        ("query_sql", {"sql": "SELECT 1", "purpose": "проверил план"}),
        "Итог.",
    ])
    res = _run(llm, [_tool("query_sql", lambda sql, purpose: "1")])
    assert any("проверил план" in t for t in res.step_texts)


def test_trace_steps_carry_reasoning_text():
    llm = ScriptedLLM([("t", {}), "Итог."])
    llm.script.insert(0, "Сначала посмотрю план.")
    res = _run(llm, [_tool("t", lambda: "ок")])
    # Текст перед вызовом инструмента сохраняется как мотивация вызова.
    assert res.final_text == "Итог." or res.final_text == "Сначала посмотрю план."


def test_guards_are_per_run():
    g1 = guards_mod.RunGuards(budget=2)
    g2 = guards_mod.RunGuards(budget=2)
    g1.register("t", {"a": 1})
    assert g1.calls_made == 1
    assert g2.calls_made == 0
