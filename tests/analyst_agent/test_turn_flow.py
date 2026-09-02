"""Сквозные прогоны графа на MemorySaver с фейковой моделью.

Проверяем контракт хода целиком: что видит клиент (сообщения, теги, порядок),
что переживает ход (карта отклонений), сколько раз зовётся модель.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from _fixtures import make_dataset_obj, make_metric  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from langgraph_executor.aegra_agents.analyst_agent import graph as graph_module  # noqa: E402
from langgraph_executor.aegra_agents.analyst_agent.contract.messages import (  # noqa: E402
    FINAL_KEY,
    STEP_KEY,
)

DATASET = make_dataset_obj([
    make_metric("Продажи", date="2026-04-06", fact=90.0, plan=100.0),
    make_metric("Продажи", date="2026-04-13", fact=70.0, plan=100.0,
                children=[make_metric("Звонки", date="2026-04-13", fact=30.0,
                                      plan=50.0, influent_percent=60)]),
])


class FakeLLM:
    """Модель, отвечающая заранее заданными репликами (на каждый вызов — своя)."""

    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.calls = 0
        self.prompts = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls += 1
        self.prompts.append(messages)
        if self.answers:
            item = self.answers.pop(0)
            if isinstance(item, tuple):
                name, args = item
                return AIMessage(
                    content="",
                    tool_calls=[{"name": name, "args": args, "id": f"c{self.calls}"}],
                )
            return AIMessage(content=item)
        return AIMessage(content="Ответ по показателям сотрудника.")


def _build(llm, monkeypatch=None):
    return graph_module.build_graph(llm, checkpointer=MemorySaver())


def _cfg(thread="t1", **kw):
    base = {
        "thread_id": thread,
        "boss_tabnum": "1",
        "employee_tabnum": "100500",
        "easyrag_enabled": False,
    }
    base.update(kw)
    return {"configurable": base}


def _patch_load(monkeypatch, dataset=DATASET, aggregates=None):
    """Сервис данных подменяем: тест про граф, а не про внешний HTTP."""
    from langgraph_executor.aegra_agents.analyst_agent.nodes import load as load_mod

    def fake_node():
        async def load_data(state, config):
            from langgraph_executor.aegra_agents.analyst_agent.contract.messages import (
                last_user_text,
            )

            cfg = (config or {}).get("configurable") or {}
            return {
                "boss_tabnum": "1",
                "employee_tabnum": "100500",
                "direction_key": "test-dir",
                "metrics": dataset,
                "aggregates": aggregates,
                "metrics_error": None,
                "briefing": (last_user_text(state) or "").strip() or None,
                "source_type": cfg.get("source_type"),
                "source_id": cfg.get("source_id"),
                "dashboard_mode": bool(cfg.get("dashboard_mode")),
                "reasoning_trace": [],
                "turn_no": 1,
                "loaded": True,
            }

        return load_data

    monkeypatch.setattr(load_mod, "make_load_data_node", fake_node)
    monkeypatch.setattr(
        "langgraph_executor.aegra_agents.analyst_agent.graph.make_load_data_node",
        fake_node,
    )


def test_first_turn_returns_single_final_message(monkeypatch):
    _patch_load(monkeypatch)
    llm = FakeLLM(["Продажи просели до 70 при плане 100."])
    app = _build(llm)
    out = asyncio.run(
        app.ainvoke({"messages": [HumanMessage(content="Разбери показатели")]}, _cfg())
    )
    finals = [m for m in out["messages"] if m.additional_kwargs.get(FINAL_KEY)]
    assert len(finals) == 1
    assert out["messages"][-1] is finals[0]
    assert "Продажи" in finals[0].additional_kwargs["orchestrator_markdown"]
    assert "<p>" in finals[0].content            # answer_html по умолчанию
    assert llm.calls == 1                        # один цикл, без роутера и пересказа


def test_first_turn_prompt_carries_data_blocks(monkeypatch):
    _patch_load(monkeypatch)
    llm = FakeLLM(["Ответ."])
    app = _build(llm)
    asyncio.run(app.ainvoke({"messages": [HumanMessage(content="Разбери")]}, _cfg()))
    system_prompt = llm.prompts[0][0].content
    assert "СХЕМА ДАННЫХ" in system_prompt
    assert "КАТАЛОГ ПОКАЗАТЕЛЕЙ" in system_prompt
    assert "КАРТА ОТКЛОНЕНИЙ" in system_prompt
    assert "Продажи" in system_prompt


def test_tool_call_produces_step_message_before_final(monkeypatch):
    _patch_load(monkeypatch)
    llm = FakeLLM([("metric_card", {"metric": "Продажи"}), "Разобрал продажи."])
    app = _build(llm)
    out = asyncio.run(app.ainvoke({"messages": [HumanMessage(content="Разбери")]}, _cfg()))
    steps = [m for m in out["messages"] if m.additional_kwargs.get(STEP_KEY)]
    # Шаги идут в порядке узлов: сперва загрузка данных, затем вызов инструмента.
    assert "Загрузил показатели" in steps[0].content
    assert any("Продажи" in m.content for m in steps[1:])
    assert out["messages"][-1].additional_kwargs.get(FINAL_KEY)


def test_progress_can_be_disabled(monkeypatch):
    _patch_load(monkeypatch)
    llm = FakeLLM([("metric_card", {"metric": "Продажи"}), "Готово."])
    app = _build(llm)
    out = asyncio.run(
        app.ainvoke(
            {"messages": [HumanMessage(content="Разбери")]},
            _cfg(emit_progress_messages=False),
        )
    )
    assert not [m for m in out["messages"] if m.additional_kwargs.get(STEP_KEY)]


def test_second_turn_reuses_state_and_deviations(monkeypatch):
    _patch_load(monkeypatch)
    llm = FakeLLM(["Первый ответ.", "Второй ответ."])
    app = _build(llm)
    cfg = _cfg("t-two")
    first = asyncio.run(
        app.ainvoke({"messages": [HumanMessage(content="Разбери")]}, cfg)
    )
    # deviations — внутренний канал: наружу не отдаётся, но чекпойнтится.
    assert "deviations" not in first
    assert app.get_state(cfg).values["deviations"]
    second = asyncio.run(
        app.ainvoke({"messages": [HumanMessage(content="А что со звонками?")]}, cfg)
    )
    assert second["messages"][-1].additional_kwargs["orchestrator_markdown"] == "Второй ответ."
    # Карта отклонений пережила ход, данные не перезагружались.
    assert app.get_state(cfg).values["deviations"]
    assert second["metrics_summary"] == "Первый ответ."


def test_metrics_error_answers_gracefully(monkeypatch):
    from langgraph_executor.aegra_agents.analyst_agent.nodes import load as load_mod

    def broken():
        async def load_data(state, config):
            return {
                "loaded": True, "metrics": None, "turn_no": 1,
                "metrics_error": "сервис недоступен", "reasoning_trace": [],
            }
        return load_data

    monkeypatch.setattr(load_mod, "make_load_data_node", broken)
    monkeypatch.setattr(
        "langgraph_executor.aegra_agents.analyst_agent.graph.make_load_data_node", broken
    )
    llm = FakeLLM()
    app = _build(llm)
    out = asyncio.run(app.ainvoke({"messages": [HumanMessage(content="Разбери")]}, _cfg()))
    assert out["messages"][-1].additional_kwargs.get(FINAL_KEY)
    assert llm.calls == 0     # модель не звалась вовсе


def test_describe_answer_section_appended(monkeypatch):
    _patch_load(monkeypatch)
    llm = FakeLLM([("metric_card", {"metric": "Продажи"}), "Итог.", "Посмотрел продажи."])
    app = _build(llm)
    out = asyncio.run(
        app.ainvoke(
            {"messages": [HumanMessage(content="Разбери")]},
            _cfg(describe_answer=True, answer_html=False),
        )
    )
    text = out["messages"][-1].content
    assert "### Как я пришёл к выводу" in text
    assert "**Исходные данные:**" in text


def test_insight_written_once_with_source(monkeypatch):
    _patch_load(monkeypatch)
    sent = []

    class FakeComponent:
        def __init__(self, **kw):
            self.kw = kw

        def submit(self):
            sent.append(self.kw)
            return {"ok": True}

    monkeypatch.setattr(
        "langgraph_executor.aegra_agents.analyst_agent.nodes.insight."
        "SendAssignmentsComponent",
        FakeComponent,
    )
    llm = FakeLLM(["Ответ."])
    app = _build(llm)
    out = asyncio.run(
        app.ainvoke(
            {"messages": [HumanMessage(content="Разбери")]},
            _cfg(source_type="task", source_id="42"),
        )
    )
    assert len(sent) == 1
    insight = sent[0]["insight"]
    assert insight["type"] in ("main_problem", "achievement", "norm")
    assert insight["metric_name"]
    assert out["reasoning_trace"][-1]["stage"] == "assignments"


def test_no_insight_without_source(monkeypatch):
    _patch_load(monkeypatch)
    sent = []

    class FakeComponent:
        def __init__(self, **kw):
            pass

        def submit(self):
            sent.append(1)

    monkeypatch.setattr(
        "langgraph_executor.aegra_agents.analyst_agent.nodes.insight."
        "SendAssignmentsComponent",
        FakeComponent,
    )
    app = _build(FakeLLM(["Ответ."]))
    asyncio.run(app.ainvoke({"messages": [HumanMessage(content="Разбери")]}, _cfg()))
    assert sent == []


def test_dashboard_runs_tasks_sequentially(monkeypatch):
    _patch_load(monkeypatch)
    planner_json = (
        '{"tasks": [{"id": "t1", "title": "Продажи", "question": "Что с продажами?",'
        ' "depends_on": [], "metric_hints": ["Продажи"]},'
        ' {"id": "t2", "title": "Звонки", "question": "Что со звонками?",'
        ' "depends_on": ["t1"], "metric_hints": ["Звонки"]}]}'
    )
    llm = FakeLLM([
        planner_json,          # планировщик
        "Продажи просели.",    # задача 1
        "Звонков мало.",       # задача 2
        "Сводный ответ.",      # сводка
    ])
    app = _build(llm)
    out = asyncio.run(
        app.ainvoke(
            {"messages": [HumanMessage(content="1. Продажи\n2. Звонки")]},
            _cfg("t-dash", dashboard_mode=True),
        )
    )
    assert [t["title"] for t in out["tasks"]] == ["Продажи", "Звонки"]
    assert [r["answer_md"] for r in out["task_results"]] == [
        "Продажи просели.", "Звонков мало."
    ]
    finals = [m for m in out["messages"] if m.additional_kwargs.get(FINAL_KEY)]
    assert len(finals) == 1
    assert finals[0].additional_kwargs["orchestrator_markdown"] == "Сводный ответ."
    assert out["messages"][-1] is finals[0]
    assert llm.calls == 4


def test_dashboard_second_task_sees_first_result(monkeypatch):
    _patch_load(monkeypatch)
    planner_json = (
        '{"tasks": [{"id": "t1", "title": "Продажи", "question": "Что с продажами?"},'
        ' {"id": "t2", "title": "Звонки", "question": "Что со звонками?",'
        ' "depends_on": ["t1"]}]}'
    )
    llm = FakeLLM([planner_json, "Продажи просели на 30 %.", "Звонки.", "Сводка."])
    app = _build(llm)
    asyncio.run(
        app.ainvoke(
            {"messages": [HumanMessage(content="разбери")]},
            _cfg("t-dash2", dashboard_mode=True),
        )
    )
    second_task_prompt = llm.prompts[2][0].content
    assert "Продажи просели на 30 %." in second_task_prompt
    assert "ЗАДАЧА 2 из 2" in second_task_prompt


def test_dashboard_falls_back_to_plain_turn(monkeypatch):
    _patch_load(monkeypatch)
    llm = FakeLLM(['{"tasks": []}', "Обычный ответ."])
    app = _build(llm)
    out = asyncio.run(
        app.ainvoke(
            {"messages": [HumanMessage(content="разбери")]},
            _cfg("t-dash3", dashboard_mode=True),
        )
    )
    assert out["tasks"] == []
    assert out["messages"][-1].additional_kwargs["orchestrator_markdown"] == "Обычный ответ."
