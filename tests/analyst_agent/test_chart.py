"""Режим интерактивного графика (``interactive_chart``).

При включённом флаге модель описывает график инструментом ``build_chart``
(типизированное подмножество plotly), код собирает из него фигуру plotly, а
итог хода несёт её в ``additional_kwargs["orchestrator_chart"]`` — списке из
одного элемента ``{type: "plotly", data: фигура, tool_call_id, title}``. Не вызвала
инструмент — графика нет. При выключенном флаге не меняется ничего: ни набор
инструментов, ни промпт, ни итоговое сообщение.
"""
from __future__ import annotations

import asyncio
import json

from _fixtures import make_dataset_obj, make_metric  # noqa: F401
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph_executor.aegra_agents.analyst_agent import graph as graph_module, prompts
from langgraph_executor.aegra_agents.analyst_agent.agent.runctx import RunContext
from langgraph_executor.aegra_agents.analyst_agent.agent.trace import default_step_text
from langgraph_executor.aegra_agents.analyst_agent.config import RunConfig
from langgraph_executor.aegra_agents.analyst_agent.contract import messages
from langgraph_executor.aegra_agents.analyst_agent.db import core
from langgraph_executor.aegra_agents.analyst_agent.deviations import builder, ledger
from langgraph_executor.aegra_agents.analyst_agent.prompts.chart import (
    CHART_PROMPT_BLOCK,
    MAX_POINTS,
    MAX_TRACES,
)
from langgraph_executor.aegra_agents.analyst_agent.tools import build_tools
from langgraph_executor.aegra_agents.analyst_agent.tools.chart import TOOL_NAME, build_figure

PERSON = "100500"
DATASET = make_dataset_obj([
    make_metric("Продажи", date="2026-04-06", fact=90.0, plan=100.0),
    make_metric("Продажи", date="2026-04-13", fact=70.0, plan=100.0),
])

LINE_ARGS = {
    "chart_type": "line",
    "title": "Продажи: факт и план",
    "x_title": "Неделя",
    "y_title": "Продажи",
    "traces": [
        {"name": "Факт", "x": ["2026-04-06", "2026-04-13"], "y": [90.0, 70.0]},
        {"name": "План", "x": ["2026-04-06", "2026-04-13"], "y": [100.0, 100.0]},
    ],
}


def _ctx(**kw):
    db = core.build_run_db(DATASET)
    devs = builder.build_deviations(db, focus_person_key=PERSON)
    return RunContext(
        db=db, person_key=PERSON, direction_key="test-dir",
        ledger=ledger.DeviationLedger(db, devs, turn=1), **kw,
    )


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


# --- флаг -----------------------------------------------------------------

def test_flag_off_by_default_and_parsed_from_configurable():
    assert RunConfig.from_config({"configurable": {}}).interactive_chart is False
    on = RunConfig.from_config({"configurable": {"interactive_chart": "true"}})
    assert on.interactive_chart is True


# --- сборка фигуры ---------------------------------------------------------

def test_line_figure_is_plotly_json():
    fig = build_figure(LINE_ARGS)
    assert fig == {
        "data": [
            {"type": "scatter", "mode": "lines+markers", "name": "Факт",
             "x": ["2026-04-06", "2026-04-13"], "y": [90.0, 70.0]},
            {"type": "scatter", "mode": "lines+markers", "name": "План",
             "x": ["2026-04-06", "2026-04-13"], "y": [100.0, 100.0]},
        ],
        "layout": {
            "title": {"text": "Продажи: факт и план"},
            "xaxis": {"title": {"text": "Неделя"}},
            "yaxis": {"title": {"text": "Продажи"}},
        },
    }
    json.dumps(fig)  # фигура сериализуется как есть


def test_bar_and_scatter_types_and_per_trace_override():
    fig = build_figure({
        "chart_type": "bar",
        "title": "Сравнение",
        "traces": [
            {"name": "Факт", "x": ["Я", "Коллеги"], "y": [70, 85]},
            {"name": "План", "type": "line", "x": ["Я", "Коллеги"], "y": [100, 100]},
            {"name": "Точки", "type": "scatter", "x": ["Я"], "y": [1]},
        ],
    })
    bar, line, points = fig["data"]
    assert bar == {"type": "bar", "name": "Факт", "x": ["Я", "Коллеги"], "y": [70.0, 85.0]}
    assert (line["type"], line["mode"]) == ("scatter", "lines+markers")
    assert (points["type"], points["mode"]) == ("scatter", "markers")
    # Пустые подписи осей в layout не попадают.
    assert fig["layout"] == {"title": {"text": "Сравнение"}}


def test_values_are_coerced_and_gaps_kept():
    fig = build_figure({
        "chart_type": "line",
        "title": "Т",
        "traces": [{"name": "Ф", "x": [1, "2026-04-13", "c"], "y": ["90,5", None, "нет"]}],
    })
    trace = fig["data"][0]
    assert trace["x"] == ["1", "2026-04-13", "c"]
    assert trace["y"] == [90.5, None, None]


def test_broken_traces_are_dropped():
    fig = build_figure({
        "chart_type": "line",
        "title": "Т",
        "traces": [
            {"name": "Разная длина", "x": ["a", "b"], "y": [1]},
            {"name": "Пусто", "x": [], "y": []},
            {"name": "Без чисел", "x": ["a"], "y": [None]},
            {"name": "Ок", "x": ["a"], "y": [1]},
            "не объект",
        ],
    })
    assert [t["name"] for t in fig["data"]] == ["Ок"]


def test_nothing_valid_gives_none():
    assert build_figure({"chart_type": "line", "title": "Т", "traces": []}) is None
    assert build_figure({"chart_type": "line", "title": "Т",
                         "traces": [{"name": "x", "x": ["a"], "y": []}]}) is None


def test_caps_traces_and_points_and_names_unnamed():
    many = [{"name": "", "x": ["a"], "y": [i]} for i in range(MAX_TRACES + 3)]
    fig = build_figure({"chart_type": "bar", "title": "Т", "traces": many})
    assert len(fig["data"]) == MAX_TRACES
    assert fig["data"][0]["name"] == "Серия 1"

    long = {"name": "Длинный", "x": [str(i) for i in range(MAX_POINTS + 10)],
            "y": list(range(MAX_POINTS + 10))}
    fig = build_figure({"chart_type": "line", "title": "Т", "traces": [long]})
    assert len(fig["data"][0]["x"]) == len(fig["data"][0]["y"]) == MAX_POINTS


# --- инструмент -----------------------------------------------------------

def test_tool_bound_only_with_flag():
    assert TOOL_NAME not in {t.name for t in build_tools(_ctx())}
    assert TOOL_NAME in {t.name for t in build_tools(_ctx(interactive_chart=True))}


def test_tool_stores_figure_and_overwrites_previous_call():
    ctx = _ctx(interactive_chart=True)
    tool = _tool(build_tools(ctx), TOOL_NAME)
    reply = tool.invoke({**LINE_ARGS, "title": "Старый"})
    assert ctx.chart["layout"]["title"]["text"] == "Старый"
    assert "итоговый ответ" in reply.lower()
    tool.invoke(LINE_ARGS)
    assert ctx.chart == build_figure(LINE_ARGS)


def test_tool_with_broken_args_returns_hint_and_keeps_previous():
    ctx = _ctx(interactive_chart=True)
    tool = _tool(build_tools(ctx), TOOL_NAME)
    reply = tool.invoke({"chart_type": "line", "title": "Т", "traces": []})
    assert ctx.chart is None
    assert "traces" in reply
    tool.invoke(LINE_ARGS)
    tool.invoke({"chart_type": "line", "title": "Т", "traces": []})
    assert ctx.chart == build_figure(LINE_ARGS)


def test_tool_has_typed_schema_for_gigachat():
    """Ultra шлёт `{}` вместо значений без явных типов — схема типизирована до листьев."""
    tool = _tool(build_tools(_ctx(interactive_chart=True)), TOOL_NAME)
    schema = tool.args_schema.model_json_schema()
    props = schema["properties"]
    assert set(props["chart_type"]["enum"]) == {"line", "bar", "scatter"}
    assert props["title"]["type"] == "string"
    assert props["traces"]["type"] == "array"
    item = props["traces"]["items"]
    if "$ref" in item:
        item = schema["$defs"][item["$ref"].rsplit("/", 1)[-1]]
    assert set(item["required"]) == {"name", "x", "y"}
    assert item["properties"]["x"]["type"] == "array"
    assert item["properties"]["y"]["type"] == "array"


def test_step_text_for_tool():
    assert "график" in default_step_text(TOOL_NAME, {}).lower()


# --- контракт итога --------------------------------------------------------

def test_final_message_carries_chart_only_when_given():
    cfg = {"configurable": {"answer_html": False}}
    plain = messages.final_message("Ответ", cfg)
    assert messages.CHART_KEY not in plain.additional_kwargs
    none = messages.final_message("Ответ", cfg, chart=None)
    assert messages.CHART_KEY not in none.additional_kwargs
    fig = build_figure(LINE_ARGS)
    with_chart = messages.final_message("Ответ", cfg, chart=fig)
    assert with_chart.additional_kwargs[messages.CHART_KEY] == [{
        "type": "plotly",
        "data": fig,
        "tool_call_id": "interactive_chart",
        "title": "Продажи: факт и план",
    }]
    assert with_chart.additional_kwargs[messages.FINAL_KEY] is True
    assert messages.CHART_KEY == "orchestrator_chart"


def test_chart_payload_title_falls_back_when_figure_has_none():
    fig = build_figure({**LINE_ARGS, "title": "  "})
    assert "title" not in fig["layout"]
    (entry,) = messages.chart_payload(fig)
    assert entry["title"] == messages.CHART_DEFAULT_TITLE
    assert entry["data"] is fig


def test_history_for_llm_ignores_chart():
    msg = messages.final_message("Ответ", {"configurable": {"answer_html": True}},
                                 chart=build_figure(LINE_ARGS))
    hist = messages.history_for_llm([HumanMessage(content="в"), msg])
    assert hist[-1].content == "Ответ"
    assert not hist[-1].additional_kwargs


# --- промпт ----------------------------------------------------------------

def _pctx(**kw):
    base = dict(enrichment_block="СОСТАВ ДАННЫХ", deviations_block="КАРТА",
                turn_kind="initial")
    base.update(kw)
    return prompts.PromptContext(**base)


def test_prompt_block_only_with_flag():
    off = prompts.compose_system_prompt(_pctx())
    on = prompts.compose_system_prompt(_pctx(interactive_chart=True))
    assert CHART_PROMPT_BLOCK not in off
    assert TOOL_NAME not in off
    assert CHART_PROMPT_BLOCK in on
    assert on.index("КАК РАБОТАТЬ С ДАННЫМИ") < on.index(CHART_PROMPT_BLOCK)


def test_prompt_block_follows_suggestions_block():
    from langgraph_executor.aegra_agents.analyst_agent.prompts.suggestions import (
        SUGGESTIONS_PROMPT_BLOCK,
    )

    on = prompts.compose_system_prompt(
        _pctx(interactive_chart=True, interactive_suggestions=True)
    )
    assert on.index(SUGGESTIONS_PROMPT_BLOCK) < on.index(CHART_PROMPT_BLOCK)


def test_prompt_block_names_tool_and_default_chart():
    assert TOOL_NAME in CHART_PROMPT_BLOCK
    assert "проблемн" in CHART_PROMPT_BLOCK
    assert "карт" in CHART_PROMPT_BLOCK.lower()


def test_prompt_block_survives_override():
    on = prompts.compose_system_prompt(
        _pctx(interactive_chart=True, system_prompt_override="Свой промпт")
    )
    assert CHART_PROMPT_BLOCK in on


# --- сквозной ход ----------------------------------------------------------

class FakeLLM:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0
        self.prompts = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, msgs):
        self.calls += 1
        self.prompts.append(msgs)
        item = self.answers.pop(0)
        if isinstance(item, tuple):
            name, args = item
            return AIMessage(content="", tool_calls=[{"name": name, "args": args,
                                                       "id": f"c{self.calls}"}])
        return AIMessage(content=item)


def _patch_load(monkeypatch):
    from langgraph_executor.aegra_agents.analyst_agent.nodes import load as load_mod

    def fake_node():
        async def load_data(state, config):
            return {
                "boss_tabnum": "1", "employee_tabnum": PERSON, "direction_key": "test-dir",
                "metrics": DATASET, "aggregates": None, "metrics_error": None,
                "briefing": messages.last_user_text(state) or None,
                "reasoning_trace": [], "turn_no": 1, "loaded": True,
            }
        return load_data

    monkeypatch.setattr(load_mod, "make_load_data_node", fake_node)
    monkeypatch.setattr(
        "langgraph_executor.aegra_agents.analyst_agent.graph.make_load_data_node", fake_node
    )


def _run(monkeypatch, llm, **cfg):
    _patch_load(monkeypatch)
    app = graph_module.build_graph(llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t", "boss_tabnum": "1",
                               "employee_tabnum": PERSON, "easyrag_enabled": False,
                               "answer_html": False, **cfg}}
    return asyncio.run(app.ainvoke({"messages": [HumanMessage(content="Разбери")]}, config))


def test_turn_with_flag_puts_chart_into_final_message(monkeypatch):
    llm = FakeLLM([(TOOL_NAME, LINE_ARGS), "Продажи просели до 70 при плане 100."])
    out = _run(monkeypatch, llm, interactive_chart=True)
    final = out["messages"][-1]
    assert final.additional_kwargs[messages.FINAL_KEY] is True
    assert final.additional_kwargs[messages.CHART_KEY] == messages.chart_payload(
        build_figure(LINE_ARGS)
    )
    assert CHART_PROMPT_BLOCK in llm.prompts[0][0].content


def test_turn_with_flag_but_no_call_has_no_chart(monkeypatch):
    llm = FakeLLM(["Продажи просели до 70 при плане 100."])
    out = _run(monkeypatch, llm, interactive_chart=True)
    assert messages.CHART_KEY not in out["messages"][-1].additional_kwargs


def test_turn_without_flag_is_unchanged(monkeypatch):
    llm = FakeLLM(["Продажи просели до 70 при плане 100."])
    out = _run(monkeypatch, llm)
    final = out["messages"][-1]
    assert messages.CHART_KEY not in final.additional_kwargs
    assert CHART_PROMPT_BLOCK not in llm.prompts[0][0].content
    assert TOOL_NAME not in {t.name for t in llm.tools}
