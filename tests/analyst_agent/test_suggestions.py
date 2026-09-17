"""Режим интерактивных подсказок (``interactive_suggestions``).

При включённом флаге модель отдаёт варианты продолжения инструментом
``suggest_followups``, а итог хода несёт их в
``additional_kwargs["orchestrator_suggestions"]``. При выключенном флаге не
меняется ничего: ни набор инструментов, ни промпт, ни итоговое сообщение.
"""
from __future__ import annotations

import asyncio

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
from langgraph_executor.aegra_agents.analyst_agent.prompts.suggestions import (
    MAX_SUGGESTIONS,
    SUGGESTIONS_PROMPT_BLOCK,
)
from langgraph_executor.aegra_agents.analyst_agent.tools import build_tools
from langgraph_executor.aegra_agents.analyst_agent.tools.suggestions import TOOL_NAME

PERSON = "100500"
DATASET = make_dataset_obj([
    make_metric("Продажи", date="2026-04-06", fact=90.0, plan=100.0),
    make_metric("Продажи", date="2026-04-13", fact=70.0, plan=100.0),
])


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
    assert RunConfig.from_config({"configurable": {}}).interactive_suggestions is False
    on = RunConfig.from_config({"configurable": {"interactive_suggestions": "true"}})
    assert on.interactive_suggestions is True


# --- инструмент -----------------------------------------------------------

def test_tool_bound_only_with_flag():
    assert TOOL_NAME not in {t.name for t in build_tools(_ctx())}
    assert TOOL_NAME in {t.name for t in build_tools(_ctx(interactive_suggestions=True))}


def test_tool_stores_cleaned_suggestions_in_context():
    ctx = _ctx(interactive_suggestions=True)
    tool = _tool(build_tools(ctx), TOOL_NAME)
    reply = tool.invoke({"suggestions": [
        {"label": "  Динамика AHT ", "content": " Разбери динамику AHT по неделям. "},
        {"label": "", "content": "пусто — выбросить"},
        {"label": "Коллеги", "content": ""},
        {"label": "Повторные", "content": "Сравни повторные обращения с коллегами."},
    ]})
    assert ctx.suggestions == [
        {"type": "message", "label": "Динамика AHT",
         "content": "Разбери динамику AHT по неделям."},
        {"type": "message", "label": "Повторные",
         "content": "Сравни повторные обращения с коллегами."},
    ]
    assert "2" in reply and "итоговый ответ" in reply.lower()


def test_tool_caps_count_and_overwrites_previous_call():
    ctx = _ctx(interactive_suggestions=True)
    tool = _tool(build_tools(ctx), TOOL_NAME)
    tool.invoke({"suggestions": [{"label": "Старое", "content": "Старый вопрос."}]})
    many = [{"label": f"В{i}", "content": f"Вопрос {i}."} for i in range(MAX_SUGGESTIONS + 3)]
    tool.invoke({"suggestions": many})
    assert len(ctx.suggestions) == MAX_SUGGESTIONS
    assert ctx.suggestions[0]["label"] == "В0"


def test_tool_without_suggestions_returns_hint_and_keeps_context_empty():
    ctx = _ctx(interactive_suggestions=True)
    tool = _tool(build_tools(ctx), TOOL_NAME)
    reply = tool.invoke({"suggestions": []})
    assert ctx.suggestions == []
    assert "label" in reply and "content" in reply


def test_tool_has_typed_schema_for_gigachat():
    """Ultra шлёт `{}` вместо значений без явных типов — схема должна быть массивом объектов."""
    tool = _tool(build_tools(_ctx(interactive_suggestions=True)), TOOL_NAME)
    schema = tool.args_schema.model_json_schema()
    prop = schema["properties"]["suggestions"]
    assert prop.get("type") == "array"
    item = prop["items"]
    if "$ref" in item:
        item = schema["$defs"][item["$ref"].rsplit("/", 1)[-1]]
    assert set(item["required"]) == {"label", "content"}


def test_step_text_for_tool():
    assert "варианты" in default_step_text(TOOL_NAME, {}).lower()


# --- контракт итога --------------------------------------------------------

def test_final_message_carries_suggestions_only_when_given():
    cfg = {"configurable": {"answer_html": False}}
    plain = messages.final_message("Ответ", cfg)
    assert messages.SUGGESTIONS_KEY not in plain.additional_kwargs
    empty = messages.final_message("Ответ", cfg, suggestions=[])
    assert messages.SUGGESTIONS_KEY not in empty.additional_kwargs
    items = [{"type": "message", "label": "AHT", "content": "Разбери AHT."}]
    with_items = messages.final_message("Ответ", cfg, suggestions=items)
    assert with_items.additional_kwargs[messages.SUGGESTIONS_KEY] == items
    assert with_items.additional_kwargs[messages.FINAL_KEY] is True
    assert messages.SUGGESTIONS_KEY == "orchestrator_suggestions"


def test_history_for_llm_ignores_suggestions():
    items = [{"type": "message", "label": "AHT", "content": "Разбери AHT."}]
    msg = messages.final_message("Ответ", {"configurable": {"answer_html": True}},
                                 suggestions=items)
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
    on = prompts.compose_system_prompt(_pctx(interactive_suggestions=True))
    assert SUGGESTIONS_PROMPT_BLOCK not in off
    assert TOOL_NAME not in off
    assert SUGGESTIONS_PROMPT_BLOCK in on
    # Блок — операционный: живёт после руководства по инструментам и до подсказки хода.
    assert on.index("КАК РАБОТАТЬ С ДАННЫМИ") < on.index(SUGGESTIONS_PROMPT_BLOCK)


def test_prompt_block_names_tool_and_forbids_text_duplicate():
    assert TOOL_NAME in SUGGESTIONS_PROMPT_BLOCK
    assert "Варианты для углубления" in SUGGESTIONS_PROMPT_BLOCK
    assert str(MAX_SUGGESTIONS) in SUGGESTIONS_PROMPT_BLOCK


def test_prompt_block_survives_override():
    on = prompts.compose_system_prompt(
        _pctx(interactive_suggestions=True, system_prompt_override="Свой промпт")
    )
    assert SUGGESTIONS_PROMPT_BLOCK in on


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


def test_turn_with_flag_puts_suggestions_into_final_message(monkeypatch):
    llm = FakeLLM([
        (TOOL_NAME, {"suggestions": [{"label": "AHT", "content": "Разбери AHT."}]}),
        "Продажи просели до 70 при плане 100.",
    ])
    out = _run(monkeypatch, llm, interactive_suggestions=True)
    final = out["messages"][-1]
    assert final.additional_kwargs[messages.FINAL_KEY] is True
    assert final.additional_kwargs[messages.SUGGESTIONS_KEY] == [
        {"type": "message", "label": "AHT", "content": "Разбери AHT."}
    ]
    assert SUGGESTIONS_PROMPT_BLOCK in llm.prompts[0][0].content


def test_turn_without_flag_is_unchanged(monkeypatch):
    llm = FakeLLM(["Продажи просели до 70 при плане 100."])
    out = _run(monkeypatch, llm)
    final = out["messages"][-1]
    assert messages.SUGGESTIONS_KEY not in final.additional_kwargs
    assert SUGGESTIONS_PROMPT_BLOCK not in llm.prompts[0][0].content
    assert TOOL_NAME not in {t.name for t in llm.tools}


def test_tool_keeps_long_labels_up_to_limit():
    """Подписи у клиента бывают длинными — до 1000 символов режется только то,
    что длиннее."""
    from langgraph_executor.aegra_agents.analyst_agent.prompts.suggestions import MAX_LABEL_LEN

    assert MAX_LABEL_LEN == 1000
    ctx = _ctx(interactive_suggestions=True)
    tool = _tool(build_tools(ctx), TOOL_NAME)
    long_label = "Производительность: вклад AHT и Adherence " * 5   # ~210 символов
    too_long = "х" * (MAX_LABEL_LEN + 50)
    tool.invoke({"suggestions": [
        {"label": long_label, "content": "Разложи вклад AHT."},
        {"label": too_long, "content": "Вопрос."},
    ]})
    assert ctx.suggestions[0]["label"] == long_label.strip()
    assert len(ctx.suggestions[1]["label"]) == MAX_LABEL_LEN
    assert ctx.suggestions[1]["label"].endswith("…")
