"""Главный вывод для сервиса инсайтов — из ответа агента.

Модель называет тип, показатель и суть; показатель сверяется с каталогом, а
идентификатор, факт и план берутся из данных.
"""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from _fixtures import make_dataset_obj, make_metric  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from langgraph_executor.aegra_agents.analyst_agent.db import core  # noqa: E402
from langgraph_executor.aegra_agents.analyst_agent.nodes import insight as mod  # noqa: E402
from langgraph_executor.aegra_agents.analyst_agent.prompts.insight import (  # noqa: E402
    MAIN_INSIGHT_SCHEMA,
)

PERSON = "100500"


def _db():
    return core.build_run_db(make_dataset_obj([
        make_metric("Продажи", date="2026-04-06", fact=90.0, plan=100.0, id="m-1"),
        make_metric("Продажи", date="2026-04-13", fact=70.0, plan=100.0, id="m-1",
                    children=[make_metric("Доля переводов", date="2026-04-13",
                                          fact=23.0, plan=10.0, id="m-2")]),
    ]))


def _raw(kind="main_problem", metric="Продажи", text="Продажи 70 при плане 100."):
    return {"type": kind, "metric": metric, "text": text}


def test_metric_resolved_and_numbers_taken_from_data():
    ins = mod._build_insight(_db(), PERSON, _raw(metric="продажи", text="Просели до 50."))
    assert ins == {
        "type": "main_problem", "metric_id": "m-1", "metric_name": "Продажи",
        # Последний срез серии — из данных; «50» из текста модели не берём.
        "fact": 70.0, "plan": 100.0, "text": "Просели до 50.",
    }


def test_child_metric_resolved():
    ins = mod._build_insight(_db(), PERSON, _raw(metric="Доля переводов"))
    assert ins["metric_name"] == "Доля переводов"
    assert ins["metric_id"] == "m-2" and ins["fact"] == 23.0


def test_unknown_name_falls_back_to_metric_named_in_text():
    ins = mod._build_insight(
        _db(), PERSON, _raw(metric="Эффективность", text="Доля переводов 23 % при плане 10 %.")
    )
    assert ins["metric_name"] == "Доля переводов"


def test_norm_without_metric_keeps_empty_fields():
    ins = mod._build_insight(_db(), PERSON, _raw(kind="norm", metric="", text="Всё в норме."))
    assert ins["type"] == "norm"
    assert ins["metric_id"] is None and ins["metric_name"] is None
    assert ins["fact"] is None and ins["plan"] is None


class _LLM:
    def __init__(self, reply):
        self.reply = reply

    def bind_tools(self, tools, **kw):
        self.tools, self.kw = tools, kw
        return self

    def invoke(self, messages):
        self.messages = messages
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _call(args):
    return AIMessage(content="", tool_calls=[{"name": "save_main_insight", "args": args, "id": "c1"}])


def test_ask_forces_the_tool_and_passes_the_answer():
    llm = _LLM(_call(_raw()))
    raw = asyncio.run(mod._ask_main_insight(llm, "Ответ агента."))
    assert raw == _raw()
    assert llm.kw == {"tool_choice": "save_main_insight"}
    assert llm.tools[0]["function"]["parameters"] is MAIN_INSIGHT_SCHEMA
    assert llm.messages[-1].content == "Ответ агента."


def test_ask_rejects_unknown_type_empty_text_and_failures():
    assert asyncio.run(mod._ask_main_insight(_LLM(_call(_raw(kind="problem"))), "x")) is None
    assert asyncio.run(mod._ask_main_insight(_LLM(_call(_raw(text=" "))), "x")) is None
    assert asyncio.run(mod._ask_main_insight(_LLM(AIMessage(content="текст")), "x")) is None
    assert asyncio.run(mod._ask_main_insight(_LLM(RuntimeError("сеть")), "x")) is None


def test_schema_has_single_type_per_param():
    """API v2 не принимает anyOf: у каждого параметра ровно один тип."""
    for prop in MAIN_INSIGHT_SCHEMA["properties"].values():
        assert isinstance(prop["type"], str)
    assert MAIN_INSIGHT_SCHEMA["required"] == ["type", "metric", "text"]
