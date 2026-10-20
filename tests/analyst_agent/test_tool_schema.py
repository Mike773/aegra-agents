"""Схемы инструментов для GigaChat API v2: у каждого параметра ровно один тип.

API v2 не принимает `anyOf` в описании функции, а `langchain_gigachat` на v2
отдаёт каждое `str | None` именно так.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

import pytest
from test_tools import _ctx

from langgraph_executor.aegra_agents.analyst_agent.agent import tool_schema
from langgraph_executor.aegra_agents.analyst_agent.tools import StarRatingArgs, build_tools


def _all_tools():
    ctx = _ctx(use_peer_aggregates=True, interactive_suggestions=True, interactive_chart=True)
    return build_tools(ctx)


def _multi_typed(node, path=""):
    """Пути, где у параметра не один тип."""
    found = []
    if isinstance(node, dict):
        if "anyOf" in node or "oneOf" in node or isinstance(node.get("type"), list):
            found.append(path)
        for key, value in node.items():
            found += _multi_typed(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found += _multi_typed(value, f"{path}[{i}]")
    return found


def test_every_tool_has_single_type_per_param_on_v2():
    for tool in [*_all_tools(), StarRatingArgs]:
        params = tool_schema.v2_tool_spec(tool)["function"]["parameters"]
        assert _multi_typed(params) == [], getattr(tool, "name", tool)


def test_flatten_keeps_type_description_and_required():
    tools = {t.name: t for t in _all_tools()}
    params = tool_schema.v2_tool_spec(tools["metric_card"])["function"]["parameters"]
    assert params["required"] == ["metric"]
    assert params["properties"]["person"]["type"] == "string"
    assert params["properties"]["person"]["description"].startswith("Чей результат")
    assert "default" not in params["properties"]["person"]  # был null
    assert params["properties"]["depth"] == {
        "type": "integer", "default": 2, "description": "Сколько уровней состава показать",
    }


def test_chart_nested_optional_fields_flattened():
    tools = {t.name: t for t in _all_tools()}
    spec = tool_schema.v2_tool_spec(tools["build_chart"])["function"]
    dumped = json.dumps(spec, ensure_ascii=False)
    assert "anyOf" not in dumped and '"null"' not in dumped
    trace = spec["parameters"]["properties"]["traces"]["items"]["properties"]
    assert trace["type"]["enum"] == ["line", "bar", "scatter"]
    assert trace["y"]["items"] == {"type": "number"}


def test_few_shot_examples_survive_conversion():
    tools = {t.name: t for t in _all_tools()}
    assert tool_schema.v2_tool_spec(tools["query_sql"])["function"]["few_shot_examples"]


def test_real_union_fails_loudly():
    with pytest.raises(ValueError, match="anyOf"):
        tool_schema._flatten(
            {"anyOf": [{"type": "string"}, {"type": "integer"}]}, "x"
        )


class _FakeLLM:
    def __init__(self, use_api_v2):
        self.use_api_v2 = use_api_v2
        self.bound = None

    def bind_tools(self, tools):
        self.bound = tools
        return self


def test_bind_tools_converts_only_on_v2():
    tools = _all_tools()
    legacy = _FakeLLM(False)
    tool_schema.bind_tools(legacy, tools)
    assert legacy.bound is tools

    v2 = _FakeLLM(True)
    tool_schema.bind_tools(v2, tools)
    assert [s["function"]["name"] for s in v2.bound] == [t.name for t in tools]
    assert _multi_typed(v2.bound) == []
