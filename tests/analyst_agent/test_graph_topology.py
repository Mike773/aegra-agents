"""Структура графа analyst_agent.

Один граф вместо двух: роутера-LLM, подграфа-аналитика и стадии пересказа
больше нет. Ход turn-based: один вызов = одно входящее сообщение → ответ → END.

Граф строится с заглушкой llm (узлы ленивы), поэтому ни сети, ни кредов не надо.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from langgraph_executor.aegra_agents.analyst_agent.graph import build_graph
from langgraph_executor.aegra_agents.analyst_agent.nodes.routing import (
    after_agent,
    after_prepare,
    need_load,
)


def _edges():
    compiled = build_graph(object())
    return {(e.source, e.target) for e in compiled.get_graph().edges}


def _nodes():
    return set(build_graph(object()).get_graph().nodes)


def test_first_turn_path():
    edges = _edges()
    assert ("__start__", "load_data") in edges
    assert ("load_data", "prepare") in edges
    assert ("prepare", "agent") in edges
    assert ("agent", "auto_insight") in edges
    assert ("auto_insight", "__end__") in edges


def test_followup_turn_path():
    edges = _edges()
    assert ("__start__", "begin_turn") in edges
    assert ("begin_turn", "agent") in edges
    assert ("agent", "__end__") in edges


def test_dashboard_path():
    edges = _edges()
    assert ("prepare", "plan_tasks") in edges
    assert ("plan_tasks", "agent") in edges
    # Последовательный цикл по задачам: ребро агента в самого себя.
    assert ("agent", "agent") in edges
    assert ("agent", "summarize") in edges
    assert ("summarize", "auto_insight") in edges


def test_no_router_or_analyzer_subgraph():
    nodes = _nodes()
    assert "route" not in nodes
    assert "call_json_analyzer" not in nodes
    assert "synthesize" not in nodes
    assert "ground_wiki_initial" not in nodes


def test_need_load_gate():
    assert need_load({}) == "load_data"
    assert need_load({"loaded": False}) == "load_data"
    assert need_load({"loaded": True}) == "begin_turn"


def test_after_prepare_routes_dashboard_only_with_tasks():
    assert after_prepare({}) == "agent"
    assert after_prepare({"dashboard_mode": True}) == "plan_tasks"
    assert after_prepare({"dashboard_mode": False, "tasks": [{"id": "t1"}]}) == "agent"
    assert after_prepare({"metrics_error": "нет данных"}) == "auto_insight"


def test_after_agent_loops_over_tasks_then_summarizes():
    assert after_agent({"turn_kind": "initial"}) == "auto_insight"
    assert after_agent({"turn_kind": "followup"}) == "__end__"
    dashboard = {"turn_kind": "dashboard", "tasks": [{"id": "t1"}, {"id": "t2"}]}
    assert after_agent({**dashboard, "task_idx": 1}) == "agent"
    assert after_agent({**dashboard, "task_idx": 2}) == "summarize"


def test_graph_compiles_with_checkpointer():
    from langgraph.checkpoint.memory import MemorySaver

    compiled = build_graph(object(), checkpointer=MemorySaver())
    assert compiled is not None


def test_module_level_graph_attribute_exists():
    from langgraph_executor.aegra_agents.analyst_agent import graph as graph_module

    assert hasattr(graph_module, "graph")
    assert hasattr(graph_module, "build_graph")


def test_registered_in_link_and_aegra_json():
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    link = (repo / "link.py").read_text(encoding="utf-8")
    assert "analyst_agent" in link
    assert '"analyst_agent_graph"' in link

    aegra = json.loads((repo / "aegra.json").read_text(encoding="utf-8"))
    assert aegra["graphs"]["analyst_agent"] == "./link.py:analyst_agent_graph"
