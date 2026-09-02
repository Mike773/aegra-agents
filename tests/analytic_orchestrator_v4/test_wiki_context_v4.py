"""analytic_orchestrator_v4: проброс wiki-сниппетов на вход json_analyzer_v5.

_wiki_snippets_text отдаёт только строки сниппетов (без респондерных инструкций
про заглушки/ошибки); _run_analyzer кладёт их во вход подграфа полем
wiki_context; call_json_analyzer передаёт сниппеты текущего state.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from langchain_core.messages import HumanMessage

from langgraph_executor.aegra_agents.analytic_orchestrator_v4.nodes import (
    _run_analyzer,
    _wiki_snippets_text,
    make_call_json_analyzer_node,
)

SNIPPETS = [
    {
        "page_title": "AHT",
        "section_title": "Методика",
        "similarity": 0.91,
        "body_md": "AHT считается без удержания.",
    },
]

METRICS = {"me": None, "employees": [{"fio": "Иванов Иван", "metrics": []}]}


class FakeGraph:
    def __init__(self):
        self.payload = None

    async def ainvoke(self, payload):
        self.payload = payload
        return {"answer": "ок", "tool_steps": []}


def test_wiki_snippets_text_formats_snippets():
    text = _wiki_snippets_text({"easyrag_snippets": SNIPPETS})
    assert "AHT считается без удержания." in text
    assert "AHT / Методика" in text


def test_wiki_snippets_text_empty_and_stubs_give_none():
    assert _wiki_snippets_text({}) is None
    assert _wiki_snippets_text({"easyrag_snippets": []}) is None
    # Заглушки и ошибки — респондерная механика, аналитику они не нужны.
    assert _wiki_snippets_text({
        "easyrag_snippets": [],
        "easyrag_stub_pages": [{"title": "AHT"}],
    }) is None
    assert _wiki_snippets_text({
        "easyrag_snippets": [],
        "easyrag_error": "timeout",
    }) is None


def test_run_analyzer_passes_wiki_context():
    graph = FakeGraph()
    asyncio.run(_run_analyzer(
        graph, METRICS, "что происходит?", "dir-1", wiki_context="Wiki-справка",
    ))
    assert graph.payload["wiki_context"] == "Wiki-справка"


def test_run_analyzer_default_wiki_context_is_none():
    graph = FakeGraph()
    asyncio.run(_run_analyzer(graph, METRICS, "что происходит?", "dir-1"))
    assert graph.payload["wiki_context"] is None


def test_call_json_analyzer_forwards_state_snippets():
    graph = FakeGraph()
    node = make_call_json_analyzer_node(graph)
    state = {
        "messages": [HumanMessage(content="почему просел AHT?")],
        "metrics": METRICS,
        "direction_key": "dir-1",
        "easyrag_snippets": SNIPPETS,
    }
    asyncio.run(node(state, {"configurable": {}}))
    assert "AHT считается без удержания." in graph.payload["wiki_context"]
