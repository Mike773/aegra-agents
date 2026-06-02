"""Юнит-тесты wiki-grounding узла и сопутствующих хелперов.

Импортируем только nodes.py (не graph.py), поэтому GigaChat-креды/сеть не нужны.
Асинхронность гоняем через asyncio.run — pytest-asyncio в окружении нет.

Фейки: FakeLLM.invoke возвращает объект с .content (JSON-массив строк-запросов);
FakeEasyrag.ainvoke записывает вызовы и отдаёт заранее заданные сниппеты по запросу.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langchain_core.messages import HumanMessage

from langgraph_executor.aegra_agents.analytic_orchestrator import nodes
from langgraph_executor.aegra_agents.analytic_orchestrator.nodes import (
    _EASYRAG_RESET,
    _distinct_metric_specs,
    _parse_query_list_json,
    make_ground_wiki_node,
    make_route_node,
)


# --- Фейки -----------------------------------------------------------------

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


class FakeEasyrag:
    """Возвращает сниппеты по карте query -> list[snippet]; raise_on — наборы
    запросов, на которых .ainvoke кидает исключение."""

    def __init__(self, by_query=None, raise_on=None):
        self.by_query = by_query or {}
        self.raise_on = set(raise_on or [])
        self.calls: list[dict] = []

    async def ainvoke(self, payload):
        self.calls.append(payload)
        q = payload.get("query")
        if q in self.raise_on:
            raise RuntimeError(f"boom: {q}")
        return {"snippets": list(self.by_query.get(q, []))}


def _metrics(*names):
    return {
        "me": {"tabnum": 1, "fio": "Босс", "metrics": []},
        "employees": [
            {
                "tabnum": 2,
                "fio": "Сотрудник",
                "metrics": [
                    {"metric_name": n, "metric_description": f"описание {n}", "fact": 1}
                    for n in names
                ],
            }
        ],
    }


def _state(**over):
    base = {
        "messages": [HumanMessage(content="Что с метриками?")],
        "metrics": _metrics("CSAT", "AHT"),
        "direction_key": "dir-1",
        "reasoning_trace": [],
    }
    base.update(over)
    return base


def _cfg(**configurable):
    return {"configurable": configurable}


def _snip(section_id, sim, page="P", body="b"):
    return {
        "section_id": section_id,
        "page_title": page,
        "section_title": "S",
        "body_md": body,
        "similarity": sim,
    }


# --- _parse_query_list_json ------------------------------------------------

def test_parse_plain_array():
    assert _parse_query_list_json('["a", "b"]', 3) == ["a", "b"]


def test_parse_fenced_json():
    assert _parse_query_list_json('```json\n["x"]\n```', 3) == ["x"]


def test_parse_dedup_casefold_and_cap():
    assert _parse_query_list_json('["A", "a", "b", "c", "d"]', 2) == ["A", "b"]


def test_parse_non_list_and_garbage():
    assert _parse_query_list_json('{"q": 1}', 3) == []
    assert _parse_query_list_json("не json", 3) == []
    assert _parse_query_list_json("", 3) == []


def test_parse_coerces_and_skips_empty():
    assert _parse_query_list_json('["  ", "ok", 0]', 3) == ["ok"]


# --- _distinct_metric_specs -------------------------------------------------

def test_distinct_metric_specs_dedup_by_name():
    specs = _distinct_metric_specs(_metrics("CSAT", "CSAT", "AHT"))
    names = [s["name"] for s in specs]
    assert names == ["CSAT", "AHT"]
    assert specs[0]["description"] == "описание CSAT"


def test_distinct_metric_specs_bad_input():
    assert _distinct_metric_specs(None) == []
    assert _distinct_metric_specs("garbage") == []


# --- make_ground_wiki_node: happy path -------------------------------------

def test_happy_path_merge_dedup_sort_cap():
    llm = FakeLLM('["методика CSAT", "норматив AHT"]')
    easy = FakeEasyrag(by_query={
        "методика CSAT": [_snip("s1", 0.5), _snip("s2", 0.9)],
        # тот же section_id s2 с большей similarity — должен победить дубликат выше
        "норматив AHT": [_snip("s2", 0.95), _snip("s3", 0.7)],
    })
    node = make_ground_wiki_node(llm, easy)
    out = asyncio.run(node(_state(), _cfg(easyrag_top_k=2)))

    assert len(easy.calls) == 2
    snippets = out["easyrag_snippets"]
    # cap=2, сорт по similarity desc, дедуп по section_id (s2 берём 0.95)
    sims = [s["similarity"] for s in snippets]
    assert sims == sorted(sims, reverse=True)
    assert len(snippets) == 2
    # дедуп: s1=0.5, s2=0.95(победил), s3=0.7 → топ-2 это s2 и s3
    assert {s["section_id"] for s in snippets} == {"s2", "s3"}
    s2 = next(s for s in snippets if s["section_id"] == "s2")
    assert s2["similarity"] == 0.95
    assert out["easyrag_query"] == "методика CSAT | норматив AHT"
    assert out["easyrag_error"] is None
    # трасса содержит kb_hit
    assert any(st["kind"] == "kb_hit" for st in out["reasoning_trace"])


# --- no-op ветки ------------------------------------------------------------

def test_noop_easyrag_disabled():
    llm = FakeLLM('["q"]')
    easy = FakeEasyrag()
    node = make_ground_wiki_node(llm, easy)
    out = asyncio.run(node(_state(), _cfg(easyrag_enabled=False)))
    assert out == {}
    assert easy.calls == []
    assert llm.calls == []


def test_noop_grounding_disabled():
    llm = FakeLLM('["q"]')
    easy = FakeEasyrag()
    node = make_ground_wiki_node(llm, easy)
    out = asyncio.run(node(_state(), _cfg(wiki_grounding_enabled=False)))
    assert out == {}
    assert easy.calls == []


def test_noop_no_metrics_or_error_or_direction():
    node = make_ground_wiki_node(FakeLLM("[]"), FakeEasyrag())
    assert asyncio.run(node(_state(metrics=None), _cfg())) == {}
    assert asyncio.run(node(_state(metrics_error="boom"), _cfg())) == {}
    assert asyncio.run(node(_state(direction_key=""), _cfg())) == {}


def test_noop_llm_returns_empty():
    llm = FakeLLM("[]")
    easy = FakeEasyrag()
    node = make_ground_wiki_node(llm, easy)
    out = asyncio.run(node(_state(), _cfg()))
    assert out == {}
    assert easy.calls == []  # без запросов — easyrag не зван


def test_noop_llm_returns_garbage():
    out = asyncio.run(make_ground_wiki_node(FakeLLM("definitely not json"), FakeEasyrag())(_state(), _cfg()))
    assert out == {}


# --- analytics-путь: выводы аналитика попадают в промпт --------------------

def test_analytics_answer_in_prompt():
    llm = FakeLLM('["q"]')
    easy = FakeEasyrag(by_query={"q": [_snip("s1", 0.8)]})
    node = make_ground_wiki_node(llm, easy)
    asyncio.run(node(_state(analytics_answer="CSAT просел до 70%"), _cfg()))
    human = llm.calls[0][1].content
    assert "CSAT просел до 70%" in human


# --- частичный / полный сбой easyrag ---------------------------------------

def test_partial_failure_keeps_snippets_no_error():
    llm = FakeLLM('["ok", "bad"]')
    easy = FakeEasyrag(by_query={"ok": [_snip("s1", 0.8)]}, raise_on=["bad"])
    node = make_ground_wiki_node(llm, easy)
    out = asyncio.run(node(_state(), _cfg()))
    assert out["easyrag_error"] is None
    assert [s["section_id"] for s in out["easyrag_snippets"]] == ["s1"]


def test_all_failures_set_error_empty_snippets():
    llm = FakeLLM('["a", "b"]')
    easy = FakeEasyrag(raise_on=["a", "b"])
    node = make_ground_wiki_node(llm, easy)
    out = asyncio.run(node(_state(), _cfg()))
    assert out["easyrag_snippets"] == []
    assert out["easyrag_error"] is not None


# --- пустой результат → stub-fallback (без БД, патчим хелпер) ---------------

def test_empty_snippets_triggers_stub_lookup(monkeypatch):
    async def fake_stub(direction_key, query):
        assert query == "q1 | q2"
        return [{"slug": "csat", "title": "CSAT"}]

    monkeypatch.setattr(nodes, "_find_relevant_stub_pages", fake_stub)
    llm = FakeLLM('["q1", "q2"]')
    easy = FakeEasyrag()  # ничего не находит
    node = make_ground_wiki_node(llm, easy)
    out = asyncio.run(node(_state(), _cfg()))
    assert out["easyrag_snippets"] == []
    assert out["easyrag_stub_pages"] == [{"slug": "csat", "title": "CSAT"}]
    assert out["easyrag_error"] is None


# --- регресс stale-snippet: route чистит easyrag-поля ----------------------

def test_route_clears_easyrag_empty_text():
    route = make_route_node(FakeLLM("chat"))
    out = route({"messages": []})
    for k, v in _EASYRAG_RESET.items():
        assert out[k] == v


def test_route_clears_easyrag_pending():
    route = make_route_node(FakeLLM("chat"))
    out = route(_state(pending_assignments=[{"title": "x"}]))
    for k, v in _EASYRAG_RESET.items():
        assert out[k] == v


def test_route_clears_easyrag_classified():
    route = make_route_node(FakeLLM("analytics"))
    out = route(_state())
    assert out["intent"] == "analytics"
    for k, v in _EASYRAG_RESET.items():
        assert out[k] == v
