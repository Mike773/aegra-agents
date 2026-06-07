"""Контракт сообщений хода: промежуточные шаги + итог последним.

Рабочие узлы кладут шаговые AIMessage (additional_kwargs.orchestrator_step),
терминальные листья — итог (orchestrator_final). Шаги вырезаются из истории,
подаваемой в LLM. Прогресс гасится флагом emit_progress_messages=false.

Импортируем только nodes.py — без GigaChat/сети. Фейки заимствуем по образцу
test_ground_wiki (FakeLLM/FakeEasyrag через локальные определения).
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langchain_core.messages import AIMessage, HumanMessage

from langgraph_executor.aegra_agents.analytic_orchestrator.nodes import (
    _FINAL_KEY,
    _STEP_KEY,
    _final_message,
    _history_for_llm,
    _is_step,
    _last_ai_text,
    _step_update,
    make_extract_assignments_node,
    make_ground_wiki_node,
    make_propose_assignments_node,
)


# --- Фейки (как в test_ground_wiki) ----------------------------------------

class _AIMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, messages):
        return _AIMsg(self._content)


class FakeEasyrag:
    def __init__(self, by_query=None):
        self.by_query = by_query or {}

    async def ainvoke(self, payload):
        return {"snippets": list(self.by_query.get(payload.get("query"), []))}


def _metrics(*names):
    return {
        "me": {"tabnum": 1, "fio": "Босс", "metrics": []},
        "employees": [{
            "tabnum": 2, "fio": "Сотрудник",
            "metrics": [
                {"metric_name": n, "metric_description": f"опис {n}", "fact": 1}
                for n in names
            ],
        }],
    }


def _state(**over):
    base = {
        "messages": [HumanMessage(content="Что с метриками?")],
        "metrics": _metrics("CSAT"),
        "direction_key": "dir-1",
        "reasoning_trace": [],
    }
    base.update(over)
    return base


def _cfg(**configurable):
    return {"configurable": configurable}


def _snip(section_id, sim):
    return {"section_id": section_id, "page_title": "P", "section_title": "S",
            "body_md": "b", "similarity": sim}


# --- хелперы шаг/итог -------------------------------------------------------

def test_step_update_tags_and_is_detected():
    out = _step_update(_cfg(), "📖 Нашёл в wiki 2 фрагмент(ов).")
    msg = out["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.additional_kwargs.get(_STEP_KEY) is True
    assert _is_step(msg)
    # dict-форма (как приходит по HTTP)
    assert _is_step({"additional_kwargs": {_STEP_KEY: True}})


def test_step_update_suppressed_when_disabled_or_empty():
    assert _step_update(_cfg(emit_progress_messages=False), "x") == {}
    assert _step_update(_cfg(emit_progress_messages="false"), "x") == {}
    assert _step_update(_cfg(), "   ") == {}
    assert _step_update(_cfg(), "") == {}


def test_final_message_tagged_and_not_step():
    msg = _final_message("ИТОГ")
    assert msg.additional_kwargs.get(_FINAL_KEY) is True
    assert not _is_step(msg)


# --- история для LLM: шаги вырезаются --------------------------------------

def test_history_for_llm_drops_steps():
    history = [
        HumanMessage(content="вопрос"),
        AIMessage(content="📖 шаг", additional_kwargs={_STEP_KEY: True}),
        _final_message("итог-ответ"),
    ]
    out = _history_for_llm(history)
    contents = [m.content for m in out]
    assert "📖 шаг" not in contents
    assert contents == ["вопрос", "итог-ответ"]


def test_last_ai_text_skips_steps():
    st = {"messages": [
        AIMessage(content="настоящий итог прошлого хода"),
        AIMessage(content="📊 шаг", additional_kwargs={_STEP_KEY: True}),
    ]}
    assert _last_ai_text(st) == "настоящий итог прошлого хода"


# --- рабочий узел кладёт шаг, итог его не содержит -------------------------

def test_ground_wiki_emits_step_message():
    llm = FakeLLM('["методика CSAT"]')
    easy = FakeEasyrag(by_query={"методика CSAT": [_snip("s1", 0.9)]})
    out = asyncio.run(make_ground_wiki_node(llm, easy)(_state(), _cfg()))
    msgs = out.get("messages") or []
    assert len(msgs) == 1 and _is_step(msgs[0])
    assert "wiki" in msgs[0].content.lower()


def test_ground_wiki_no_step_when_progress_disabled():
    llm = FakeLLM('["методика CSAT"]')
    easy = FakeEasyrag(by_query={"методика CSAT": [_snip("s1", 0.9)]})
    out = asyncio.run(make_ground_wiki_node(llm, easy)(
        _state(), _cfg(emit_progress_messages=False)))
    assert "messages" not in out
    assert out["easyrag_snippets"]  # работа сделана, просто без шага


def test_extract_assignments_step_then_propose_final_last():
    # extract кладёт шаг (без итога), propose — итог. Итог помечен final.
    llm = FakeLLM(
        '{"insights": [{"type": "problem", "metric_id": "1", '
        '"metric_name": "CSAT", "text": "CSAT ниже плана"}]}'
    )
    # Источник фактов — ответ агента в диалоге: добавляем его в state.
    state = _state(messages=[
        HumanMessage(content="Что с метриками?"),
        AIMessage(content="CSAT = 70%, ниже плана."),
        HumanMessage(content="оформи поручения"),
    ], metrics=_metrics("CSAT"))
    extract = make_extract_assignments_node(llm)
    ex_out = asyncio.run(extract(state, _cfg()))
    ex_msgs = ex_out.get("messages") or []
    assert len(ex_msgs) == 1 and _is_step(ex_msgs[0])
    assert ex_out["pending_assignments"][0]["metric_name"] == "CSAT"

    propose = make_propose_assignments_node()
    pr_out = propose({"pending_assignments": ex_out["pending_assignments"]})
    final = pr_out["messages"][0]
    assert final.additional_kwargs.get(_FINAL_KEY) is True
    assert not _is_step(final)
