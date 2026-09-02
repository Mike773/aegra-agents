"""json_analyzer_v5: справочный wiki-контекст на входе (wiki_context).

Оркестратор передаёт готовый текстовый блок wiki-сниппетов полем wiki_context.
Стадия 1 подмешивает его в системный промпт после «Состава датасета» с рамкой
«справка, не данные»; стадия 2 добавляет его в user_content перед транскриптом.
Пустой/отсутствующий контекст → промпты байт-в-байт как раньше.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from langgraph_executor.aegra_agents.json_analyzer_v5 import agent_classic
from langgraph_executor.aegra_agents.json_analyzer_v5.agent_classic import (
    ClassicStrategy,
    compose_system_prompt,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.nodes import make_synthesize_node

OVERVIEW = {"dates": [], "metrics": [], "people": [], "elements": []}
WIKI = "Релевантные фрагменты wiki:\n- [AHT / Методика]: AHT считается без удержания."


def test_prompt_without_wiki_unchanged():
    assert compose_system_prompt(OVERVIEW) == compose_system_prompt(OVERVIEW, wiki_context=None)
    assert compose_system_prompt(OVERVIEW) == compose_system_prompt(OVERVIEW, wiki_context="  ")
    assert "wiki" not in compose_system_prompt(OVERVIEW).lower()


def test_prompt_with_wiki_has_block_and_framing():
    prompt = compose_system_prompt(OVERVIEW, wiki_context=WIKI)
    assert "AHT считается без удержания" in prompt
    # Рамка: это справка об интерпретации, а не данные — числа только из
    # инструментов, направление только из metric_type.
    assert "СПРАВОЧНЫЙ КОНТЕКСТ ИЗ КОРПОРАТИВНОЙ WIKI" in prompt
    assert "не данные" in prompt
    assert "metric_type" in prompt.split("СПРАВОЧНЫЙ КОНТЕКСТ")[1]
    # Справка идёт ПОСЛЕ состава датасета: инструментальные правила — раньше.
    assert prompt.index("СОСТАВ ЗАГРУЖЕННОГО ДАТАСЕТА") < prompt.index(
        "СПРАВОЧНЫЙ КОНТЕКСТ ИЗ КОРПОРАТИВНОЙ WIKI"
    )


def test_build_threads_wiki_into_agent(monkeypatch):
    captured = {}

    def fake_create_agent(model, tools, system_prompt):
        captured["system_prompt"] = system_prompt

        class _Agent:
            def with_config(self, *args, **kwargs):
                return self

        return _Agent()

    monkeypatch.setattr(agent_classic, "create_agent", fake_create_agent)
    ClassicStrategy().build(object(), [], OVERVIEW, wiki_context=WIKI)
    assert "AHT считается без удержания" in captured["system_prompt"]


class _FakeLLM:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages

        class _Resp:
            content = "ФАКТЫ: ..."

        return _Resp()


def _synthesize(state):
    llm = _FakeLLM()
    make_synthesize_node(llm)(state, None)
    return llm.messages


def test_synthesize_includes_wiki_with_framing():
    messages = _synthesize({
        "question": "что с AHT?",
        "gathered_facts": "get_metric(...) → AHT 300 сек",
        "parsed_rows": [],
        "wiki_context": WIKI,
    })
    human = messages[1].content
    assert "AHT считается без удержания" in human
    assert "только из данных" in human
    # Справка не должна попасть в системный промпт формата выжимки.
    assert "AHT считается без удержания" not in messages[0].content


def test_synthesize_without_wiki_unchanged():
    base_state = {
        "question": "что с AHT?",
        "gathered_facts": "get_metric(...) → AHT 300 сек",
        "parsed_rows": [],
    }
    without = _synthesize(dict(base_state))
    with_none = _synthesize({**base_state, "wiki_context": None})
    assert without[1].content == with_none[1].content
    assert "wiki" not in without[1].content.lower()
