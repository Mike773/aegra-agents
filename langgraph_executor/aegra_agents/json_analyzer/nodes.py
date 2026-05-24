from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_gigachat import GigaChat

from .prompts import SYSTEM_PROMPT
from .state import JsonAnalyzerState


def make_parse_input_node():
    def parse_input(state: JsonAnalyzerState) -> dict:
        raw: Any = state.get("raw_json")
        if raw is None:
            raw = _last_user_text(state)

        if isinstance(raw, (dict, list)):
            parsed: Any = raw
        elif isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                parsed = {"_parse_error": str(e), "_raw_excerpt": raw[:500]}
        else:
            parsed = {"_parse_error": f"unsupported raw_json type: {type(raw).__name__}"}

        return {"parsed": parsed, "findings": [], "summary": ""}

    return parse_input


def make_analyze_node(llm: GigaChat):
    def analyze(state: JsonAnalyzerState, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable", {})
        system_prompt = cfg.get("system_prompt_override") or SYSTEM_PROMPT
        schema_hint = cfg.get("schema_hint")
        # TODO: max_depth — обрезать сериализацию по глубине вложенности.
        _ = cfg.get("max_depth", 5)

        parsed = state.get("parsed")
        try:
            payload = json.dumps(parsed, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            payload = repr(parsed)
        payload = payload[:8000]

        user_parts = [f"JSON-документ:\n{payload}"]
        if schema_hint:
            user_parts.append(
                "\nОжидаемая схема (подсказка):\n"
                + json.dumps(schema_hint, ensure_ascii=False)
            )

        ai = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="\n".join(user_parts)),
        ])
        # TODO: распарсить findings/summary из ответа структурированно.
        return {"messages": [ai], "summary": ai.content, "findings": []}

    return analyze


def _last_user_text(state: JsonAnalyzerState) -> str:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""
