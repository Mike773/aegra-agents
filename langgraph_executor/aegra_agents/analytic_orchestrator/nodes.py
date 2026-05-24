from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_gigachat import GigaChat
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from .prompts import ASK_USER_PROMPT, RESPONDER_PROMPT, ROUTER_PROMPT
from .state import OrchestratorState

_VALID_INTENTS = {"knowledge", "json", "chat", "done"}


def make_ask_user_node():
    def ask_user(state: OrchestratorState) -> dict:
        user_text = interrupt({"prompt": ASK_USER_PROMPT})
        return {"messages": [HumanMessage(content=str(user_text))]}

    return ask_user


def make_route_node(llm: GigaChat):
    def route(state: OrchestratorState, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable", {})
        enabled = set(cfg.get("enabled_subagents") or ["knowledge", "json"])

        last_text = _last_user_text(state)
        ai = llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=last_text),
        ])
        label = (ai.content or "").strip().lower()

        if label not in _VALID_INTENTS:
            label = "chat"
        if label in {"knowledge", "json"} and label not in enabled:
            label = "chat"

        return {"intent": label}

    return route


def make_knowledge_node(knowledge_graph: CompiledStateGraph):
    def knowledge(state: OrchestratorState, config: RunnableConfig) -> dict:
        sub_input = {
            "messages": state.get("messages") or [],
            "query": _last_user_text(state),
        }
        result = knowledge_graph.invoke(sub_input, config=config)
        merged = dict(state.get("sub_results") or {})
        merged["knowledge"] = result.get("answer", "")
        return {"sub_results": merged}

    return knowledge


def make_json_node(json_graph: CompiledStateGraph):
    def json_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        sub_input = {
            "messages": state.get("messages") or [],
            "raw_json": _last_user_text(state),
        }
        result = json_graph.invoke(sub_input, config=config)
        merged = dict(state.get("sub_results") or {})
        merged["json"] = result.get("summary", "")
        return {"sub_results": merged}

    return json_node


def make_finalize_node(llm: GigaChat):
    def finalize(state: OrchestratorState, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable", {})
        system_prompt = cfg.get("system_prompt_override") or RESPONDER_PROMPT

        sub = state.get("sub_results") or {}
        ctx_parts: list[str] = []
        if "knowledge" in sub:
            ctx_parts.append(f"Результат поиска по знаниям:\n{sub['knowledge']}")
        if "json" in sub:
            ctx_parts.append(f"Результат разбора JSON:\n{sub['json']}")

        user_text = _last_user_text(state)
        body = user_text + ("\n\n" + "\n\n".join(ctx_parts) if ctx_parts else "")

        ai = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=body.strip()),
        ])
        return {"messages": [ai], "sub_results": {}}

    return finalize


def route_intent(state: OrchestratorState) -> str:
    intent = state.get("intent") or "chat"
    if intent == "knowledge":
        return "knowledge"
    if intent == "json":
        return "json"
    return "finalize"


def after_finalize(state: OrchestratorState) -> str:
    if state.get("intent") == "done":
        return "__end__"
    return "ask_user"


def _last_user_text(state: OrchestratorState) -> str:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""
