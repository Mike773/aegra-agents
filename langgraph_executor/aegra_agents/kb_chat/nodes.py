"""Узлы графа kb_chat.

``route`` классифицирует реплику (kb / chat), ``retrieve`` дёргает easyrag-подграф
под последний вопрос, ``respond`` отвечает с опорой на найденные фрагменты и всю
историю диалога. Все узлы возвращают частичное обновление state.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_gigachat import GigaChat

from .prompts import CHAT_SYSTEM_PROMPT, ROUTER_PROMPT
from .state import KbChatState

_DEFAULT_TOP_K = 5
_SNIPPET_PREVIEW = 400
_ROUTE_LABELS = {"kb", "chat"}


def _last_user_text(state: KbChatState) -> str:
    """Текст последней реплики пользователя (учитывает list-content блоки)."""
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            content = m.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
    return ""


def _resolve_direction_key(state: KbChatState, config: RunnableConfig) -> str:
    cfg = (config or {}).get("configurable") or {}
    return (cfg.get("direction_key") or state.get("direction_key") or "").strip()


def make_route_node(llm: GigaChat):
    """Классифицирует реплику: искать в базе знаний (kb) или просто болтать (chat)."""

    def route(state: KbChatState, config: RunnableConfig) -> dict:
        last_text = _last_user_text(state)
        direction_key = _resolve_direction_key(state, config)

        # Нет вопроса или нет direction_key — искать негде, отвечаем как чат.
        if not last_text or not direction_key:
            return {"intent": "chat", "direction_key": direction_key}

        ai = llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=last_text),
        ])
        label = (ai.content or "").strip().lower()
        if label not in _ROUTE_LABELS:
            # Сомнение трактуем в пользу базы знаний — это её профиль.
            label = "kb"
        return {"intent": label, "direction_key": direction_key}

    return route


def make_retrieve_node(easyrag_graph: Any):
    """Дёргает easyrag-подграф под последний вопрос пользователя.

    Подграф эмбеддит запрос, ищет релевантные секции wiki по ``direction_key`` и
    пишет gap, если ничего не нашёл. Ошибки изолируются — respond ответит без
    контекста базы знаний (и честно об этом скажет).
    """

    async def retrieve(state: KbChatState, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable") or {}
        direction_key = _resolve_direction_key(state, config)
        last_text = _last_user_text(state)
        if not direction_key or not last_text:
            return {"snippets": [], "snippet_error": None}

        top_k = int(cfg.get("top_k") or state.get("top_k") or _DEFAULT_TOP_K)
        try:
            result = await easyrag_graph.ainvoke({
                "query": last_text,
                "direction_key": direction_key,
                "top_k": top_k,
            })
            return {"snippets": result.get("snippets") or [], "snippet_error": None}
        except Exception as exc:  # noqa: BLE001 — внешний подграф (сеть/БД), сужать нечем
            return {
                "snippets": [],
                "snippet_error": f"{type(exc).__name__}: {exc}"[:300],
            }

    return retrieve


def make_respond_node(llm: GigaChat):
    """Отвечает пользователю: системный промпт + блок базы знаний + история диалога."""

    def respond(state: KbChatState, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable") or {}
        system_prompt = cfg.get("system_prompt_override") or CHAT_SYSTEM_PROMPT

        parts: list[str] = [system_prompt]
        block = _snippets_block(state)
        if block:
            parts.append(block)
        system_text = "\n\n".join(parts)

        messages: list[Any] = [SystemMessage(content=system_text)]
        messages.extend(state.get("messages") or [])

        ai = llm.invoke(messages)
        return {"messages": [ai]}

    return respond


def after_route(state: KbChatState) -> str:
    return "retrieve" if state.get("intent") == "kb" else "respond"


def _snippets_block(state: KbChatState) -> str | None:
    """Системный блок с фрагментами из базы знаний (или указание их отсутствия)."""
    snippets = state.get("snippets") or []
    err = state.get("snippet_error")
    if not snippets:
        if err:
            return (
                f"База знаний временно недоступна ({err}). Скажи пользователю, что "
                "сейчас не получается обратиться к базе знаний, и не выдумывай факты."
            )
        if state.get("intent") == "kb":
            return (
                "По этому вопросу в базе знаний ничего не нашлось. Честно скажи, что "
                "в базе знаний нет информации по этой теме, и не выдумывай факты."
            )
        return None

    lines = ["Релевантные фрагменты из базы знаний (по направлению пользователя):"]
    for s in snippets[:5]:
        page = s.get("page_title") or s.get("slug") or "-"
        title = s.get("section_title") or s.get("anchor") or "-"
        sim = s.get("similarity")
        sim_str = f" sim={sim:.2f}" if isinstance(sim, (int, float)) else ""
        body = (s.get("body_md") or "").strip().replace("\n", " ")
        if len(body) > _SNIPPET_PREVIEW:
            body = body[:_SNIPPET_PREVIEW] + "…"
        lines.append(f"- [{page} / {title}{sim_str}]: {body}")
    lines.append("")
    lines.append(
        "Отвечай, опираясь строго на эти фрагменты. Если их недостаточно — честно "
        "скажи об этом, не выдумывай."
    )
    return "\n".join(lines)


__all__ = [
    "after_route",
    "make_respond_node",
    "make_retrieve_node",
    "make_route_node",
]
