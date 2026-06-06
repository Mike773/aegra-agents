"""Узлы графа json_analyzer: gather (загрузка + сбор tool-вызовами) и synthesize.

gather — async: доступ к LangGraph Store (кэш эмбеддингов) асинхронный, а
блокирующий код (sqlite3, llm.invoke, tool-loop) уведён в asyncio.to_thread,
чтобы не держать event loop. synthesize остаётся sync (в БД не ходит).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_gigachat import GigaChat
from langgraph.config import get_store
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from ..shared.clients import create_gigachat_embeddings
from .agent_base import extract_tool_steps, extract_tool_transcript
from .agent_classic import ClassicStrategy
from .analytics import compute_analytics
from .loader import load_dataset_obj
from .prompts import SYNTHESIS_PROMPT
from .relations_cache import sync_relations
from .sqlite_store import SqliteStore
from .store_cache import EmbeddingIndex, sync_embeddings
from .tools import build_tools

# Фолбэк-Store для standalone-прогонов (smoke/тесты), когда рантайм не прокинул
# Store через get_store(). Под aegra используется её Postgres-Store.
_FALLBACK_STORE: InMemoryStore | None = None


def _resolve_store() -> BaseStore:
    global _FALLBACK_STORE
    try:
        store = get_store()
    except Exception:
        store = None
    # get_store() ВНЕ aegra-рантайма (standalone-вызов подграфа) возвращает None,
    # а не бросает исключение — поэтому фоллбэк нужен и на None, иначе дальше
    # упадёт `None.asearch`. Под aegra store не None и используется он.
    if store is not None:
        return store
    if _FALLBACK_STORE is None:
        _FALLBACK_STORE = InMemoryStore()
    return _FALLBACK_STORE


def _prepare_store(rows: list[dict[str, Any]]) -> SqliteStore:
    """Блокирующая подготовка: in-memory SQLite + производная аналитика."""
    store = SqliteStore()
    store.load(rows)
    compute_analytics(store)
    return store


def _run_agent(
    store: SqliteStore,
    index: EmbeddingIndex,
    llm: GigaChat,
    question: str,
    embed_query: Callable[[str], list[float]],
) -> tuple[list[Any], bool]:
    """Блокирующий прогон стадии 1: tool-loop по SQLite + in-memory индексу."""
    tools = build_tools(store, index, embed_query=embed_query)
    strategy = ClassicStrategy()
    agent = strategy.build(llm, tools, store.schema_overview())
    return strategy.run(agent, [HumanMessage(content=question)])


def _last_human_text(messages: list[Any]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            content = m.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "".join(parts)
    return ""


def _parse_raw_json(raw: Any) -> dict | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def _resolve_inputs(state: dict, config: RunnableConfig) -> tuple[dict | None, str, str]:
    cfg = (config or {}).get("configurable", {}) if config else {}
    raw_obj = _parse_raw_json(state.get("raw_json"))
    question = (state.get("question") or "").strip()
    if not question:
        question = _last_human_text(state.get("messages") or []).strip()
    direction_key = (
        (cfg.get("direction_key") if isinstance(cfg, dict) else None)
        or state.get("direction_key")
        or ""
    ).strip()
    return raw_obj, question, direction_key


def make_gather_node(llm: GigaChat):
    embedder = create_gigachat_embeddings()

    async def gather(state: dict, config: RunnableConfig) -> dict:
        raw_obj, question, direction_key = _resolve_inputs(state, config)
        if raw_obj is None:
            return {
                "answer": "Не задан raw_json (ожидался JSON-датасет метрик).",
                "completed": True,
                "messages": [
                    AIMessage(content="Не задан raw_json (ожидался JSON-датасет метрик).")
                ],
            }
        if not question:
            return {
                "answer": "Не задан вопрос (question или последний HumanMessage пуст).",
                "completed": True,
                "messages": [
                    AIMessage(content="Не задан вопрос (question или последний HumanMessage пуст).")
                ],
            }
        if not direction_key:
            return {
                "answer": "Не задан direction_key (нужен для изоляции pgvector-кэша).",
                "completed": True,
                "messages": [
                    AIMessage(content="Не задан direction_key (нужен для изоляции pgvector-кэша).")
                ],
            }

        rows = load_dataset_obj(raw_obj)
        store = await asyncio.to_thread(_prepare_store, rows)

        # Кэш эмбеддингов — в LangGraph Store (подключение aegra). Доступ async,
        # сам подсчёт недостающих эмбеддингов (GigaChat) — внутри в to_thread.
        lg_store = _resolve_store()
        index = await sync_embeddings(
            lg_store,
            store,
            direction_key=direction_key,
            embed_documents=embedder.embed_documents,
        )

        # Граф смысловых связей метрик (Блок D ТЗ) — LLM по названиям/описаниям,
        # кэш в Store per-direction. Грузим в SQLite ДО сборки инструментов, чтобы
        # related_metrics/metric_tree видели его. Сбой изолирован внутри.
        relations = await sync_relations(
            lg_store, store, direction_key=direction_key, llm=llm
        )
        await asyncio.to_thread(store.load_relations, relations)

        # Tool-loop блокирующий (llm.invoke + sqlite) — уводим в поток; поиск
        # внутри идёт по in-memory индексу, обращений к БД нет.
        collected, completed = await asyncio.to_thread(
            _run_agent, store, index, llm, question, embedder.embed_query
        )

        transcript, tool_calls = extract_tool_transcript(collected)
        tool_steps = extract_tool_steps(collected)
        if tool_calls == 0:
            # Стадия 1 не вызвала ни одного инструмента — её прямой ответ
            # (последнее сообщение) и есть итог; synthesize_node его пробросит.
            direct = collected[-1].content if collected else ""
            if isinstance(direct, list):
                direct = "".join(
                    block.get("text", "")
                    for block in direct
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            return {
                "parsed_rows": rows,
                "gathered_facts": "",
                "tool_steps": tool_steps,
                "question": question,
                "completed": completed,
                "answer": direct or "",
            }

        return {
            "parsed_rows": rows,
            "gathered_facts": transcript,
            "tool_steps": tool_steps,
            "question": question,
            "completed": completed,
        }

    return gather


def make_synthesize_node(llm: GigaChat):
    def synthesize(state: dict, config: RunnableConfig) -> dict:
        # Если gather уже выставил answer (нет вызовов tools или ранний выход),
        # стадия 2 не нужна — пробрасываем как есть.
        if state.get("answer"):
            answer = state["answer"]
            return {"messages": [AIMessage(content=answer)]}

        question = (state.get("question") or "").strip()
        transcript = state.get("gathered_facts") or ""
        completed = state.get("completed", True)

        user_content = (
            f"Вопрос пользователя: {question}\n\n"
            f"Данные, собранные инструментами из базы:\n{transcript}"
        )
        response = llm.invoke(
            [
                SystemMessage(content=SYNTHESIS_PROMPT),
                HumanMessage(content=user_content),
            ]
        )
        answer = response.content if isinstance(response.content, str) else str(response.content)
        if not completed:
            answer += (
                "\n\n(Примечание: агент не уложился в лимит шагов сбора — "
                "ответ собран по тем данным, что успели получить.)"
            )
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    return synthesize


__all__ = ["make_gather_node", "make_synthesize_node"]
