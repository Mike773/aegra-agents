"""Узлы графа json_analyzer: gather (загрузка + сбор tool-вызовами) и synthesize.

Узлы синхронные: внутри блокирующие операции (sqlite3, psycopg, llm.invoke) —
LangGraph сам выполняет их в thread, если граф invoked через ainvoke.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_gigachat import GigaChat

from ..shared.clients import create_gigachat_embeddings
from .agent_base import extract_tool_transcript, synthesize_answer
from .agent_classic import ClassicStrategy
from .analytics import compute_analytics
from .loader import load_dataset_obj
from .pg_cache import PgCache, sync_embeddings
from .prompts import SYNTHESIS_PROMPT
from .sqlite_store import SqliteStore
from .tools import build_tools


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

    def gather(state: dict, config: RunnableConfig) -> dict:
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
        store = SqliteStore()
        store.load(rows)
        compute_analytics(store)

        # Размерность вектора берётся из существующей колонки таблицы
        # (создана миграцией migrations/json_analyzer/0001_initial.sql).
        # Рассинхрон между моделью и таблицей словится sync_embeddings.
        pg = PgCache()
        try:
            sync_embeddings(
                store,
                pg,
                direction_key=direction_key,
                embed_documents=embedder.embed_documents,
            )
            tools = build_tools(
                store,
                pg,
                direction_key=direction_key,
                embed_query=embedder.embed_query,
            )
            strategy = ClassicStrategy()
            agent = strategy.build(llm, tools, store.schema_overview())
            collected, completed = strategy.run(agent, [HumanMessage(content=question)])
        finally:
            pg.close()

        transcript, tool_calls = extract_tool_transcript(collected)
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
                "question": question,
                "completed": completed,
                "answer": direct or "",
            }

        return {
            "parsed_rows": rows,
            "gathered_facts": transcript,
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


# Внутри extract_tool_transcript ожидается, что synthesize_answer вызывается
# отдельно — здесь импорт оставлен для обратной совместимости с возможными
# внешними утилитами и интеграционными тестами.
_ = synthesize_answer
