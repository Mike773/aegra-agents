"""Узел агента — единственное место, где модель работает с данными.

Обычный ход: один цикл вызовов инструментов, его финальный текст и есть ответ
руководителю. Составной разбор: тот же узел отрабатывает ОДНУ задачу за вызов,
а граф возвращает управление сюда же, пока задачи не кончатся.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from ..agent.loop import run_tool_loop
from ..agent.runctx import RunContext
from ..agent.trace import trace_steps
from ..config import RunConfig
from ..contract.describe import with_description
from ..contract.messages import final_message, history_for_llm, last_user_text, step_updates
from ..db import analytics, core, enrich
from ..dashboard.summarizer import prior_results_digest
from ..deviations import ledger as ledger_mod
from ..deviations.format import deviations_block
from ..prompts import PromptContext, compose_system_prompt
from ..tools import build_tools
from .prepare import focus_person_key

logger = logging.getLogger(__name__)

# Первое сообщение диалога — брифинг; если он пуст, спрашиваем общий разбор.
_DEFAULT_QUESTION = "Что происходит с показателями сотрудника?"


def _org_block(state: Any) -> str:
    """Кого разбираем — короткая справка для модели."""
    metrics = state.get("metrics")
    if not isinstance(metrics, dict):
        return ""
    me = metrics.get("me") or {}
    parts = []
    boss = str((me.get("fio") if isinstance(me, dict) else "") or "").strip()
    if boss:
        parts.append(f"руководитель — {boss}")
    names = [
        str(e.get("fio") or "").strip()
        for e in (metrics.get("employees") or [])
        if isinstance(e, dict) and str(e.get("fio") or "").strip()
    ]
    if names:
        position = str(state.get("position") or "").strip()
        pos = f" (позиция: {position})" if position else ""
        parts.append("в фокусе анализа — " + ", ".join(names) + pos)
    return "Кого анализируем: " + "; ".join(parts) + "." if parts else ""


async def _rebuild_db(state: Any) -> Any:
    """База пересобирается каждый ход из чекпойнтнутого сырого датасета."""

    def _build() -> Any:
        db = core.build_run_db(state.get("metrics"), state.get("aggregates"))
        return db

    return await asyncio.to_thread(_build)


def make_agent_node(llm: Any, easyrag_graph: Any = None):
    async def agent(state: Any, config: RunnableConfig) -> dict:
        cfg = RunConfig.from_config(config)
        db = await _rebuild_db(state)
        person_key = focus_person_key(db, state.get("employee_tabnum"))
        deviations = state.get("deviations") or []
        turn_no = int(state.get("turn_no") or 1)
        log = ledger_mod.DeviationLedger(db, deviations, turn=turn_no)

        ctx = RunContext(
            db=db,
            person_key=person_key,
            direction_key=str(state.get("direction_key") or ""),
            ledger=log,
            easyrag_graph=easyrag_graph,
            easyrag_enabled=cfg.easyrag_enabled,
            easyrag_top_k=cfg.easyrag_top_k,
            gap_on_unanswered=cfg.gap_on_unanswered,
            use_peer_aggregates=cfg.use_peer_aggregates and db.has_aggregates,
            text2sql_enabled=cfg.text2sql_enabled,
        )
        tools = build_tools(ctx)

        turn_kind = state.get("turn_kind") or "initial"
        tasks = state.get("tasks") or []
        task_idx = int(state.get("task_idx") or 0)
        is_dashboard = turn_kind == "dashboard" and task_idx < len(tasks)

        # Блоки промпта: посчитанные на первом ходе плюс свежая карта отклонений.
        enrichment = state.get("enrichment_block") or enrich.enrichment_block(
            db, person_key=person_key
        )
        catalog = state.get("catalog_block") or enrich.catalog_block(db, person_key=person_key)

        task_block = ""
        if is_dashboard:
            task = tasks[task_idx]
            task_block = (
                f"ЗАДАЧА {task_idx + 1} из {len(tasks)}: {task.get('title')}\n"
                + prior_results_digest(state.get("task_results") or [], task)
            )
            question = task.get("question") or task.get("title") or _DEFAULT_QUESTION
            budget = cfg.task_tool_budget
            history: list[Any] = []
        else:
            question = (last_user_text(state) or "").strip()
            if turn_kind == "initial":
                question = question or state.get("briefing") or _DEFAULT_QUESTION
            else:
                question = question or _DEFAULT_QUESTION
            budget = cfg.tool_budget
            history = history_for_llm(state.get("messages") or [])[:-1]

        prompt = compose_system_prompt(
            PromptContext(
                enrichment_block=enrichment,
                catalog_block=catalog,
                knowledge_block=state.get("knowledge_block") or "",
                deviations_block=deviations_block(log.select(limit=12)),
                org_block=_org_block(state),
                memory_context=state.get("memory_context"),
                briefing=state.get("briefing"),
                task_block=task_block,
                turn_kind="dashboard" if is_dashboard else turn_kind,
                has_stars=db.has_stars,
                tool_budget=budget,
                has_sql=cfg.text2sql_enabled,
                system_prompt_override=cfg.system_prompt_override,
            )
        )

        result = await run_tool_loop(
            llm=llm,
            tools=tools,
            system_prompt=prompt,
            history=history,
            question=question,
            budget=budget,
        )

        trace = (state.get("reasoning_trace") or []) + trace_steps(result.tool_steps)
        update: dict[str, Any] = {
            "reasoning_trace": trace,
            "deviations": log.to_state(),
            "intent": _intent_of(ctx),
            **_wiki_fields(ctx),
        }
        if not result.completed:
            update["analytics_error"] = f"Сбор данных завершён досрочно: {result.stop_reason}"

        if is_dashboard:
            task = tasks[task_idx]
            update["task_results"] = (state.get("task_results") or []) + [
                {
                    "id": task.get("id"),
                    "title": task.get("title"),
                    "question": question,
                    "answer_md": result.final_text,
                    "error": None if result.completed else result.stop_reason,
                }
            ]
            update["task_idx"] = task_idx + 1
            steps = step_updates(
                config,
                result.step_texts
                + [f"✅ Задача {task_idx + 1}/{len(tasks)} «{task.get('title')}» — готово."],
            )
            if steps:
                update["messages"] = steps
            return update

        text = await with_description(
            result.final_text, {**state, **update}, config, llm=llm, question=question
        )
        messages = step_updates(config, result.step_texts) + [final_message(text, config)]
        update["messages"] = messages
        update["analytics_question"] = question
        update["analytics_answer"] = result.final_text
        if turn_kind == "initial" and not state.get("metrics_summary"):
            update["metrics_summary"] = result.final_text
        return update

    return agent


def _intent_of(ctx: RunContext) -> str:
    """Совместимость с v4: чем занимался ход."""
    if ctx.used_data_tools:
        return "analytics"
    if ctx.used_wiki_tool:
        return "wiki"
    return "chat"


def _wiki_fields(ctx: RunContext) -> dict[str, Any]:
    """Поля easyrag_* для описи источников — как в v4."""
    if not ctx.wiki_queries:
        return {}
    return {
        "easyrag_query": " | ".join(ctx.wiki_queries),
        "easyrag_snippets": ctx.wiki_snippets,
        "easyrag_stub_pages": ctx.wiki_stub_pages,
        "easyrag_error": ctx.wiki_error,
    }


__all__ = ["make_agent_node"]
