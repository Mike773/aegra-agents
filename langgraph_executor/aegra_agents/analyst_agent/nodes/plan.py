"""Узлы составного разбора: планирование задач и итоговая сводка."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import RunConfig
from ..contract.describe import with_description
from ..contract.messages import final_message, step_update
from ..dashboard.planner import plan_tasks
from ..dashboard.summarizer import SUMMARIZER_PROMPT_TAIL, summary_input
from ..db import analytics, core
from ..deviations import builder, tasks as tasks_mod
from ..deviations.format import deviations_block
from ..prompts import PromptContext, compose_system_prompt

logger = logging.getLogger(__name__)


def make_plan_tasks_node(llm: Any):
    async def plan(state: Any, config: RunnableConfig) -> dict:
        cfg = RunConfig.from_config(config)
        tasks = await asyncio.to_thread(
            plan_tasks,
            llm,
            briefing=state.get("briefing"),
            data_summary=state.get("enrichment_block") or "",
            max_tasks=cfg.max_tasks,
        )
        if not tasks:
            # Разложить запрос не удалось — работаем обычным одношаговым ходом.
            return {
                "tasks": [],
                "turn_kind": "initial",
                "dashboard_mode": False,
                "reasoning_trace": (state.get("reasoning_trace") or [])
                + [
                    {
                        "stage": "plan",
                        "kind": "decision",
                        "summary": "Составной разбор не потребовался: задач не выделено.",
                        "detail": {},
                    }
                ],
            }

        # Карта отклонений пересобирается под задачи: то, что просил
        # руководитель, поднимается в приоритете.
        db = await asyncio.to_thread(_rebuild, state)
        resolved = await asyncio.to_thread(tasks_mod.resolve_task_metrics, db, tasks)
        person_key = str(state.get("employee_tabnum") or "")
        resolved_key = db.resolve_person(person_key) or person_key
        deviations = await asyncio.to_thread(
            builder.build_deviations,
            db,
            focus_person_key=resolved_key,
            tasks=resolved,
            previous=state.get("deviations") or [],
            turn=int(state.get("turn_no") or 1),
        )
        titles = "; ".join(t["title"] for t in resolved[:5])
        return {
            **step_update(config, f"🗂 Разбил запрос на {len(resolved)} задач: {titles}."),
            "tasks": resolved,
            "task_idx": 0,
            "task_results": [],
            "deviations": deviations,
            "turn_kind": "dashboard",
            "reasoning_trace": (state.get("reasoning_trace") or [])
            + [
                {
                    "stage": "plan",
                    "kind": "decision",
                    "summary": f"Разбил запрос на {len(resolved)} задач: {titles}.",
                    "detail": {"tasks": [t["title"] for t in resolved]},
                }
            ],
        }

    return plan


def _rebuild(state: Any) -> Any:
    db = core.build_run_db(state.get("metrics"), state.get("aggregates"))
    analytics.compute_analytics(db.conn)
    return db


def make_summarize_node(llm: Any):
    async def summarize(state: Any, config: RunnableConfig) -> dict:
        cfg = RunConfig.from_config(config)
        results = state.get("task_results") or []
        body = summary_input(results)
        if not body:
            text = "Не удалось выполнить ни одной задачи разбора."
        else:
            prompt = compose_system_prompt(
                PromptContext(
                    deviations_block=deviations_block(state.get("deviations") or []),
                    org_block="",
                    memory_context=state.get("memory_context"),
                    briefing=state.get("briefing"),
                    turn_kind="followup",
                    tool_budget=cfg.tool_budget,
                    system_prompt_override=cfg.system_prompt_override,
                )
            )
            try:
                ai = await asyncio.to_thread(
                    llm.invoke,
                    [
                        SystemMessage(content=prompt),
                        HumanMessage(content=f"{body}\n\n{SUMMARIZER_PROMPT_TAIL}"),
                    ],
                )
                text = (ai.content if isinstance(ai.content, str) else str(ai.content)).strip()
            except Exception:  # noqa: BLE001 — LLM-вызов
                logger.warning("summarize: вызов модели упал", exc_info=True)
                # Сводка не собралась — отдаём ответы задач как есть, чтобы
                # работа не пропала.
                text = body

        described = await with_description(
            text, state, config, llm=llm, question=state.get("briefing") or ""
        )
        return {
            "messages": [final_message(described, config)],
            "analytics_answer": text,
            "metrics_summary": state.get("metrics_summary") or text,
        }

    return summarize


__all__ = ["make_plan_tasks_node", "make_summarize_node"]
