"""Подготовка хода: база, обогащение промпта, карта отклонений.

Сборка базы и все SQL синхронны, поэтому уводятся в поток. На первом ходе
считаются и кладутся в стейт готовые текстовые блоки промпта — на последующих
ходах их не пересчитываем, меняется только карта отклонений.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from ..config import RunConfig
from ..contract.messages import final_message, step_update
from ..db import analytics, core, enrich
from ..deviations import builder, tasks as tasks_mod
from ..prompts import LOAD_ERROR_PROMPT
from .load import metrics_obj

logger = logging.getLogger(__name__)


def build_run_state(
    metrics: Any, aggregates: Any, person_key: str
) -> tuple[Any, dict[str, str]]:
    """Собирает базу и текстовые блоки промпта. Синхронно — зовётся в потоке."""
    db = core.build_run_db(metrics, aggregates)
    analytics.compute_analytics(db.conn)
    blocks = {
        "schema_doc": core.schema_doc(db),
        "enrichment_block": enrich.enrichment_block(db, person_key=person_key),
        "catalog_block": enrich.catalog_block(db),
    }
    return db, blocks


def focus_person_key(db: Any, employee_tabnum: str | None) -> str:
    """Кого разбираем: сотрудник из configurable, иначе первый не-руководитель."""
    if employee_tabnum:
        resolved = db.resolve_person(employee_tabnum)
        if resolved:
            return resolved
    row = db.conn.execute(
        "SELECT person_key FROM person ORDER BY is_me, person_id LIMIT 1"
    ).fetchone()
    return row["person_key"] if row else ""


def make_prepare_node():
    async def prepare(state: Any, config: RunnableConfig) -> dict:
        cfg = RunConfig.from_config(config)
        metrics = metrics_obj(state.get("metrics")) or state.get("metrics")
        if state.get("metrics_error") or not metrics:
            # Данных нет — честно говорим об этом и заканчиваем ход.
            return {
                "messages": [final_message(LOAD_ERROR_PROMPT, config)],
                "reasoning_trace": [
                    {
                        "stage": "prepare",
                        "kind": "error",
                        "summary": f"Данные не загружены: {state.get('metrics_error')}",
                        "detail": {},
                    }
                ],
            }

        try:
            db, blocks = await asyncio.to_thread(
                build_run_state,
                metrics,
                state.get("aggregates"),
                str(state.get("employee_tabnum") or ""),
            )
        except Exception as exc:  # noqa: BLE001 — форма датасета не гарантирована
            logger.warning("prepare: не удалось собрать базу", exc_info=True)
            return {
                "metrics_error": f"Не удалось разобрать датасет: {exc}"[:300],
                "messages": [final_message(LOAD_ERROR_PROMPT, config)],
            }

        person_key = focus_person_key(db, state.get("employee_tabnum"))
        # Заново пересчитываем блоки под реального фокус-человека.
        blocks["enrichment_block"] = await asyncio.to_thread(
            enrich.enrichment_block, db, person_key=person_key
        )

        # Задачи из брифинга: без dashboard_mode они всё равно смещают фокус
        # карты отклонений на то, что просил руководитель.
        parsed_tasks = tasks_mod.parse_tasks(state.get("briefing"))[: cfg.max_tasks]
        resolved_tasks = (
            await asyncio.to_thread(tasks_mod.resolve_task_metrics, db, parsed_tasks)
            if parsed_tasks
            else []
        )
        deviations = await asyncio.to_thread(
            builder.build_deviations,
            db,
            focus_person_key=person_key,
            tasks=resolved_tasks,
            previous=state.get("deviations") or [],
            turn=int(state.get("turn_no") or 1),
        )

        n_metrics = db.conn.execute("SELECT COUNT(*) FROM metric").fetchone()[0]
        n_periods = db.conn.execute("SELECT COUNT(*) FROM period").fetchone()[0]
        step = step_update(
            config,
            f"📊 Загрузил показатели: {n_metrics} шт., периодов {n_periods}, "
            f"в карте отклонений {len(deviations)}.",
        )
        return {
            **step,
            **blocks,
            "tasks": resolved_tasks,
            "task_idx": 0,
            "task_results": [],
            "deviations": deviations,
            "turn_kind": "dashboard" if cfg.dashboard_mode else "initial",
            "reasoning_trace": (state.get("reasoning_trace") or [])
            + [
                {
                    "stage": "prepare",
                    "kind": "decision",
                    "summary": (
                        f"Собрал базу: показателей {n_metrics}, периодов {n_periods}, "
                        f"отклонений {len(deviations)}."
                    ),
                    "detail": {"person_key": person_key},
                }
            ],
        }

    return prepare


def make_begin_turn_node():
    """Ход 2+: сбрасываем то, что живёт один ход. Без LLM и без роутера."""

    async def begin_turn(state: Any, config: RunnableConfig) -> dict:
        return {
            "reasoning_trace": [],
            "easyrag_query": None,
            "easyrag_snippets": [],
            "easyrag_stub_pages": [],
            "easyrag_error": None,
            "analytics_error": None,
            "turn_kind": "followup",
            "turn_no": int(state.get("turn_no") or 1) + 1,
        }

    return begin_turn


__all__ = ["build_run_state", "focus_person_key", "make_begin_turn_node", "make_prepare_node"]
