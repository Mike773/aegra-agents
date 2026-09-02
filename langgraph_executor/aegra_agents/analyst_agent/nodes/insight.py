"""Запись главного вывода в сервис инсайтов.

Вывод берётся с вершины карты отклонений детерминированно — отдельного
LLM-классификатора больше нет. Пишется один раз за диалог и только когда
известны и тип, и идентификатор источника. Любая ошибка изолируется: инсайт —
побочный эффект, руководитель свой ответ уже получил.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from ...shared.assignments_service import SendAssignmentsComponent
from ...shared.offload import run_blocking
from ..config import RunConfig
from ..deviations.insight import pick_main_insight

logger = logging.getLogger(__name__)


def make_auto_insight_node():
    async def auto_insight(state: Any, config: RunnableConfig) -> dict:
        cfg = RunConfig.from_config(config)
        if state.get("insight_written"):
            return {}
        if not cfg.has_source():
            return {
                "reasoning_trace": (state.get("reasoning_trace") or [])
                + [
                    {
                        "stage": "assignments",
                        "kind": "decision",
                        "summary": "Инсайт не записан: нет source_type/source_id.",
                        "detail": {},
                    }
                ]
            }

        insight = pick_main_insight(state.get("deviations") or [])
        if insight is None:
            return {
                "reasoning_trace": (state.get("reasoning_trace") or [])
                + [
                    {
                        "stage": "assignments",
                        "kind": "decision",
                        "summary": "Инсайт не записан: отклонений в карте нет.",
                        "detail": {},
                    }
                ]
            }

        try:
            await run_blocking(
                lambda: SendAssignmentsComponent(
                    boss_tabnum=state.get("boss_tabnum") or "",
                    employee_tabnum=state.get("employee_tabnum") or "",
                    direction_key=state.get("direction_key") or "",
                    thread_id=cfg.thread_id,
                    insight=insight,
                    source_type=cfg.source_type,
                    source_id=cfg.source_id,
                ).submit()
            )
        except Exception as exc:  # noqa: BLE001 — внешний сервис
            logger.warning("auto_insight: не удалось записать инсайт", exc_info=True)
            return {
                "reasoning_trace": (state.get("reasoning_trace") or [])
                + [
                    {
                        "stage": "assignments",
                        "kind": "error",
                        "summary": f"Инсайт не записан: {type(exc).__name__}.",
                        "detail": {"error": str(exc)[:300]},
                    }
                ]
            }

        return {
            "insight_written": True,
            "reasoning_trace": (state.get("reasoning_trace") or [])
            + [
                {
                    "stage": "assignments",
                    "kind": "decision",
                    "summary": (
                        f"Записал главный вывод «{insight['metric_name']}» "
                        f"({insight['type']})."
                    ),
                    "detail": {"insight": insight},
                }
            ],
        }

    return auto_insight


__all__ = ["make_auto_insight_node"]
