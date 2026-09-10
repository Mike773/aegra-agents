"""Запись главного вывода в сервис инсайтов.

Вывод берётся с вершины карты отклонений детерминированно — отдельного
LLM-классификатора больше нет. Пишется один раз за диалог и только когда
известны и тип, и идентификатор источника. Любая ошибка изолируется: инсайт —
побочный эффект, руководитель свой ответ уже получил.

Сигнальный режим (``run_mode=signal``): инсайт уходит с полями ``author``,
``confirmed``, ``signal`` и ``signal_description``. Вердикт ``signal`` — единственный
LLM-вызов узла, и только в этом режиме; если карта отклонений пуста, всё равно
шлём запись «в норме» с ``signal=false``. Без ответа агента (данные не
загрузились) сигналить нечем — запись не пишем.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from ...shared.assignments_service import SendAssignmentsComponent
from ...shared.insight_signal import detect_signal, empty_norm_insight, signal_insight
from ...shared.offload import run_blocking
from ..config import RunConfig
from ..contract.messages import last_user_text
from ..deviations.insight import TYPE_MAIN_PROBLEM, pick_main_insight

logger = logging.getLogger(__name__)


def _skip(state: Any, summary: str) -> dict:
    return {
        "reasoning_trace": (state.get("reasoning_trace") or [])
        + [
            {
                "stage": "assignments",
                "kind": "decision",
                "summary": summary,
                "detail": {},
            }
        ]
    }


def make_auto_insight_node(llm: Any):
    async def auto_insight(state: Any, config: RunnableConfig) -> dict:
        cfg = RunConfig.from_config(config)
        if state.get("insight_written"):
            return {}
        if not cfg.has_source():
            return _skip(state, "Инсайт не записан: нет source_type/source_id.")

        signal_mode = cfg.is_signal_mode()
        insight = pick_main_insight(state.get("deviations") or [])
        if insight is None:
            if not signal_mode:
                return _skip(state, "Инсайт не записан: отклонений в карте нет.")
            insight = empty_norm_insight()

        detail: dict[str, Any] = {}
        if signal_mode:
            answer = (state.get("analytics_answer") or "").strip()
            if not answer:
                return _skip(
                    state, "Инсайт не записан: ответа нет (данные не загружены)."
                )
            question = (
                state.get("analytics_question")
                or last_user_text(state)
                or state.get("briefing")
                or ""
            )
            signal = await detect_signal(
                llm, question, answer, fallback=insight["type"] == TYPE_MAIN_PROBLEM
            )
            insight = signal_insight(insight, signal=signal, description=answer)
            detail = {"run_mode": cfg.run_mode, "signal": signal}

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

        name = insight.get("metric_name") or "—"
        summary = (
            f"Записал сигнал (signal={'true' if insight['signal'] else 'false'}) "
            f"по метрике «{name}» ({insight['type']})."
            if signal_mode
            else f"Записал главный вывод «{name}» ({insight['type']})."
        )
        return {
            "insight_written": True,
            "reasoning_trace": (state.get("reasoning_trace") or [])
            + [
                {
                    "stage": "assignments",
                    "kind": "decision",
                    "summary": summary,
                    "detail": {**detail, "insight": insight},
                }
            ],
        }

    return auto_insight


__all__ = ["make_auto_insight_node"]
