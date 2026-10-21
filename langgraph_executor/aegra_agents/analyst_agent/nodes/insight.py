"""Запись главного вывода в сервис инсайтов.

Вывод берётся из ответа агента: фокус выбирает модель, по алгоритму из первого
запроса руководителя, и инсайт должен говорить о том же. Один вызов модели с
обязательным инструментом называет тип, показатель и суть вывода; показатель
сверяется с каталогом, а его идентификатор, факт и план код берёт из данных —
числа со слов модели в сервис не уходят. Пишется один раз за диалог и только
когда известны и тип, и идентификатор источника. Любая ошибка изолируется:
инсайт — побочный эффект, руководитель свой ответ уже получил.

Сигнальный режим (``run_mode=signal``): инсайт уходит с полями ``author``,
``confirmed``, ``signal`` и ``signal_description``; вердикт ``signal`` — второй
вызов модели, только в этом режиме. Если главный вывод выделить не удалось,
всё равно шлём запись «в норме» с ``signal=false``. Без ответа агента (данные
не загрузились) сигналить нечем — запись не пишем.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ...shared.assignments_service import SendAssignmentsComponent
from ...shared.insight_signal import detect_signal, empty_norm_insight, signal_insight
from ...shared.offload import run_blocking
from ..config import RunConfig
from ..contract.messages import last_user_text
from ..db import core
from ..prompts.insight import (
    MAIN_INSIGHT_PROMPT,
    MAIN_INSIGHT_SCHEMA,
    MAIN_INSIGHT_TOOL,
    MAIN_INSIGHT_TOOL_DESCRIPTION,
)
from .prepare import focus_person_key

logger = logging.getLogger(__name__)

# Типы, которые понимает сервис инсайтов (контракт v4).
TYPE_MAIN_PROBLEM = "main_problem"
_TYPES = ("main_problem", "achievement", "norm")


async def _ask_main_insight(llm: Any, answer: str) -> dict[str, str] | None:
    """Ответ агента → {type, metric, text} от модели. Любой сбой → None."""
    spec = {
        "type": "function",
        "function": {
            "name": MAIN_INSIGHT_TOOL,
            "description": MAIN_INSIGHT_TOOL_DESCRIPTION,
            "parameters": MAIN_INSIGHT_SCHEMA,
        },
    }
    try:
        bound = llm.bind_tools([spec], tool_choice=MAIN_INSIGHT_TOOL)
        ai = await asyncio.to_thread(
            bound.invoke,
            [SystemMessage(content=MAIN_INSIGHT_PROMPT), HumanMessage(content=answer)],
        )
    except Exception:  # noqa: BLE001 — LLM-вызов, сужать нечем
        logger.warning("auto_insight: главный вывод не получен", exc_info=True)
        return None
    for call in getattr(ai, "tool_calls", None) or []:
        if call.get("name") != MAIN_INSIGHT_TOOL:
            continue
        args = call.get("args") or {}
        kind = str(args.get("type") or "").strip()
        text = str(args.get("text") or "").strip()
        if kind in _TYPES and text:
            return {"type": kind, "metric": str(args.get("metric") or "").strip(), "text": text}
    logger.warning("auto_insight: модель не вернула главный вывод")
    return None


def _resolve_metric(db: Any, name: str, text: str) -> Any:
    """Название от модели → показатель каталога; не нашли — ищем имя в тексте вывода."""
    ref = db.resolve_metric(name) if name else None
    if ref is not None:
        return ref
    low = (text or "").casefold()
    best = None
    for row in db.conn.execute("SELECT metric_id, name, path FROM metric"):
        if row["name"] and row["name"].casefold() in low:
            if best is None or len(row["name"]) > len(best.name):
                best = core.MetricRef(row["metric_id"], row["name"], row["path"])
    return best


def _build_insight(db: Any, person_key: str, raw: dict[str, str]) -> dict[str, Any]:
    """Вывод модели + данные: id показателя, факт и план — из последнего среза."""
    insight: dict[str, Any] = {
        "type": raw["type"],
        "metric_id": None,
        "metric_name": None,
        "fact": None,
        "plan": None,
        "text": raw["text"],
    }
    ref = _resolve_metric(db, raw["metric"], raw["text"])
    if ref is None:
        return insight
    insight["metric_name"] = ref.name
    row = db.conn.execute(
        "SELECT ext_id, fact, plan FROM v_fact_latest WHERE person_key = ? "
        "AND metric = ? AND path IS ? AND element IS NULL",
        (person_key, ref.name, ref.path),
    ).fetchone()
    if row is not None:
        insight.update(metric_id=row["ext_id"], fact=row["fact"], plan=row["plan"])
    else:
        ext = db.conn.execute(
            "SELECT ext_id FROM metric WHERE metric_id = ?", (ref.metric_id,)
        ).fetchone()
        insight["metric_id"] = ext["ext_id"] if ext is not None else None
    return insight


async def _main_insight(llm: Any, state: Any, answer: str) -> dict[str, Any] | None:
    raw = await _ask_main_insight(llm, answer)
    if raw is None:
        return None

    def _build() -> dict[str, Any]:
        db = core.build_run_db(state.get("metrics"), state.get("aggregates"))
        person_key = focus_person_key(db, state.get("employee_tabnum"))
        return _build_insight(db, person_key, raw)

    return await asyncio.to_thread(_build)


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
        answer = (state.get("analytics_answer") or "").strip()
        if not answer:
            return _skip(state, "Инсайт не записан: ответа нет (данные не загружены).")
        insight = await _main_insight(llm, state, answer)
        if insight is None:
            if not signal_mode:
                return _skip(
                    state, "Инсайт не записан: главный вывод из ответа не выделился."
                )
            insight = empty_norm_insight()

        detail: dict[str, Any] = {}
        if signal_mode:
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
