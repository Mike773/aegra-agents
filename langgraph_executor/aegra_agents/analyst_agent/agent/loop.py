"""Цикл вызова инструментов — сердце агента.

Модель GigaChat отдаёт один вызов инструмента за сообщение, поэтому «агент» —
это обычный цикл: спросили → выполнили один вызов → вернули результат → снова
спросили. Отдельный подграф не нужен, а гварды получаются обычным кодом, без
подмены функций инструментов.

Цикл живёт ВНУТРИ одного узла графа: aegra запускает графы с дефолтным лимитом
рекурсии, и разворачивать цикл в супершаги значило бы упереться в него.

Финальный текст модели — это и есть ответ руководителю: стадии пересказа нет,
правила формулировок лежат в системном промпте.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .guards import BUDGET_NOTICE, REPEAT_NOTICE, RunGuards, clean_args
from .trace import default_step_text

logger = logging.getLogger(__name__)

# Сколько символов результата инструмента попадает в трассу и шаговое сообщение.
_STEP_SUMMARY_CAP = 280

# Ответ, если модель так и не выдала текст (уперлась в гвард и молчит).
# Руководитель обязан получить сообщение, а не пустоту.
FALLBACK_ANSWER = (
    "Не удалось собрать законченный разбор по этому запросу. "
    "Уточните вопрос — например, назовите конкретный показатель или период."
)


@dataclass
class LoopResult:
    final_text: str = ""
    tool_steps: list[dict[str, Any]] = field(default_factory=list)
    step_texts: list[str] = field(default_factory=list)
    completed: bool = True
    stop_reason: str = "answer"
    llm_calls: int = 0


def _text_of(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content or "")


def _tool_calls_of(message: Any) -> list[dict[str, Any]]:
    calls = getattr(message, "tool_calls", None) or []
    out = []
    for call in calls:
        if isinstance(call, dict) and call.get("name"):
            out.append(call)
    return out


async def _invoke_tool(tool: Any, args: dict[str, Any]) -> str:
    """Инструменты бывают синхронные и асинхронные — вызываем единообразно."""
    if hasattr(tool, "ainvoke"):
        result = await tool.ainvoke(args)
    else:  # pragma: no cover — у StructuredTool ainvoke есть всегда
        result = await asyncio.to_thread(tool.invoke, args)
    return result if isinstance(result, str) else str(result)


def _step_text(tool_name: str, args: dict[str, Any], step_text_fn: Any) -> str:
    if step_text_fn is None:
        return ""
    try:
        return step_text_fn(tool_name, args) or ""
    except Exception:  # noqa: BLE001 — текст прогресса не должен ломать ход
        return ""


async def run_tool_loop(
    *,
    llm: Any,
    tools: list[Any],
    system_prompt: str,
    history: list[Any],
    question: str,
    budget: int | None = None,
    output_cap: int | None = None,
    max_rounds: int | None = None,
    step_text_fn: Any = default_step_text,
    stream_writer: Any = None,
) -> LoopResult:
    """Гоняет модель с инструментами до финального текста без вызовов."""
    guards = RunGuards(
        budget=budget if budget is not None else RunGuards.budget,
        output_cap=output_cap if output_cap is not None else RunGuards.output_cap,
        max_rounds=max_rounds,
    )
    by_name = {t.name: t for t in tools}
    bound = llm.bind_tools(tools) if tools else llm

    messages: list[Any] = [SystemMessage(content=system_prompt), *history]
    if question:
        messages.append(HumanMessage(content=question))

    result = LoopResult()
    reasoning = ""
    rounds = 0

    while True:
        if rounds >= (guards.max_rounds or 0):
            result.completed = False
            result.stop_reason = "rounds"
            break
        rounds += 1
        try:
            ai = await asyncio.to_thread(bound.invoke, messages)
        except Exception:  # noqa: BLE001 — сеть/модель
            logger.warning("agent: вызов модели упал", exc_info=True)
            result.completed = False
            result.stop_reason = "llm_error"
            break
        result.llm_calls += 1
        messages.append(ai)

        calls = _tool_calls_of(ai)
        if not calls:
            result.final_text = _text_of(ai).strip()
            result.stop_reason = "answer"
            break

        # GigaChat отдаёт один вызов за сообщение; если пришло больше, берём первый.
        call = calls[0]
        name = call["name"]
        args = clean_args(call.get("args"))
        text_before = _text_of(ai).strip()
        if text_before:
            reasoning = text_before

        tool = by_name.get(name)
        if tool is None:
            payload = (
                f"Инструмента «{name}» нет. Доступны: {', '.join(by_name) or 'нет'}."
            )
            stop = guards.note_block()
        elif guards.is_repeat(name, args):
            payload = REPEAT_NOTICE
            stop = guards.note_block()
        elif guards.over_budget():
            payload = BUDGET_NOTICE
            stop = guards.note_block()
        else:
            guards.register(name, args)
            try:
                payload = await _invoke_tool(tool, args)
            except Exception as exc:  # noqa: BLE001 — ошибка инструмента не ломает ход
                logger.warning("agent: инструмент %s упал", name, exc_info=True)
                payload = f"Инструмент вернул ошибку: {exc}"
            payload = guards.clip(payload)
            summary = payload[:_STEP_SUMMARY_CAP]
            result.tool_steps.append(
                {
                    "tool": name,
                    "args": args,
                    "result_summary": summary,
                    "reasoning": reasoning,
                }
            )
            text = _step_text(name, args, step_text_fn)
            if text:
                result.step_texts.append(text)
                if stream_writer is not None:
                    try:
                        stream_writer({"step": text})
                    except Exception:  # noqa: BLE001 — стрим не критичен
                        pass
            reasoning = ""
            stop = False

        messages.append(
            ToolMessage(content=payload, tool_call_id=call.get("id") or name)
        )
        if stop:
            result.completed = False
            result.stop_reason = "blocked"
            break

    if not result.final_text:
        # Цикл прервался гвардом — добиваем один вызов без инструментов, чтобы
        # руководитель получил ответ по уже собранным данным.
        result.final_text = await _force_answer(llm, messages, result)
        if result.stop_reason == "answer":
            result.stop_reason = "forced"
        elif guards.over_budget() and result.stop_reason == "blocked":
            result.stop_reason = "budget"
    if not result.final_text:
        # Даже в самом плохом сценарии отдаём осмысленный текст, а не пустоту.
        result.final_text = FALLBACK_ANSWER
        result.completed = False
    return result


async def _force_answer(llm: Any, messages: list[Any], result: LoopResult) -> str:
    """Последний вызов без инструментов: «заверши по собранным данным»."""
    try:
        ai = await asyncio.to_thread(
            llm.invoke, [*messages, HumanMessage(content=BUDGET_NOTICE)]
        )
        result.llm_calls += 1
        return _text_of(ai).strip()
    except Exception:  # noqa: BLE001
        logger.warning("agent: добивающий вызов упал", exc_info=True)
        return ""


__all__ = ["LoopResult", "run_tool_loop"]
