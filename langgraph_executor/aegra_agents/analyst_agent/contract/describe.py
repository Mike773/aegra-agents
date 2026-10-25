"""Раздел «Как я пришёл к выводу» (describe_answer).

Нумерованные шаги в конце ответа, без LLM — только то, что реально было:

1. Структура — кто в данных: руководитель, сотрудники, позиция.
2. Данные — сколько показателей загружено.
3. Агрегаты групп сравнения — какие уровни пришли.
4. Долгосрочная память.
5… Каждый вызов инструмента за ход: имя и причина вызова (та же фраза, что
   в шаговом сообщении; у query_sql — цель запроса). Результаты не выводятся.

Структура, данные и агрегаты считаются из персистентного стейта (сырой
датасет и агрегаты), инструменты — из трассы хода. Прежняя проза от LLM
убрана: она пересказывала ответ и могла досочинить шаги, а вызов модели
удлинял ход.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.runnables import RunnableConfig

from ..agent.trace import default_step_text
from ..db import loader
from .messages import TRACE_SECTION_TITLE, config_flag

# Сколько имён перечислять в строке, дальше — «и ещё N».
_NAMES_CAP = 10
_MEMORY_PREVIEW = 200
_RESULT_CAP = 200


def _obj(raw: Any) -> Any:
    """Датасет и агрегаты в стейте бывают JSON-строкой."""
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw


def _names(names: list[str]) -> str:
    shown = names[:_NAMES_CAP]
    tail = f" и ещё {len(names) - len(shown)}" if len(names) > len(shown) else ""
    return ", ".join(shown) + tail


def _clip(text: Any, cap: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= cap else text[: cap - 1] + "…"


def _structure_step(metrics: Any, state: Any) -> str:
    if not isinstance(metrics, dict):
        err = state.get("metrics_error")
        return "Структура: данные не загружены" + (f" ({err})." if err else ".")
    parts: list[str] = []
    me = metrics.get("me")
    boss = str((me or {}).get("fio") or "").strip() if isinstance(me, dict) else ""
    if boss:
        parts.append(f"руководитель — {boss}")
    staff = [
        str(e.get("fio") or e.get("tabnum") or "").strip()
        for e in (metrics.get("employees") or [])
        if isinstance(e, dict)
    ]
    staff = [s for s in staff if s]
    if staff:
        parts.append(f"сотрудников — {len(staff)}: {_names(staff)}")
    position = str(state.get("position") or "").strip()
    if position:
        parts.append(f"позиция — {position}")
    if not parts:
        return "Структура: в датасете нет ни руководителя, ни сотрудников."
    return "Структура: " + "; ".join(parts) + "."


def _data_step(metrics: Any) -> str:
    if not isinstance(metrics, dict):
        return "Данные: показатели не загружены."
    try:
        parsed = loader.parse_dataset(metrics)
    except Exception:  # noqa: BLE001 — датасет от внешнего клиента
        return "Данные: датасет не разобран."
    if not parsed.facts:
        return "Данные: датасет пуст."
    return f"Данные: загружено показателей — {len(parsed.metrics)}."


def _aggregates_step(state: Any) -> str:
    raw = _obj(state.get("aggregates"))
    rows = loader.parse_aggregates(raw) if raw else []
    if not rows:
        err = state.get("aggregates_error")
        if err:
            return f"Агрегаты групп сравнения: не загружены ({err})."
        return "Агрегаты групп сравнения: не пришли."
    levels: list[str] = []
    for r in rows:
        name = str(r.get("level_name") or r.get("level") or "").strip()
        if name and name not in levels:
            levels.append(name)
    return "Агрегаты групп сравнения: уровни — " + ", ".join(levels) + "."


def _memory_step(state: Any) -> str:
    if state.get("memory_error"):
        return "Долгосрочная память: не удалось загрузить."
    ctx = (state.get("memory_context") or "").strip()
    if ctx and "отсутствует" not in ctx.lower():
        return f"Долгосрочная память: {_clip(ctx, _MEMORY_PREVIEW)}"
    return "Долгосрочная память: сохранённого контекста прошлых диалогов нет."


def _tool_step(step: dict[str, Any]) -> str:
    """Инструмент и причина вызова — та же фраза, что в шаговом сообщении."""
    detail = step.get("detail") or {}
    tool = detail.get("tool")
    if not tool:
        # Шаг без деталей (старый формат трассы) — как есть.
        return _clip(step.get("summary"), _RESULT_CAP)
    reason = default_step_text(tool, detail.get("args") or {})
    # Эмодзи в начале фразы — для ленты шагов, в описи он лишний.
    head, _, rest = reason.partition(" ")
    reason = rest if rest and not head[:1].isalnum() else reason
    if reason == tool:
        return f"Инструмент `{tool}`."
    return f"Инструмент `{tool}`: {reason[:1].lower()}{reason[1:]}."


def describe_steps(state: Any) -> list[str]:
    """Шаги раздела по порядку: структура, данные, агрегаты, память, инструменты."""
    metrics = _obj(state.get("metrics"))
    steps = [
        _structure_step(metrics, state),
        _data_step(metrics),
        _aggregates_step(state),
        _memory_step(state),
    ]
    for step in state.get("reasoning_trace") or []:
        if step.get("kind") == "tool_call":
            steps.append(_tool_step(step))
        elif step.get("kind") == "error" and (step.get("summary") or "").strip():
            steps.append(f"Сбой: {_clip(step['summary'], _RESULT_CAP)}")
    return steps


def render_description(state: Any) -> str:
    lines = ["---", TRACE_SECTION_TITLE, ""]
    lines += [f"{i}. {text}" for i, text in enumerate(describe_steps(state), 1)]
    return "\n".join(lines)


def with_description(text: str, state: Any, config: RunnableConfig | None) -> str:
    """Дописывает в конец ответа раздел «Как я пришёл к выводу», если
    describe_answer=true."""
    if not config_flag(config, "describe_answer", default=False):
        return text
    return f"{text}\n\n{render_description(state)}"


__all__ = ["describe_steps", "render_description", "with_description"]
