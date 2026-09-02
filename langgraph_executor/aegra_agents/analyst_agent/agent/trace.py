"""Шаговые сообщения и трасса рассуждения из вызовов инструментов.

Тексты шагов пишутся на человеческом языке: их видит руководитель, пока идёт
анализ. Имена инструментов и полей туда не попадают.
"""
from __future__ import annotations

from typing import Any

_ARG_CAP = 60
_PURPOSE_CAP = 80


def _short(value: Any) -> str:
    text = str(value or "").strip()
    return text if len(text) <= _ARG_CAP else text[: _ARG_CAP - 1] + "…"


def default_step_text(tool_name: str, args: dict[str, Any]) -> str:
    """Короткая фраза о том, что агент сейчас сделал."""
    args = args or {}
    metric = _short(args.get("metric"))
    if tool_name == "query_sql":
        purpose = str(args.get("purpose") or "").strip()[:_PURPOSE_CAP]
        return f"📊 {purpose}" if purpose else "📊 Посчитал данные запросом"
    if tool_name == "metric_card":
        return f"📈 Разобрал показатель «{metric}»" if metric else "📈 Разобрал показатель"
    if tool_name == "peer_context":
        return f"👥 Сравнил «{metric}» с коллегами" if metric else "👥 Сравнил с коллегами"
    if tool_name == "search_wiki":
        query = _short(args.get("query"))
        return f"📖 Уточнил в базе знаний: {query}" if query else "📖 Заглянул в базу знаний"
    if tool_name == "list_deviations":
        return "🗂 Просмотрел карту отклонений"
    if tool_name == "note_deviation":
        return f"📝 Отметил находку по «{metric}»" if metric else "📝 Отметил находку"
    return f"⚙️ {tool_name}"


def trace_steps(tool_steps: list[dict[str, Any]], *, stage: str = "agent") -> list[dict]:
    """Вызовы инструментов → шаги трассы (Блок A ТЗ, формат v4)."""
    out: list[dict] = []
    for step in tool_steps or []:
        args = step.get("args") or {}
        arg_text = ", ".join(f"{k}={_short(v)}" for k, v in args.items() if v is not None)
        summary = f"{step['tool']}({arg_text}) → {step.get('result_summary', '')}"
        out.append(
            {
                "stage": stage,
                "kind": "tool_call",
                "summary": summary.strip(),
                "detail": step,
            }
        )
    return out


__all__ = ["default_step_text", "trace_steps"]
