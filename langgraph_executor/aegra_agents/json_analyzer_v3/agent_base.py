"""Общие части стратегии-агента: форматирование фактов и сбор транскрипта.

Стадия 1 (классическая стратегия) — в agent_classic.py: tool-loop с защитой
от зацикливания. Стадия 2 (синтез финального ответа) — в nodes.make_synthesize_node:
один вызов чат-модели БЕЗ инструментов с собранным транскриптом. Здесь — только
извлечение фактов/транскрипта из сообщений стадии 1.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage


def _text(msg: Any) -> str:
    """Извлекает текст из сообщения (content — строка или список блоков)."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def _fmt_metric_entry(m: dict[str, Any]) -> str:
    """Метрика в формате '<name> [agg+|agg-: A, B, ...]'.

    Тег показывает, есть ли у метрики агрегатная строка; список — её
    element-значения.
    """
    name = m["metric_name"]
    tag = "agg+" if m.get("has_aggregate", True) else "agg-"
    elems = m.get("elements") or []
    return f"{name} [{tag}: {', '.join(elems)}]" if elems else f"{name} [{tag}]"


def format_facts(overview: dict[str, Any]) -> str:
    """Компактная сводка состава датасета для системного промпта."""
    dates = overview.get("dates") or []
    metric_entries = sorted(
        {_fmt_metric_entry(m) for m in overview.get("metrics", [])}
    )
    people = overview.get("people") or []
    posts = sorted({p["person_post"] for p in people if p.get("person_post")})
    departs = sorted({p["person_depart"] for p in people if p.get("person_depart")})
    elements = overview.get("elements") or []
    managers = sum(1 for p in people if p.get("person_is_me"))

    if len(people) == 1:
        only = people[0]
        post = only.get("person_post") or "должность не указана"
        people_line = (
            f"- В датасете ОДИН человек: {only['person_fio']} ({post}). Любой "
            "вопрос про «этого сотрудника», «него», «её», «оператора» относится "
            f"к нему — сразу подставляй ФИО '{only['person_fio']}' в аргумент "
            "person. НЕ переспрашивай у пользователя имя."
        )
    else:
        people_line = (
            f"- Людей: {len(people)} ({managers} рук. + {len(people) - managers} "
            "сотр.). Человека по неточному имени ищи через resolve_entity или "
            "list_people."
        )

    periods = ", ".join(dates)
    if dates:
        periods += f" (первый период = {dates[0]}, последний/«сейчас» = {dates[-1]})"
    lines = [
        "СОСТАВ ЗАГРУЖЕННОГО ДАТАСЕТА (используй эти точные значения в аргументах):",
        f"- Периоды по порядку: {periods}",
        f"- Метрики: {'; '.join(metric_entries)}",
        "  Пометки: agg+ — у метрики есть агрегатная строка (element IS NULL); "
        "agg- — только разрезы. Для agg- запрос без element вернёт все разрезы "
        "с пометкой 'разрезы_вместо_агрегата'; для rank по agg- метрике "
        "ОБЯЗАТЕЛЬНО указывай element.",
        f"- Должности: {', '.join(posts)}",
        f"- Подразделения: {', '.join(departs)}",
        f"- Значения element (продукты/разрезы): {', '.join(elements)}",
        people_line,
    ]
    return "\n".join(lines)


def _collect_tool_calls(
    messages: list[Any],
) -> list[tuple[str, dict[str, Any], str | None, str]]:
    """Сводит сообщения стадии 1 к (имя, аргументы, текст-результат, мотивация).

    Единый источник для транскрипта и структурированных шагов трассы: результаты
    берутся из ToolMessage по tool_call_id, имена/аргументы — из tool_calls.
    Третий элемент — None, если по id результата нет (отличаем «нет» от «пусто»).
    Четвёртый — текст AIMessage, в котором лежит tool_call (рассуждение модели
    «зачем зову»; GigaChat пишет его не всегда — тогда пустая строка).
    """
    results: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            results[msg.tool_call_id] = _text(msg)

    calls: list[tuple[str, dict[str, Any], str | None, str]] = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            continue
        reasoning = " ".join(_text(msg).split())
        for call in tool_calls:
            args = call.get("args") or {}
            calls.append((call.get("name"), args, results.get(call.get("id")), reasoning))
    return calls


def extract_tool_transcript(messages: list[Any]) -> tuple[str, int]:
    """Собирает из сообщений стадии 1 транскрипт «вызов инструмента -> результат»."""
    blocks: list[str] = []
    for name, args, result, _reasoning in _collect_tool_calls(messages):
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        text = result if result is not None else "(результат отсутствует)"
        blocks.append(f"{len(blocks) + 1}. {name}({arg_str}) ->\n{text}")
    return "\n\n".join(blocks), len(blocks)


_STEP_SUMMARY_CAP = 280
_STEP_REASONING_CAP = 200


def extract_tool_steps(messages: list[Any]) -> list[dict[str, Any]]:
    """Структурированные шаги tool-loop для сквозной трассы (Блок A.4 ТЗ).

    Тот же источник, что у extract_tool_transcript (_collect_tool_calls), но на
    выходе — список {"tool", "args", "result_summary"[, "reasoning"]} со сжатой
    выжимкой результата (полные выдачи в трассу не тянем — экономим контекст).
    reasoning — текст рассуждения модели рядом с tool_call, если она его писала.
    """
    steps: list[dict[str, Any]] = []
    for name, args, result, reasoning in _collect_tool_calls(messages):
        summary = " ".join((result or "").split())
        if len(summary) > _STEP_SUMMARY_CAP:
            summary = summary[:_STEP_SUMMARY_CAP] + "…"
        step: dict[str, Any] = {
            "tool": name,
            "args": {k: v for k, v in args.items() if v not in (None, "")},
            "result_summary": summary,
        }
        if reasoning:
            if len(reasoning) > _STEP_REASONING_CAP:
                reasoning = reasoning[:_STEP_REASONING_CAP] + "…"
            step["reasoning"] = reasoning
        steps.append(step)
    return steps
