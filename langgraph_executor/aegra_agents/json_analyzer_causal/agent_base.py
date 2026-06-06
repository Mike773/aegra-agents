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
) -> list[tuple[str, dict[str, Any], str | None]]:
    """Сводит сообщения стадии 1 к (имя, аргументы, текст-результат) по каждому вызову.

    Единый источник для транскрипта и структурированных шагов трассы: результаты
    берутся из ToolMessage по tool_call_id, имена/аргументы — из tool_calls.
    Третий элемент — None, если по id результата нет (отличаем «нет» от «пусто»).
    """
    results: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            results[msg.tool_call_id] = _text(msg)

    calls: list[tuple[str, dict[str, Any], str | None]] = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            args = call.get("args") or {}
            calls.append((call.get("name"), args, results.get(call.get("id"))))
    return calls


# Инструменты с АВТОРИТЕТНОЙ готовой раскладкой: их выдача идёт в синтез первой и
# целиком, чтобы синтез брал числа из неё, а не тонул в шумных дампах get_metric.
_PRIORITY_TOOLS = ("attribute_change", "attribute_anomaly")
# Потолок размера транскрипта, уходящего в синтез (стадия 2). Приоритетные блоки
# включаются целиком всегда; остальные добиваются до бюджета и при переполнении
# усекаются. Защищает синтез от разбавления десятками строк и от перегруза токенов.
_TRANSCRIPT_CAP = 5000


def extract_tool_transcript(messages: list[Any]) -> tuple[str, int]:
    """Собирает транскрипт «вызов -> результат» для синтеза.

    Приоритетные инструменты (attribute_change/anomaly) ставятся первыми и целиком;
    остальные блоки добиваются до _TRANSCRIPT_CAP и при переполнении усекаются —
    так точная раскладка вклада не размывается шумными выдачами, а вход синтеза
    остаётся в рабочем окне модели.
    """
    calls = _collect_tool_calls(messages)
    ordered = [c for c in calls if c[0] in _PRIORITY_TOOLS] + [
        c for c in calls if c[0] not in _PRIORITY_TOOLS
    ]
    blocks: list[str] = []
    used = 0
    for name, args, result in ordered:
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        text = result if result is not None else "(результат отсутствует)"
        block = f"{len(blocks) + 1}. {name}({arg_str}) ->\n{text}"
        if name in _PRIORITY_TOOLS:
            blocks.append(block)  # авторитетные — всегда целиком
            used += len(block)
            continue
        if used >= _TRANSCRIPT_CAP:
            continue  # бюджет исчерпан — остальной шум отбрасываем
        room = _TRANSCRIPT_CAP - used
        if len(block) > room:
            block = block[:room] + "\n…[усечено для синтеза]"
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks), len(calls)


_STEP_SUMMARY_CAP = 280


def extract_tool_steps(messages: list[Any]) -> list[dict[str, Any]]:
    """Структурированные шаги tool-loop для сквозной трассы (Блок A.4 ТЗ).

    Тот же источник, что у extract_tool_transcript (_collect_tool_calls), но на
    выходе — список {"tool", "args", "result_summary"} со сжатой выжимкой
    результата (полные выдачи в трассу не тянем — экономим контекст).
    """
    steps: list[dict[str, Any]] = []
    for name, args, result in _collect_tool_calls(messages):
        summary = " ".join((result or "").split())
        if len(summary) > _STEP_SUMMARY_CAP:
            summary = summary[:_STEP_SUMMARY_CAP] + "…"
        steps.append(
            {
                "tool": name,
                "args": {k: v for k, v in args.items() if v not in (None, "")},
                "result_summary": summary,
            }
        )
    return steps
