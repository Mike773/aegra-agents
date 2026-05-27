"""Общие части стратегии-агента: синтез (стадия 2) и форматирование.

Стадия 1 (классическая стратегия) — в agent_classic.py: tool-loop с защитой
от зацикливания. Стадия 2 — синтез финального ответа: один вызов чат-модели
БЕЗ инструментов с собранными tool-результатами в виде текста. Без функций
лимит запроса GigaChat в 4096 токенов не действует — в финальный ответ
вкладывается весь транскрипт.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from .prompts import SYNTHESIS_PROMPT


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
        periods += f" (первая неделя = {dates[0]}, последняя/«сейчас» = {dates[-1]})"
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


def extract_tool_transcript(messages: list[Any]) -> tuple[str, int]:
    """Собирает из сообщений стадии 1 транскрипт «вызов инструмента -> результат»."""
    results: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            results[msg.tool_call_id] = _text(msg)

    blocks: list[str] = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            args = ", ".join(
                f"{k}={v!r}" for k, v in (call.get("args") or {}).items()
            )
            result = results.get(call.get("id"), "(результат отсутствует)")
            blocks.append(f"{len(blocks) + 1}. {call.get('name')}({args}) ->\n{result}")
    return "\n\n".join(blocks), len(blocks)


def synthesize_answer(model: Any, question: str, messages: list[Any]) -> str:
    """Стадия 2: финальный ответ из собранных данных вызовом модели без инструментов."""
    transcript, tool_calls = extract_tool_transcript(messages)
    if tool_calls == 0:
        # Инструменты не вызывались — стадия 1 уже дала прямой ответ или отказ.
        return _text(messages[-1]) if messages else ""
    user_content = (
        f"Вопрос пользователя: {question}\n\n"
        f"Данные, собранные инструментами из базы:\n{transcript}"
    )
    response = model.invoke(
        [SystemMessage(content=SYNTHESIS_PROMPT), HumanMessage(content=user_content)]
    )
    return _text(response)
