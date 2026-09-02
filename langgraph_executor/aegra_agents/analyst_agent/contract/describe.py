"""Раздел «Как я пришёл к выводу» (describe_answer) — перенос из v4.

Раздел открывается детерминированной описью источников (оргструктура, метрики,
wiki, память — из персистентного стейта, а не из трассы: трасса живёт один ход).
Дальше — рассказ о ходе анализа: основной путь LLM по трассе, фоллбэк —
детерминированный список шагов, поэтому режим не может сломать ответ.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..db import loader
from ..prompts.describe import DESCRIBE_ANSWER_PROMPT
from .messages import TRACE_SECTION_TITLE, config_flag

logger = logging.getLogger(__name__)

KIND_LABELS = {
    "intent": "Классификация запроса",
    "kb_hit": "База знаний (wiki)",
    "tool_call": "Метрики",
    "derived_metric": "Производный показатель",
    "hypothesis": "Гипотеза",
    "decision": "Вывод",
    "error": "Сбой",
}

# Превью памяти и запроса к wiki в описи источников: там нужна опись, не текст.
_SOURCES_MEMORY_PREVIEW = 200
_SOURCES_QUERY_PREVIEW = 120


def render_trace(trace: list[dict[str, Any]], preamble: str = "") -> str:
    """Детерминированный markdown-раздел из трассы — без LLM, без галлюцинаций."""
    steps = [s for s in (trace or []) if (s.get("summary") or "").strip()]
    if not steps and not preamble:
        return ""
    lines = ["---", TRACE_SECTION_TITLE, ""]
    if preamble:
        lines.append(preamble)
        lines.append("")
    for i, step in enumerate(steps, 1):
        label = KIND_LABELS.get(step.get("kind", ""), step.get("stage") or "Шаг")
        lines.append(f"{i}. **{label}.** {step['summary'].strip()}")
    return "\n".join(lines).rstrip()


def _trace_for_description(trace: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, step in enumerate(trace or [], 1):
        summary = (step.get("summary") or "").strip()
        if not summary:
            continue
        label = KIND_LABELS.get(step.get("kind", ""), step.get("stage") or "шаг")
        lines.append(f"{i}. [{label}] {summary}")
        if step.get("kind") == "tool_call":
            reasoning = ((step.get("detail") or {}).get("reasoning") or "").strip()
            if reasoning:
                lines.append(f"   Мотивация вызова: {reasoning}")
    return "\n".join(lines)


async def describe_trace_llm(
    llm: Any, question: str, answer: str, trace: list[dict[str, Any]]
) -> str:
    """Рассказ о ходе анализа по трассе. Любая ошибка → "" (будет фоллбэк)."""
    steps_text = _trace_for_description(trace)
    if not steps_text:
        return ""
    parts = []
    if (question or "").strip():
        parts.append(f"Вопрос руководителя:\n{question.strip()}")
    parts.append(f"Итоговый ответ:\n{answer}")
    parts.append(f"Трасса (шаги анализа по порядку):\n{steps_text}")
    try:
        ai = await asyncio.to_thread(
            llm.invoke,
            [
                SystemMessage(content=DESCRIBE_ANSWER_PROMPT),
                HumanMessage(content="\n\n".join(parts)),
            ],
        )
        content = ai.content if isinstance(ai.content, str) else str(ai.content)
        return content.strip()
    except Exception:  # noqa: BLE001 — LLM-вызов, сужать нечем
        logger.warning(
            "describe_answer: LLM-описание трассы упало, фоллбэк на список", exc_info=True
        )
        return ""


def _metrics_obj(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        import json

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _metrics_source_line(metrics: Any) -> str:
    """«- Метрики: …»: уникальные показатели, верхнеуровневые имена, периоды."""
    try:
        parsed = loader.parse_dataset(metrics)
    except Exception:  # noqa: BLE001 — датасет от внешнего клиента
        return "- Метрики: датасет не разобран."
    if not parsed.facts:
        return "- Метрики: датасет пуст."
    children = {child for _, child in parsed.edges}
    top_names = [rec.name for norm, rec in parsed.metrics.items() if norm not in children]
    periods: dict[str, set[str]] = {}
    for f in parsed.facts:
        if f.date:
            periods.setdefault(f.calc_period or "период", set()).add(f.date)
    parts = [f"всего уникальных метрик — {len(parsed.metrics)}"]
    if top_names:
        parts.append("верхнеуровневые: " + ", ".join(top_names))
    if periods:
        per_strs = []
        for period, dates in sorted(periods.items()):
            ds = sorted(dates)
            span = ds[0] if len(ds) == 1 else f"{ds[0]} … {ds[-1]}"
            per_strs.append(f"{period} — {len(ds)} ({span})")
        parts.append("периоды: " + "; ".join(per_strs))
    return "- Метрики: " + "; ".join(parts) + "."


def _wiki_source_line(state: Any) -> str:
    snippets = state.get("easyrag_snippets") or []
    if snippets:
        refs = []
        for s in snippets[:5]:
            page = s.get("page_title") or s.get("slug") or "-"
            title = s.get("section_title") or s.get("anchor") or "-"
            refs.append(f"«{page} / {title}»")
        return f"- Wiki: найдено фрагментов — {len(snippets)}: " + ", ".join(refs) + "."
    if state.get("easyrag_error"):
        return "- Wiki: не удалось получить данные из базы знаний."
    query = (state.get("easyrag_query") or "").strip()
    if not query:
        return "- Wiki: не запрашивалась на этом ходе."
    if len(query) > _SOURCES_QUERY_PREVIEW:
        query = query[:_SOURCES_QUERY_PREVIEW] + "…"
    stubs = state.get("easyrag_stub_pages") or []
    if stubs:
        names = ", ".join(s.get("title") or s.get("slug") or "-" for s in stubs)
        return (
            f"- Wiki: по запросу «{query}» найдены только пустые "
            f"страницы-заглушки: {names}."
        )
    return f"- Wiki: по запросу «{query}» ничего не найдено."


def _memory_source_line(state: Any) -> str:
    if state.get("memory_error"):
        return "- Долгосрочная память: не удалось загрузить."
    ctx = (state.get("memory_context") or "").strip()
    if ctx and "отсутствует" not in ctx.lower():
        preview = ctx.replace("\n", " ")
        if len(preview) > _SOURCES_MEMORY_PREVIEW:
            preview = preview[:_SOURCES_MEMORY_PREVIEW] + "…"
        return f"- Долгосрочная память: {preview}"
    return (
        "- Долгосрочная память: отсутствует "
        "(сохранённого контекста прошлых диалогов нет)."
    )


def describe_sources_block(state: Any) -> str:
    """Детерминированная опись источников — точные имена и числа, без LLM."""
    lines: list[str] = []
    metrics = _metrics_obj(state.get("metrics"))
    if metrics is not None:
        org_parts: list[str] = []
        me = metrics.get("me") or {}
        boss_fio = str((me.get("fio") if isinstance(me, dict) else "") or "").strip()
        if boss_fio:
            org_parts.append(f"руководитель — {boss_fio}")
        names = [
            str(e.get("fio") or "").strip()
            for e in (metrics.get("employees") or [])
            if isinstance(e, dict) and str(e.get("fio") or "").strip()
        ]
        if names:
            position = str(state.get("position") or "").strip()
            pos = f" (позиция: {position})" if position else ""
            org_parts.append("в фокусе анализа — " + ", ".join(names) + pos)
        lines.append(
            "- Оргструктура: " + "; ".join(org_parts) + "."
            if org_parts
            else "- Оргструктура: в датасете нет ФИО."
        )
        lines.append(_metrics_source_line(metrics))
    else:
        err = state.get("metrics_error")
        suffix = f" ({err})" if err else ""
        lines.append(f"- Оргструктура: данные не загружены{suffix}.")
        lines.append("- Метрики: не загружены.")
    lines.append(_wiki_source_line(state))
    lines.append(_memory_source_line(state))
    return "**Исходные данные:**\n" + "\n".join(lines)


async def with_description(
    text: str,
    state: Any,
    config: RunnableConfig | None,
    llm: Any = None,
    question: str = "",
) -> str:
    """Дописывает раздел «Как я пришёл к выводу», если describe_answer=true."""
    if not config_flag(config, "describe_answer", default=False):
        return text
    sources = describe_sources_block(state)
    trace = state.get("reasoning_trace") or []
    substantive = any(s.get("kind") in ("tool_call", "kb_hit") for s in trace)
    if llm is not None and substantive:
        narrative = await describe_trace_llm(llm, question, text, trace)
        if narrative:
            body = f"{sources}\n\n{narrative}" if sources else narrative
            return f"{text}\n\n---\n{TRACE_SECTION_TITLE}\n\n{body}"
    section = render_trace(trace, preamble=sources)
    return f"{text}\n\n{section}" if section else text


__all__ = [
    "KIND_LABELS",
    "describe_sources_block",
    "describe_trace_llm",
    "render_trace",
    "with_description",
]
