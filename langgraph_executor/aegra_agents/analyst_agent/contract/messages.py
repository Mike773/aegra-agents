"""Контракт сообщений хода — перенос из analytic_orchestrator_v4 без изменений.

За один ход граф проходит один линейный путь, а ``add_messages`` добавляет в
конец, поэтому порядок сообщений = порядок отработки узлов. Рабочие узлы кладут
короткие «шаговые» сообщения, терминальный лист — итог. Итог всегда последний:
его показывает пользователю вызывающая система (``messages[-1]`` либо по флагу
``orchestrator_final``).

Прод-клиент завязан на эти теги, поэтому имена ключей и поведение флагов
менять нельзя.
"""
from __future__ import annotations

from typing import Any

import markdown2
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

STEP_KEY = "orchestrator_step"      # промежуточный шаг хода
FINAL_KEY = "orchestrator_final"    # итоговый ответ хода (его показываем юзеру)
# Исходный markdown итога при answer_html=true: content — уже HTML, а историю
# для LLM собираем из этого ключа, чтобы модель не имитировала разметку.
MARKDOWN_KEY = "orchestrator_markdown"

TRACE_SECTION_TITLE = "### Как я пришёл к выводу"


def config_flag(config: RunnableConfig | None, key: str, *, default: bool) -> bool:
    """Булев флаг из configurable: строки 'true'/'1'/'yes'/'да' тоже считаются."""
    cfg = (config or {}).get("configurable") or {}
    val = cfg.get(key)
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "1", "yes", "да"}
    return bool(val)


def config_value(config: RunnableConfig | None, key: str, default: Any = None) -> Any:
    cfg = (config or {}).get("configurable") or {}
    val = cfg.get(key)
    return default if val is None else val


def is_step(m: Any) -> bool:
    if isinstance(m, dict):
        return bool((m.get("additional_kwargs") or {}).get(STEP_KEY))
    return bool(getattr(m, "additional_kwargs", None) and m.additional_kwargs.get(STEP_KEY))


def step_update(config: RunnableConfig | None, text: str) -> dict:
    """{'messages': [шаговое AIMessage]} либо {} если прогресс выключен/пусто."""
    text = (text or "").strip()
    if not text or not config_flag(config, "emit_progress_messages", default=True):
        return {}
    return {"messages": [AIMessage(content=text, additional_kwargs={STEP_KEY: True})]}


def step_updates(config: RunnableConfig | None, texts: list[str]) -> list[AIMessage]:
    """Несколько шагов одним списком — узел агента отдаёт их вместе с итогом."""
    if not config_flag(config, "emit_progress_messages", default=True):
        return []
    return [
        AIMessage(content=t.strip(), additional_kwargs={STEP_KEY: True})
        for t in texts or []
        if t and t.strip()
    ]


def render_answer_html(text: str) -> str:
    """Markdown итога → HTML. Детерминированно, без LLM."""
    return markdown2.markdown(
        text, extras=["tables", "fenced-code-blocks", "cuddled-lists"]
    ).strip()


def final_message(text: str, config: RunnableConfig | None) -> AIMessage:
    """Итог хода с флагом; при answer_html=true content — HTML, markdown рядом."""
    kwargs: dict = {FINAL_KEY: True}
    if config_flag(config, "answer_html", default=True):
        kwargs[MARKDOWN_KEY] = text
        text = render_answer_html(text)
    return AIMessage(content=text, additional_kwargs=kwargs)


def plain_text(m: Any) -> str:
    """Текст сообщения строкой (контент бывает строкой или списком блоков)."""
    content = getattr(m, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content or "")


def last_user_text(state: Any) -> str:
    for m in reversed((state or {}).get("messages") or []):
        if isinstance(m, HumanMessage):
            return plain_text(m)
    return ""


def strip_trace_section(content: Any) -> Any:
    """Убирает раздел «Как я пришёл к выводу» из текста AI-реплики.

    Раздел клеится прямо в content и оседает в истории; если подавать его в LLM
    как есть, модель начнёт имитировать раздел и придумывать свои шаги.
    """
    if not isinstance(content, str):
        return content
    idx = content.find(TRACE_SECTION_TITLE)
    if idx == -1:
        return content
    prefix = content[:idx].rstrip()
    if prefix.endswith("---"):
        prefix = prefix[:-3].rstrip()
    return prefix


def history_for_llm(messages: list[Any]) -> list[Any]:
    """История для LLM: без шаговых сообщений, из markdown-источника, без трассы."""
    out: list[Any] = []
    for m in messages or []:
        if is_step(m):
            continue
        if isinstance(m, AIMessage):
            src = (m.additional_kwargs or {}).get(MARKDOWN_KEY)
            out.append(
                AIMessage(
                    content=strip_trace_section(src if isinstance(src, str) else m.content)
                )
            )
        else:
            out.append(m)
    return out


__all__ = [
    "FINAL_KEY",
    "MARKDOWN_KEY",
    "STEP_KEY",
    "TRACE_SECTION_TITLE",
    "config_flag",
    "config_value",
    "final_message",
    "history_for_llm",
    "is_step",
    "last_user_text",
    "plain_text",
    "render_answer_html",
    "step_update",
    "step_updates",
    "strip_trace_section",
]
