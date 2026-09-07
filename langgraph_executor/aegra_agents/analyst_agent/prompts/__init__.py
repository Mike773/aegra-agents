"""Сборка единственного системного промпта агента.

Порядок блоков: бизнес-промпт (роль, принципы, стиль, структура ответа) → как
работать инструментами → схема данных → что в этих данных → карта отклонений →
память и брифинг → подсказка хода. Бизнес-промпт заменяется целиком через
``system_prompt_override``; операционные блоки остаются всегда — без них модель
не сможет обращаться к данным.
"""
from __future__ import annotations

from dataclasses import dataclass

from .business import BUSINESS_PROMPT, STAR_PROSE_BLOCK
from .describe import DESCRIBE_ANSWER_PROMPT
from .misc import LOAD_ERROR_PROMPT, SAVE_INSIGHT_AUTO_FIXED
from .task_hints import DASHBOARD_TASK_HINT
from .tools_guide import tools_guide_block

# Память-заглушка возвращает эту формулировку — в промпт её тащить незачем.
_EMPTY_MEMORY_MARKER = "отсутствует"
_BRIEFING_CAP = 4000


@dataclass
class PromptContext:
    """Всё, из чего собирается системный промпт одного хода."""

    schema_doc: str = ""
    enrichment_block: str = ""
    catalog_block: str = ""
    deviations_block: str = ""
    knowledge_block: str = ""
    org_block: str = ""
    memory_context: str | None = None
    briefing: str | None = None
    task_block: str = ""
    turn_kind: str = "initial"          # initial | followup | dashboard
    has_stars: bool = False
    tool_budget: int = 18
    system_prompt_override: str | None = None


def _briefing_block(ctx: PromptContext) -> str:
    text = (ctx.briefing or "").strip()
    if not text:
        return ""
    if len(text) > _BRIEFING_CAP:
        text = text[:_BRIEFING_CAP] + "…"
    return (
        "Брифинг руководителя (первое сообщение диалога, держи его формат и фокус "
        "весь разговор):\n" + text
    )


def _memory_block(ctx: PromptContext) -> str:
    text = (ctx.memory_context or "").strip()
    if not text or _EMPTY_MEMORY_MARKER in text.lower():
        return ""
    return "Контекст прошлых диалогов с этим руководителем:\n" + text


def _turn_hint(ctx: PromptContext) -> str:
    """Подсказка есть только у задачи составного разбора: обычный ход ведёт
    входное сообщение руководителя, ответ на реплику — история диалога."""
    return DASHBOARD_TASK_HINT if ctx.turn_kind == "dashboard" else ""


def compose_system_prompt(ctx: PromptContext) -> str:
    """Собирает системный промпт; пустые блоки просто опускаются."""
    override = (ctx.system_prompt_override or "").strip()
    business = override or BUSINESS_PROMPT
    tail: list[str] = [STAR_PROSE_BLOCK] if ctx.has_stars and not override else []

    blocks: list[str] = [
        business,
        tools_guide_block(ctx.tool_budget),
        ctx.schema_doc,
        ctx.org_block,
        ctx.enrichment_block,
        ctx.catalog_block,
        ctx.knowledge_block,
        ctx.deviations_block,
        _memory_block(ctx),
        _briefing_block(ctx),
        ctx.task_block,
        *tail,
        _turn_hint(ctx),
    ]
    return "\n\n".join(b.strip() for b in blocks if b and b.strip())


__all__ = [
    "BUSINESS_PROMPT",
    "DASHBOARD_TASK_HINT",
    "DESCRIBE_ANSWER_PROMPT",
    "LOAD_ERROR_PROMPT",
    "PromptContext",
    "SAVE_INSIGHT_AUTO_FIXED",
    "STAR_PROSE_BLOCK",
    "compose_system_prompt",
    "tools_guide_block",
]
