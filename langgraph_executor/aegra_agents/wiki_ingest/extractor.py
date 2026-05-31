"""Извлечение сущностей-кандидатов из текста через LLM.

Порт easyRag/ingest/extractor.py. ``llm`` приходит параметром (наш
:class:`wiki_ingest.llm.LLMClient`), промпты — из :mod:`wiki_ingest.ingest_prompts`.

* :func:`analyze_document` — один вызов на документ; собирает :class:`DocumentBrief`,
  который потом подаётся как контекст в extraction.
* :func:`extract_entities` — один вызов на чанк; зовёт tool ``save_entities`` и
  приводит ответ к списку :class:`ExtractedEntity`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ingest_prompts import (
    DOC_BRIEF_SCHEMA,
    DOC_BRIEF_SYSTEM,
    DOC_BRIEF_TOOL_DESCRIPTION,
    DOC_BRIEF_TOOL_NAME,
    ENTITY_EXTRACTION_SCHEMA,
    ENTITY_EXTRACTION_SYSTEM,
    ENTITY_EXTRACTION_TOOL_DESCRIPTION,
    ENTITY_EXTRACTION_TOOL_NAME,
    build_brief_user_prompt,
    build_extraction_user_prompt,
)
from .llm import LLMClient, get_llm

_MAX_STATEMENTS = 5


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    descriptor: str
    statements: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DocumentBrief:
    """Профиль документа, построенный LLM по его началу."""

    summary: str
    entity_types: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExtractionResult:
    """Результат извлечения по одному чанку.

    ``entities`` — сущности с самостоятельными фактами (станут полноценными
    страницами). ``mentions`` — имена упомянутых, но не описанных сущностей
    (станут страницами-заглушками для будущего заполнения).
    """

    entities: tuple[ExtractedEntity, ...] = field(default_factory=tuple)
    mentions: tuple[str, ...] = field(default_factory=tuple)


async def analyze_document(
    text: str,
    *,
    source_hint: str | None = None,
    llm: LLMClient | None = None,
) -> DocumentBrief | None:
    """Построить :class:`DocumentBrief` по началу документа.

    Возвращает ``None``, если текст пустой или модель вернула неинформативный
    ответ. Исключений не кидает — вызывающий код продолжит ingest без brief'а.
    """
    if not text or not text.strip():
        return None
    client = llm or get_llm()
    try:
        raw = await client.call_json(
            system=DOC_BRIEF_SYSTEM,
            user=build_brief_user_prompt(text, source_hint=source_hint),
            tool_name=DOC_BRIEF_TOOL_NAME,
            tool_description=DOC_BRIEF_TOOL_DESCRIPTION,
            input_schema=DOC_BRIEF_SCHEMA,
        )
    except Exception:
        return None
    return _coerce_brief(raw)


async def extract_entities(
    text: str,
    *,
    source_hint: str | None = None,
    domain_brief: DocumentBrief | None = None,
    llm: LLMClient | None = None,
) -> ExtractionResult:
    """Извлечь сущности-кандидаты и упоминания из ``text``.

    Возвращает :class:`ExtractionResult`: ``entities`` (дедуп по ``name`` без
    учёта регистра/пробелов) и ``mentions`` (имена упомянутых, но не описанных
    сущностей — для заглушек). Если модель вернула мусор — пустой результат.
    """
    if not text or not text.strip():
        return ExtractionResult()
    client = llm or get_llm()
    raw = await client.call_json(
        system=ENTITY_EXTRACTION_SYSTEM,
        user=build_extraction_user_prompt(
            text, source_hint=source_hint, domain_brief=domain_brief
        ),
        tool_name=ENTITY_EXTRACTION_TOOL_NAME,
        tool_description=ENTITY_EXTRACTION_TOOL_DESCRIPTION,
        input_schema=ENTITY_EXTRACTION_SCHEMA,
    )
    entities = _coerce_entities(raw)
    entity_names = {e.name.casefold() for e in entities}
    mentions = _coerce_mentions(raw, entity_names=entity_names)
    return ExtractionResult(entities=tuple(entities), mentions=mentions)


def _coerce_brief(raw: dict[str, Any]) -> DocumentBrief | None:
    if not isinstance(raw, dict):
        return None
    summary = _clean_str(raw.get("summary"))
    entity_types = _coerce_str_list(raw.get("entity_types"))
    if not summary and not entity_types:
        return None
    return DocumentBrief(summary=summary, entity_types=entity_types)


def _coerce_entities(raw: dict[str, Any]) -> list[ExtractedEntity]:
    items = raw.get("entities") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []
    out: list[ExtractedEntity] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = _clean_str(item.get("name"))
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        descriptor = _clean_str(item.get("descriptor"))
        statements = _coerce_statements(item.get("statements"))
        out.append(
            ExtractedEntity(
                name=name,
                descriptor=descriptor,
                statements=statements,
            )
        )
    return out


def _coerce_statements(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        cleaned = _clean_str(value)
        return (cleaned,) if cleaned else ()
    if not isinstance(value, list):
        return ()
    cleaned_list: list[str] = []
    for s in value:
        if not isinstance(s, str):
            continue
        s = _clean_str(s)
        if s:
            cleaned_list.append(s)
        if len(cleaned_list) >= _MAX_STATEMENTS:
            break
    return tuple(cleaned_list)


def _coerce_mentions(raw: dict[str, Any], *, entity_names: set[str]) -> tuple[str, ...]:
    """Достать ``mentions`` из ответа: имена упомянутых, но не описанных сущностей.

    Дедуп по casefold; исключаем то, что уже попало в ``entities`` этого чанка
    (чтобы не плодить заглушку поверх полноценной сущности).
    """
    items = raw.get("mentions") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for s in items:
        if not isinstance(s, str):
            continue
        s = _clean_str(s)
        if not s:
            continue
        key = s.casefold()
        if key in entity_names or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return tuple(out)


def _coerce_str_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for s in value:
        if not isinstance(s, str):
            continue
        s = _clean_str(s)
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return tuple(out)


def _clean_str(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


__all__ = [
    "DocumentBrief",
    "ExtractedEntity",
    "ExtractionResult",
    "analyze_document",
    "extract_entities",
]
