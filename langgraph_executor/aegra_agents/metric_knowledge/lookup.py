"""Поиск трактовки показателя в базе знаний.

По названию и описанию показателя формируются короткие запросы (включая
расшифровку аббревиатур), они уходят в pgvector-поиск easyrag, а найденные
фрагменты модель сводит в структурированную трактовку.

Два правила надёжности:
- ничего не нашли → пишем строку ``status='unknown'``, чтобы не ходить за одним
  и тем же на каждом запуске (перепроверим через неделю);
- модель или база упали → не пишем НИЧЕГО: транзиентный сбой не должен
  отравлять кэш ложной трактовкой.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ..analyst_agent.db.loader import guess_kind as _guess_kind_by_name
from .types import (
    DEFAULT_KIND,
    KNOWLEDGE_VERSION,
    STATUS_FOUND,
    STATUS_UNKNOWN,
    VALID_KINDS,
    CatalogEntry,
    KnowledgeRow,
)

logger = logging.getLogger(__name__)

# Ниже этого сходства считаем, что база знаний про показатель молчит.
MIN_SIMILARITY = 0.45
MAX_QUERIES = 4
MAX_SNIPPETS = 8
_DESC_CAP = 200
_SUMMARY_CAP = 400
# Аббревиатура: 2+ заглавных подряд (латиница или кириллица).
_ABBR_RE = re.compile(r"\b[A-ZА-ЯЁ]{2,}\b")

SYSTEM_PROMPT = (
    "Ты — методолог: объясняешь, что означает показатель эффективности и как его "
    "правильно читать. Тебе дают название показателя, его описание из системы "
    "данных и фрагменты корпоративной базы знаний.\n\n"
    "Сформулируй трактовку показателя строго по этим фрагментам:\n"
    "- summary: одна-две фразы, что это за показатель (без воды и без чисел);\n"
    "- aliases: аббревиатуры и синонимы, которыми его называют;\n"
    "- kind: 'уровень' — обычная величина; 'вклад' — знаковая величина влияния "
    "или разницы (центр у нуля); 'индекс' — ранг, место, номер позиции;\n"
    "- cumulative: правда, если значение копится с начала периода;\n"
    "- plan_semantics / fact_semantics: особенности того, как считается план и "
    "факт, если про это сказано;\n"
    "- processes: в каких процессах участвует показатель;\n"
    "- notes: прочие важные оговорки трактовки;\n"
    "- confidence: от 0 до 1 — насколько уверенно фрагменты отвечают о ЭТОМ "
    "показателе.\n\n"
    "Не выдумывай: если во фрагментах чего-то нет, оставь поле пустым. Если "
    "фрагменты вообще не про этот показатель, поставь confidence близким к нулю."
)

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "aliases": {"type": "array", "items": {"type": "string"}},
        "kind": {"type": "string", "enum": list(VALID_KINDS)},
        "cumulative": {"type": "boolean"},
        "plan_semantics": {"type": "string"},
        "fact_semantics": {"type": "string"},
        "processes": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["summary", "confidence"],
}


def guess_kind(entry: CatalogEntry) -> str:
    """Вид показателя по названию — та же эвристика, что при сборке базы."""
    return _guess_kind_by_name(entry.metric_name, entry.metric_description)


def build_queries(entry: CatalogEntry) -> list[str]:
    """Короткие запросы к базе знаний: имя, имя с описанием, аббревиатуры."""
    name = (entry.metric_name or "").strip()
    if not name:
        return []
    queries = [name]
    description = (entry.metric_description or "").strip()
    if description:
        queries.append(f"{name}: {description[:_DESC_CAP]}")
    seen = {name.casefold()}
    for abbr in _ABBR_RE.findall(name):
        if abbr.casefold() in seen:
            continue
        seen.add(abbr.casefold())
        queries.append(f"что такое {abbr}")
        if len(queries) >= MAX_QUERIES:
            break
    return queries[:MAX_QUERIES]


def _clean_list(value: Any, cap: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v or "").strip()][:cap]


def _unknown_row(entry: CatalogEntry, model_name: str | None) -> KnowledgeRow:
    return KnowledgeRow(
        metric_key=entry.metric_key,
        name_key=entry.name_key,
        metric_name=entry.metric_name,
        metric_description=entry.metric_description,
        summary="",
        # Вид показателя знаем и без базы знаний — он выводится из названия.
        rules={"kind": guess_kind(entry)},
        status=STATUS_UNKNOWN,
        confidence=0.0,
        model=model_name,
        version=KNOWLEDGE_VERSION,
    )


async def build_row(
    entry: CatalogEntry,
    *,
    snippets: list[dict[str, Any]],
    llm: Any,
    model_name: str | None = None,
) -> KnowledgeRow | None:
    """Фрагменты базы знаний → строка кэша. None = транзиентный сбой, не пишем."""
    useful = [
        s
        for s in snippets or []
        if float(s.get("similarity") or 0.0) >= MIN_SIMILARITY
    ][:MAX_SNIPPETS]
    if not useful or llm is None:
        return _unknown_row(entry, model_name)

    context = "\n\n".join(
        f"[{s.get('page_title') or s.get('slug') or '-'} / "
        f"{s.get('section_title') or s.get('anchor') or '-'}]: {s.get('body_md') or ''}"
        for s in useful
    )
    user = (
        f"Показатель: {entry.metric_name}\n"
        f"Описание из системы данных: {entry.metric_description or '—'}\n\n"
        f"Фрагменты базы знаний:\n{context}"
    )
    try:
        payload = await llm.call_json(
            system=SYSTEM_PROMPT,
            user=user,
            tool_name="save_metric_knowledge",
            tool_description="Сохранить трактовку показателя",
            input_schema=TOOL_SCHEMA,
        )
    except Exception:  # noqa: BLE001 — сеть/модель: кэш не отравляем
        logger.warning(
            "knowledge: не удалось получить трактовку «%s»", entry.metric_name, exc_info=True
        )
        return None

    if not isinstance(payload, dict):
        return _unknown_row(entry, model_name)

    kind = str(payload.get("kind") or "").strip()
    if kind not in VALID_KINDS:
        kind = guess_kind(entry)
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    summary = str(payload.get("summary") or "").strip()[:_SUMMARY_CAP]

    rules: dict[str, Any] = {"kind": kind}
    if isinstance(payload.get("cumulative"), bool):
        rules["cumulative"] = payload["cumulative"]
    for key in ("plan_semantics", "fact_semantics", "notes"):
        text = str(payload.get(key) or "").strip()
        if text:
            rules[key] = text[:_SUMMARY_CAP]
    processes = _clean_list(payload.get("processes"))
    if processes:
        rules["processes"] = processes

    if not summary or confidence < 0.2:
        row = _unknown_row(entry, model_name)
        row.rules = rules
        return row

    return KnowledgeRow(
        metric_key=entry.metric_key,
        name_key=entry.name_key,
        metric_name=entry.metric_name,
        metric_description=entry.metric_description,
        aliases=_clean_list(payload.get("aliases")),
        summary=summary,
        rules=rules,
        status=STATUS_FOUND,
        confidence=confidence,
        source_section_ids=[
            str(s["section_id"]) for s in useful if s.get("section_id")
        ],
        model=model_name,
        version=KNOWLEDGE_VERSION,
    )


__all__ = [
    "MAX_QUERIES",
    "MAX_SNIPPETS",
    "MIN_SIMILARITY",
    "SYSTEM_PROMPT",
    "TOOL_SCHEMA",
    "build_queries",
    "build_row",
    "guess_kind",
]
