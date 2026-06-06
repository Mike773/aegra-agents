"""Узлы графа gap_resolver.

``load_gaps`` — нерешённые gap'ы направления (группировка по вопросу);
``investigate`` — по каждой группе ищем ответ в исходных документах
(``source_chunk``), заводим пустую wiki-заглушку по теме и, если ответ нашёлся,
помечаем все дубли вопроса решёнными;
``finalize`` — собираем человекочитаемый отчёт «решено / не решено / что загрузить».

``direction_key`` и параметры берём из ``config.configurable``, затем из state,
затем из ``config.get_settings()`` — паттерн как в kb_chat/doc_manager.
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..easyrag.db import session_scope
from ..easyrag.models import QueryGap
from ..wiki_ingest.embeddings import EmbeddingClient
from ..wiki_ingest.llm import LLMClient
from ..wiki_ingest.repository import ensure_stub_page
from .config import get_settings
from .judge import answer_in_sources
from .repository import load_unresolved_groups, mark_groups_resolved
from .retrieval import search_source_chunks
from .state import GapResolverState
from .topic import extract_topics

logger = logging.getLogger(__name__)

_PREVIEW = 200


def _reply(text: str, **extra: Any) -> dict:
    """Ответ узла: и человекочитаемый report, и AIMessage (как в doc_manager)."""
    return {"messages": [AIMessage(content=text)], "report": text, **extra}


def _configurable(config: RunnableConfig | None) -> dict:
    return (config or {}).get("configurable") or {}


def _resolve_direction_key(state: GapResolverState, config: RunnableConfig | None) -> str:
    cfg = _configurable(config)
    return (cfg.get("direction_key") or state.get("direction_key") or "").strip()


def _first_set(*values: Any) -> Any:
    """Первое не-None значение — без ловушки falsy (0/'' тоже валидны)."""
    for v in values:
        if v is not None:
            return v
    return None


def _cfg_int(config, state, key: str, default: int) -> int:
    v = _first_set(_configurable(config).get(key), state.get(key))
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _cfg_float(config, key: str, default: float) -> float:
    v = _configurable(config).get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _cfg_bool(config, key: str, default: bool) -> bool:
    cfg = _configurable(config)
    v = cfg.get(key)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return bool(v)


async def load_gaps(state: GapResolverState, config: RunnableConfig) -> dict:
    direction_key = _resolve_direction_key(state, config)
    if not direction_key:
        return {"gaps": [], "errors": [{"error": "direction_key required"}]}
    settings = get_settings()
    async with session_scope() as session:
        groups = await load_unresolved_groups(
            session, direction_key=direction_key, limit=settings.max_gaps
        )
    return {"direction_key": direction_key, "gaps": groups}


async def investigate(state: GapResolverState, config: RunnableConfig) -> dict:
    direction_key = _resolve_direction_key(state, config)
    gaps = state.get("gaps") or []
    settings = get_settings()
    top_k = _cfg_int(config, state, "top_k", settings.top_k)
    candidate_thresh = _cfg_float(config, "candidate_thresh", settings.candidate_thresh)
    create_stub = _cfg_bool(config, "create_stub_pages", settings.create_stub_pages)
    mark_resolved = _cfg_bool(config, "mark_resolved", settings.mark_resolved)

    embedder = EmbeddingClient()
    llm = LLMClient()
    topics = await extract_topics([g["query"] for g in gaps], llm=llm)

    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    stub_slugs: list[str] = []
    errors: list[dict[str, Any]] = list(state.get("errors") or [])

    for group, topic in zip(gaps, topics):
        try:
            async with session_scope() as session:
                gap = await session.get(QueryGap, UUID(group["id"]))
                if (
                    gap is None
                    or gap.resolved_at is not None
                    or gap.direction_key != direction_key
                ):
                    continue

                if gap.embedding is not None:
                    vec = [float(x) for x in gap.embedding]
                else:
                    vec = await embedder.embed_one(group["query"])

                matches = await search_source_chunks(
                    session, direction_key=direction_key, query_vec=vec, top_k=top_k
                )
                best = matches[0] if matches else None

                # Векторный отбор → LLM-судья по тексту кандидатов. Близость лишь
                # тематическая (вопрос про объект ≠ ответ про него есть), поэтому
                # «найден ли ответ» решает judge; порог — только префильтр кандидатов.
                candidates = [m for m in matches if m.similarity >= candidate_thresh]
                quote = ""
                if candidates:
                    verdict = await answer_in_sources(
                        group["query"], [m.text for m in candidates], llm=llm
                    )
                    if verdict.found is None:
                        logger.warning(
                            "gap_resolver: судья не дал вердикт по gap id=%s — оставляю открытым",
                            group["id"],
                        )
                    # found=None трактуем консервативно как «не подтверждено» —
                    # gap остаётся открытым до следующего прогона.
                    found = bool(verdict.found)
                    quote = verdict.quote
                else:
                    found = False

                stub_slug = None
                if create_stub:
                    page = await ensure_stub_page(
                        session, direction_key=direction_key, name=topic
                    )
                    if page is not None:
                        stub_slug = page.slug

                rec: dict[str, Any] = {
                    "query": group["query"],
                    "topic": topic,
                    "stub_slug": stub_slug,
                }
                if found:
                    rec["evidence"] = {
                        "uri": best.uri,
                        "similarity": round(best.similarity, 3),
                        "preview": (quote or best.text or "")[:_PREVIEW],
                    }
                    if mark_resolved:
                        await mark_groups_resolved(session, group["ids"])
                else:
                    rec["best_similarity"] = (
                        round(best.similarity, 3) if best is not None else None
                    )
            # Транзакция закоммичена — только теперь фиксируем результат в стейте,
            # чтобы при откате (например, упал mark) отчёт не разошёлся с БД.
            if stub_slug:
                stub_slugs.append(stub_slug)
            (resolved if found else unresolved).append(rec)
        except Exception as exc:  # один битый gap не валит весь прогон
            logger.exception("gap_resolver: ошибка по gap id=%s", group.get("id"))
            errors.append({"gap_id": group.get("id"), "error": str(exc)})

    return {
        "resolved": resolved,
        "unresolved": unresolved,
        "created_stub_pages": list(dict.fromkeys(stub_slugs)),
        "errors": errors,
    }


async def finalize(state: GapResolverState) -> dict:
    direction_key = state.get("direction_key") or ""
    resolved = state.get("resolved") or []
    unresolved = state.get("unresolved") or []
    stubs = state.get("created_stub_pages") or []
    errors = state.get("errors") or []

    if not direction_key:
        return _reply(
            "Не задан direction_key — нечего разбирать. "
            "Передайте его в configurable."
        )

    total = len(resolved) + len(unresolved)
    if total == 0:
        text = f"Направление «{direction_key}»: нерешённых вопросов нет."
        if errors:
            text += f" Ошибок: {len(errors)}."
        return _reply(text)

    lines = [
        f"Направление «{direction_key}». Разобрано нерешённых вопросов: {total}.",
        f"Решено (ответ есть в исходных документах): {len(resolved)}",
    ]
    for r in resolved:
        ev = r.get("evidence") or {}
        src = f"{ev.get('uri', '?')} (близость {ev.get('similarity')})" if ev else "?"
        stub = f"; заглушка «{r['stub_slug']}»" if r.get("stub_slug") else ""
        lines.append(f"  • «{r['query']}» → тема «{r['topic']}»; источник: {src}{stub}")

    lines.append(f"Не решено (данных в источниках нет): {len(unresolved)}")
    for u in unresolved:
        stub = f"; заглушка «{u['stub_slug']}»" if u.get("stub_slug") else ""
        lines.append(f"  • «{u['query']}» → тема «{u['topic']}»{stub}")

    if unresolved:
        topics = list(dict.fromkeys(u["topic"] for u in unresolved))
        lines.append("")
        lines.append(
            "Что загрузить: документы, описывающие — " + "; ".join(topics) + "."
        )

    if stubs:
        lines.append(
            f"Создано пустых wiki-страниц: {len(stubs)} ({', '.join(stubs)})."
        )
    if errors:
        lines.append(f"Ошибок: {len(errors)}.")

    return _reply("\n".join(lines))


def after_load(state: GapResolverState) -> str:
    return "investigate" if (state.get("gaps") or []) else "finalize"


__all__ = ["after_load", "finalize", "investigate", "load_gaps"]
