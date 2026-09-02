"""Узлы графа gap_resolver.

``load_gaps`` — нерешённые gap'ы направления (группировка по вопросу);
``investigate`` — по каждой группе ищем ответ в исходных документах
(``source_chunk``), заводим пустую wiki-заглушку по теме и, если ответ нашёлся,
помечаем все дубли вопроса решёнными;

Отбора кандидатов больше нет: у чанков нет эмбеддинга (миграция 0004), поэтому
судья проверяет КАЖДЫЙ чанк направления отдельным вызовом. Отсюда ограничители —
ранняя остановка на найденном ответе, кап чанков на вопрос и общий кап вызовов
судьи на прогон: вопрос без ответа стоит столько вызовов, сколько чанков в
направлении, а таких вопросов обычно большинство.
``finalize`` — собираем человекочитаемый отчёт «решено / не решено / что загрузить».

``direction_key`` и параметры берём из ``config.configurable``, затем из state,
затем из ``config.get_settings()`` — паттерн как в kb_chat/doc_manager.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..easyrag.db import session_scope
from ..easyrag.models import QueryGap
from ..wiki_ingest.llm import LLMClient
from ..wiki_ingest.repository import ensure_stub_page
from .config import get_settings
from .judge import answer_in_sources
from .repository import load_unresolved_groups, mark_groups_resolved
from .retrieval import SourceMatch, load_source_chunks
from .state import GapResolverState
from .topic import extract_topics

logger = logging.getLogger(__name__)

_PREVIEW = 200
# Слова короче трёх букв в упорядочивании не участвуют — тот же приём, что в
# поиске страниц-заглушек в оркестраторе.
_MIN_WORD = 3


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


def _words(text: str) -> set[str]:
    return {w.casefold() for w in re.findall(r"\w+", text or "") if len(w) >= _MIN_WORD}


def order_chunks(query: str, chunks: list[SourceMatch]) -> list[SourceMatch]:
    """Очередь обхода: сперва чанки со словами вопроса.

    Ничего НЕ отбрасывает — меняется только порядок. Он важен из-за ранней
    остановки: если ответ есть, его лучше найти на первых вызовах судьи, а не на
    последних. Порядок внутри равных групп остаётся исходным (документ, позиция).
    """
    wanted = _words(query)
    if not wanted:
        return list(chunks)
    scored = [
        (len(wanted & _words(c.text)), -i, c) for i, c in enumerate(chunks)
    ]
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return [c for _, _, c in scored]


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


async def _judge_chunks(
    query: str,
    chunks: list[SourceMatch],
    *,
    llm: Any,
    concurrency: int,
    budget: int,
) -> tuple[SourceMatch | None, str, int]:
    """Проверяет чанки судьёй по одному; возвращает (чанк-ответ, цитата, сколько проверено).

    Ранняя остановка: как только вердикт положительный, остальные проверки
    отменяются — незачем платить за то, что уже найдено. ``budget`` ограничивает
    число вызовов сверху (общий потолок прогона).
    """
    if not chunks or budget <= 0:
        return None, "", 0

    semaphore = asyncio.Semaphore(max(1, concurrency))
    checked = 0
    found_at: SourceMatch | None = None
    quote = ""
    stop = asyncio.Event()

    async def check(chunk: SourceMatch) -> None:
        nonlocal checked, found_at, quote
        if stop.is_set():
            return
        async with semaphore:
            if stop.is_set():
                return
            verdict = await answer_in_sources(query, [chunk.text], llm=llm)
            checked += 1
            if verdict.found is None:
                # Судья не ответил после ретраев — трактуем консервативно как
                # «не подтверждено», вопрос останется открытым.
                logger.warning(
                    "gap_resolver: судья не дал вердикт по чанку %s#%s",
                    chunk.uri, chunk.ord,
                )
                return
            if verdict.found and found_at is None:
                found_at = chunk
                quote = verdict.quote
                stop.set()

    tasks = [asyncio.create_task(check(c)) for c in chunks[:budget]]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        for task in tasks:
            task.cancel()
    return found_at, quote, checked


async def investigate(state: GapResolverState, config: RunnableConfig) -> dict:
    direction_key = _resolve_direction_key(state, config)
    gaps = state.get("gaps") or []
    settings = get_settings()
    max_chunks = _cfg_int(config, state, "max_chunks_per_gap", settings.max_chunks_per_gap)
    concurrency = _cfg_int(config, state, "judge_concurrency", settings.judge_concurrency)
    max_calls = _cfg_int(config, state, "max_judge_calls", settings.max_judge_calls)
    create_stub = _cfg_bool(config, "create_stub_pages", settings.create_stub_pages)
    mark_resolved = _cfg_bool(config, "mark_resolved", settings.mark_resolved)

    llm = LLMClient()
    topics = await extract_topics([g["query"] for g in gaps], llm=llm)

    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    stub_slugs: list[str] = []
    errors: list[dict[str, Any]] = list(state.get("errors") or [])
    calls_left = max_calls
    budget_exhausted = False

    for group, topic in zip(gaps, topics):
        if calls_left <= 0:
            # Потолок вызовов исчерпан: оставшиеся вопросы честно не трогаем,
            # они разберутся на следующем прогоне.
            budget_exhausted = True
            break
        try:
            async with session_scope() as session:
                gap = await session.get(QueryGap, UUID(group["id"]))
                if (
                    gap is None
                    or gap.resolved_at is not None
                    or gap.direction_key != direction_key
                ):
                    continue

                chunks = await load_source_chunks(
                    session, direction_key=direction_key, limit=max_chunks
                )
                ordered = order_chunks(group["query"], chunks)
                best, quote, checked = await _judge_chunks(
                    group["query"], ordered,
                    llm=llm, concurrency=concurrency, budget=calls_left,
                )
                calls_left -= checked
                found = best is not None

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
                    "chunks_checked": checked,
                }
                if found:
                    rec["evidence"] = {
                        "uri": best.uri,
                        "ord": best.ord,
                        "preview": (quote or best.text or "")[:_PREVIEW],
                    }
                    if mark_resolved:
                        await mark_groups_resolved(session, group["ids"])
                else:
                    # Уперлись в кап — значит просмотрели не весь корпус, и
                    # «ответа нет» здесь слабее, чем при полном обходе.
                    rec["capped"] = len(chunks) >= max_chunks and checked >= max_chunks
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
        "budget_exhausted": budget_exhausted,
        "judge_calls": max_calls - calls_left,
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
        src = f"{ev.get('uri', '?')}, фрагмент {ev.get('ord')}" if ev else "?"
        stub = f"; заглушка «{r['stub_slug']}»" if r.get("stub_slug") else ""
        checked = r.get("chunks_checked")
        scanned = f"; проверено фрагментов: {checked}" if checked is not None else ""
        lines.append(
            f"  • «{r['query']}» → тема «{r['topic']}»; источник: {src}{scanned}{stub}"
        )

    lines.append(f"Не решено (данных в источниках нет): {len(unresolved)}")
    for u in unresolved:
        stub = f"; заглушка «{u['stub_slug']}»" if u.get("stub_slug") else ""
        checked = u.get("chunks_checked")
        scanned = f"; проверено фрагментов: {checked}" if checked is not None else ""
        # Кап означает, что корпус просмотрен не весь, и вывод «ответа нет» слабее.
        capped = " (дошли до предела фрагментов на вопрос)" if u.get("capped") else ""
        lines.append(
            f"  • «{u['query']}» → тема «{u['topic']}»{scanned}{capped}{stub}"
        )

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
    if state.get("budget_exhausted"):
        lines.append(
            "Достигнут потолок вызовов проверки: часть вопросов не разобрана, "
            "они попадут в следующий прогон."
        )
    if errors:
        lines.append(f"Ошибок: {len(errors)}.")

    return _reply("\n".join(lines))


def after_load(state: GapResolverState) -> str:
    return "investigate" if (state.get("gaps") or []) else "finalize"


__all__ = ["after_load", "finalize", "investigate", "load_gaps"]
