"""Узлы графа external_doc_loader.

``fetch_documents`` — забрать документы из внешнего источника (заглушка в
shared) и отбросить невалидные записи;
``load_documents`` — по каждому направлению сравнить поступившие документы с
уже загруженными (ТОЛЬКО своего direction_key) и загрузить новые как pending;
``finalize`` — человекочитаемый отчёт «создано / дубли / похожие / ошибки».

Пороги и режим замены берём из ``config.configurable``, затем из
``config.get_settings()`` — паттерн как в gap_resolver.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..easyrag.db import session_scope
from ..shared.external_source import ExternalDocumentSource
from .compare import content_sha256, match_against_existing
from .config import get_settings
from .repository import (
    ExistingDoc,
    delete_doc,
    insert_pending_doc,
    load_direction_docs,
)
from .state import ExternalDocLoaderState

logger = logging.getLogger(__name__)


def _reply(text: str, **extra: Any) -> dict:
    """Ответ узла: и человекочитаемый report, и AIMessage (как в gap_resolver)."""
    return {"messages": [AIMessage(content=text)], "report": text, **extra}


def _configurable(config: RunnableConfig | None) -> dict:
    return (config or {}).get("configurable") or {}


def _cfg_int(config, key: str, default: int) -> int:
    v = _configurable(config).get(key)
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
    v = _configurable(config).get(key)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return bool(v)


async def fetch_documents(state: ExternalDocLoaderState, config: RunnableConfig) -> dict:
    settings = get_settings()
    max_docs = _cfg_int(config, "max_docs", settings.max_docs)

    try:
        raw = await ExternalDocumentSource().fetch_documents()
    except Exception as exc:
        logger.exception("external_doc_loader: внешний источник недоступен")
        return {"documents": [], "errors": [{"error": f"источник недоступен: {exc}"}]}

    documents: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for i, item in enumerate(raw or []):
        direction = str((item or {}).get("direction_id") or "").strip()
        text = str((item or {}).get("text") or "").strip()
        if not direction or not text:
            errors.append({"index": i, "error": "пустой direction_id или text"})
            continue
        if len(documents) >= max_docs:
            errors.append({"index": i, "error": f"за пределами лимита max_docs={max_docs}"})
            continue
        documents.append({"direction_id": direction, "text": text})

    return {"documents": documents, "errors": errors}


async def load_documents(state: ExternalDocLoaderState, config: RunnableConfig) -> dict:
    settings = get_settings()
    similar_threshold = _cfg_float(config, "similar_threshold", settings.similar_threshold)
    replace_similar = _cfg_bool(config, "replace_similar", settings.replace_similar)

    # Группируем по направлению: существующие документы читаются один раз на
    # направление, и сравнение не выходит за его пределы.
    by_direction: dict[str, list[str]] = {}
    for doc in state.get("documents") or []:
        by_direction.setdefault(doc["direction_id"], []).append(doc["text"])

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = list(state.get("errors") or [])

    for direction_key, texts in by_direction.items():
        direction_results: list[dict[str, Any]] = []
        try:
            async with session_scope() as session:
                existing = await load_direction_docs(session, direction_key=direction_key)
                for text in texts:
                    sha = content_sha256(text)
                    verdict = match_against_existing(
                        text, sha, existing, similar_threshold=similar_threshold
                    )
                    uri = f"external://{direction_key}/{sha[:12]}"
                    rec: dict[str, Any] = {"direction_key": direction_key, "uri": uri}

                    if verdict.kind == "duplicate":
                        rec.update(status="duplicate", match_uri=verdict.match.uri)
                    elif verdict.kind == "similar" and not replace_similar:
                        rec.update(
                            status="similar_skipped",
                            match_uri=verdict.match.uri,
                            similarity=round(verdict.similarity, 3),
                        )
                    else:
                        if verdict.kind == "similar":
                            await delete_doc(session, doc_id=verdict.match.id)
                            existing = [d for d in existing if d.id != verdict.match.id]
                            rec.update(
                                status="replaced",
                                match_uri=verdict.match.uri,
                                similarity=round(verdict.similarity, 3),
                            )
                        else:
                            rec["status"] = "created"
                        doc_id = await insert_pending_doc(
                            session,
                            direction_key=direction_key,
                            uri=uri,
                            content=text,
                            sha256=sha,
                        )
                        rec["id"] = doc_id
                        # Свежезагруженный участвует в сравнении со следующими
                        # документами батча — дубль внутри поставки тоже ловим.
                        existing.append(ExistingDoc(doc_id, uri, sha, text))
                    direction_results.append(rec)
            # Транзакция направления закоммичена — только теперь фиксируем её
            # результаты в стейте, чтобы отчёт не разошёлся с БД при откате.
            results.extend(direction_results)
        except Exception as exc:  # одно битое направление не валит весь прогон
            logger.exception(
                "external_doc_loader: ошибка по направлению %s", direction_key
            )
            errors.append({"direction_key": direction_key, "error": str(exc)})

    return {"results": results, "errors": errors}


async def finalize(state: ExternalDocLoaderState) -> dict:
    results = state.get("results") or []
    errors = state.get("errors") or []

    if not results and not errors:
        return _reply("Внешний источник не вернул документов — загружать нечего.")

    by_status: dict[str, list[dict]] = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r)

    created = by_status.get("created", [])
    replaced = by_status.get("replaced", [])
    duplicates = by_status.get("duplicate", [])
    similar = by_status.get("similar_skipped", [])

    lines = [
        f"Документов из внешнего источника обработано: {len(results)}.",
        f"Загружено новых (pending, ждут wiki_ingest): {len(created)}",
    ]
    for r in created:
        lines.append(f"  • [{r['direction_key']}] {r['uri']}")

    if replaced:
        lines.append(f"Заменено изменённых версий: {len(replaced)}")
        for r in replaced:
            lines.append(
                f"  • [{r['direction_key']}] {r['match_uri']} → {r['uri']} "
                f"(сходство {r['similarity']})"
            )
    if duplicates:
        lines.append(f"Пропущено точных дублей: {len(duplicates)}")
        for r in duplicates:
            lines.append(f"  • [{r['direction_key']}] уже загружен как {r['match_uri']}")
    if similar:
        lines.append(
            f"Пропущено похожих (сходство выше порога, replace_similar выключен): "
            f"{len(similar)}"
        )
        for r in similar:
            lines.append(
                f"  • [{r['direction_key']}] похож на {r['match_uri']} "
                f"(сходство {r['similarity']})"
            )
    if errors:
        lines.append(f"Ошибок: {len(errors)}.")
        for e in errors:
            lines.append(f"  • {e}")

    return _reply("\n".join(lines))


def after_fetch(state: ExternalDocLoaderState) -> str:
    return "load_documents" if (state.get("documents") or []) else "finalize"


__all__ = ["after_fetch", "fetch_documents", "finalize", "load_documents"]
