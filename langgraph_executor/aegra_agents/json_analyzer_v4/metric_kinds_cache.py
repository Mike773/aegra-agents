"""LLM-классификация ПРИРОДЫ метрик поверх LangGraph Store.

Относительное %-сравнение (изменение к прошлому периоду в %, отклонение от плана в %)
осмысленно только для метрик-«уровней». Для знаковых «вкладов» (влияние/разница,
значения ±, центр у нуля) и «индексов» (ранг/место) деление на базу даёт бред
(−303%, смена знака) и даже инвертирует вердикт динамики. Поэтому классифицируем
метрику по НАЗВАНИЮ и ОПИСАНИЮ (не по значениям) в один из видов и для не-«уровней»
подавляем относительные % (см. analytics.apply_metric_kinds).

Классификация зависит только от каталога метрик направления → стабильна и
кэшируется в Store по (direction_key, catalog_hash). Паттерн повторяет
relations_cache.sync_relations: namespace Store, ленивый расчёт, изоляция ошибок.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.base import BaseStore

from .prompts import METRIC_KINDS_SYSTEM_PROMPT
from .store_cache import make_hash

_NAMESPACE_ROOT = ("json_analyzer", "metric_kinds")
_MAX_CATALOG_METRICS = 80
_CHUNK_METRICS = 40
_VALID_KINDS = {"уровень", "вклад", "индекс"}
_DEFAULT_KIND = "уровень"
_VERSION = 1


def _ns(direction_key: str) -> tuple[str, ...]:
    return (*_NAMESPACE_ROOT, direction_key)


def _text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def build_catalog(sqlite_store: Any) -> list[dict[str, Any]]:
    """Каталог метрик для классификации (имя/описание/тип/единица), отсортирован по
    имени (детерминизм хэша), усечён до _MAX_CATALOG_METRICS."""
    out: list[dict[str, Any]] = []
    for m in sqlite_store.relation_catalog()[:_MAX_CATALOG_METRICS]:
        out.append({
            "metric_name": m.get("metric_name"),
            "metric_description": m.get("metric_description"),
            "metric_type": m.get("metric_type"),
            "measure_type": m.get("measure_type"),
        })
    return out


def catalog_hash(catalog: list[dict[str, Any]]) -> str:
    payload = json.dumps(catalog, ensure_ascii=False, sort_keys=True)
    return make_hash("metric_kinds_catalog", payload)


def _extract_json_array(text: str) -> Any:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _parse_kinds(text: str, valid_names: set[str]) -> dict[str, str]:
    data = _extract_json_array(text)
    if not isinstance(data, list):
        return {}
    out: dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("metric") or "").strip()
        kind = str(item.get("kind") or "").strip().lower()
        if name not in valid_names:
            continue
        if kind not in _VALID_KINDS:
            kind = _DEFAULT_KIND
        out[name] = kind
    return out


def _infer_chunk(llm: Any, chunk: list[dict[str, Any]]) -> dict[str, str]:
    valid_names = {m["metric_name"] for m in chunk}
    payload = {"метрики": chunk}
    response = llm.invoke([
        SystemMessage(content=METRIC_KINDS_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ])
    return _parse_kinds(_text(response), valid_names)


def _infer_kinds(llm: Any, catalog: list[dict[str, Any]]) -> dict[str, str] | None:
    """Классификация всех метрик. None — сбой (НЕ кэшируем); {} — легитимно пусто."""
    try:
        merged: dict[str, str] = {}
        for start in range(0, len(catalog), _CHUNK_METRICS):
            merged.update(_infer_chunk(llm, catalog[start:start + _CHUNK_METRICS]))
        return merged
    except Exception:  # noqa: BLE001 — внешний LLM, сбой не должен валить gather
        return None


async def sync_metric_kinds(
    store: BaseStore,
    sqlite_store: Any,
    direction_key: str,
    llm: Any,
) -> dict[str, str]:
    """Возвращает {metric_name: kind}. Кэш в Store по (direction_key, catalog_hash).

    Холодный кэш → один llm.invoke (в отдельном потоке). Сбой LLM/парсинга → {} и
    БЕЗ записи в кэш (транзиент не отравит кэш); легитимно пустой результат
    кэшируется. Сбой Store не пробрасывается.
    """
    catalog = build_catalog(sqlite_store)
    if not catalog:
        return {}
    h = catalog_hash(catalog)
    namespace = _ns(direction_key)

    try:
        item = await store.aget(namespace, h)
    except Exception:  # noqa: BLE001 — Store-операция, не валим конвейер
        item = None
    if item is not None and isinstance(getattr(item, "value", None), dict):
        kinds = item.value.get("kinds")
        if kinds is not None:
            return kinds

    kinds = await asyncio.to_thread(_infer_kinds, llm, catalog)
    if kinds is None:
        return {}  # сбой — не кэшируем

    try:
        await store.aput(
            namespace, h,
            {"catalog_hash": h, "kinds": kinds, "version": _VERSION},
            index=False,
        )
    except Exception:  # noqa: BLE001 — Store-операция, не валим конвейер
        pass
    return kinds


__all__ = ["build_catalog", "catalog_hash", "sync_metric_kinds"]
