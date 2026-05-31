"""LLM-граф смысловых связей между метриками поверх LangGraph Store (Блок D ТЗ).

`influent_percent` (вес влияния дочерней метрики на родителя) бизнес проставляет
вручную и часто забывает/ошибается. Здесь мы достраиваем ДОПОЛНИТЕЛЬНЫЙ граф связей,
выведенный LLM из НАЗВАНИЙ и ОПИСАНИЙ метрик (не из эмбеддингов — нужны
причинно-следственные/lead-lag связи, которых косинус не даёт).

Граф зависит только от каталога метрик направления (имена/описания/тип/родитель), не
от значений fact, поэтому стабилен per-direction и кэшируется в Store по ключу
(direction_key, catalog_hash) — LLM дёргается лишь на холодном кэше. Паттерн повторяет
store_cache.sync_embeddings: namespace Store, ленивый расчёт, изоляция ошибок.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.base import BaseStore

from .prompts import RELATIONS_SYSTEM_PROMPT
from .store_cache import make_hash

_NAMESPACE_ROOT = ("json_analyzer", "metric_relations")
# any↔any связи — O(N²); каталог и число рёбер ограничиваем.
_MAX_CATALOG_METRICS = 60
_MAX_EDGES = 120
_CHUNK_METRICS = 40
_RATIONALE_CAP = 160
_VALID_STRENGTH = {"низкая", "средняя", "высокая"}
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
    """Каталог метрик для LLM-вывода связей. Сортирован по имени (детерминизм
    хэша), усечён до _MAX_CATALOG_METRICS."""
    catalog = sqlite_store.relation_catalog()
    return catalog[:_MAX_CATALOG_METRICS]


def catalog_hash(catalog: list[dict[str, Any]]) -> str:
    """Ключ кэша/инвалидации: меняется при любом изменении состава метрик,
    описаний, типов, родителей или флага has_influent."""
    payload = json.dumps(catalog, ensure_ascii=False, sort_keys=True)
    return make_hash("metric_relations_catalog", payload)


def _extract_json_array(text: str) -> Any:
    """Достаёт JSON-массив из ответа LLM (возможна ```json-обёртка или проза)."""
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


def _parse_edges(text: str, valid_names: set[str]) -> list[dict[str, Any]]:
    data = _extract_json_array(text)
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        target = str(item.get("target") or "").strip()
        relation = str(item.get("relation") or "").strip()
        if source not in valid_names or target not in valid_names:
            continue
        if source == target or not relation:
            continue
        strength = str(item.get("strength") or "").strip().lower()
        if strength not in _VALID_STRENGTH:
            strength = "средняя"
        rationale = str(item.get("rationale") or "").strip()[:_RATIONALE_CAP]
        key = (source, target, relation)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "source": source,
            "target": target,
            "relation": relation,
            "strength": strength,
            "rationale": rationale,
        })
        if len(out) >= _MAX_EDGES:
            break
    return out


def _infer_chunk(
    llm: Any, focus: list[dict[str, Any]], catalog: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    valid_names = {m["metric_name"] for m in catalog}
    focus_names = sorted(m["metric_name"] for m in focus)
    payload = {
        "метрики": catalog,
        "фокус": focus_names,
        "максимум_связей": _MAX_EDGES,
    }
    response = llm.invoke([
        SystemMessage(content=RELATIONS_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ])
    return _parse_edges(_text(response), valid_names)


def _infer_edges(llm: Any, catalog: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Вывод графа связей. None — сбой (НЕ кэшируем); [] — легитимно пустой граф."""
    try:
        if len(catalog) <= _CHUNK_METRICS:
            return _infer_chunk(llm, catalog, catalog)
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for start in range(0, len(catalog), _CHUNK_METRICS):
            chunk = catalog[start:start + _CHUNK_METRICS]
            for edge in _infer_chunk(llm, chunk, catalog):
                key = (edge["source"], edge["target"], edge["relation"])
                if key in seen:
                    continue
                seen.add(key)
                merged.append(edge)
                if len(merged) >= _MAX_EDGES:
                    return merged
        return merged
    except Exception:  # noqa: BLE001 — внешний LLM, сбой не должен валить gather
        return None


async def sync_relations(
    store: BaseStore,
    sqlite_store: Any,
    direction_key: str,
    llm: Any,
) -> list[dict[str, Any]]:
    """Возвращает рёбра графа связей. Кэш в Store по (direction_key, catalog_hash).

    Холодный кэш → один llm.invoke БЕЗ инструментов (в отдельном потоке). Сбой
    LLM/парсинга → [] и БЕЗ записи в кэш (чтобы транзиент не отравил его);
    легитимно пустой результат кэшируется. Сбой Store не пробрасывается.
    """
    catalog = build_catalog(sqlite_store)
    if len(catalog) < 2:
        return []
    h = catalog_hash(catalog)
    namespace = _ns(direction_key)

    try:
        item = await store.aget(namespace, h)
    except Exception:  # noqa: BLE001 — Store-операция, не валим конвейер
        item = None
    if item is not None and isinstance(getattr(item, "value", None), dict):
        edges = item.value.get("edges")
        if edges is not None:
            return edges

    edges = await asyncio.to_thread(_infer_edges, llm, catalog)
    if edges is None:
        # Сбой — отдаём пусто, но НЕ кэшируем (пересчитаем на следующем прогоне).
        return []

    try:
        await store.aput(
            namespace, h,
            {"catalog_hash": h, "edges": edges, "version": _VERSION},
            index=False,
        )
    except Exception:  # noqa: BLE001 — Store-операция, не валим конвейер
        pass
    return edges


__all__ = ["build_catalog", "catalog_hash", "sync_relations"]
