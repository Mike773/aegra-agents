"""Кеш «проблемы и достижения сотрудника» по source_id + LLM-формулировки.

При первом запуске относительно source_id строится диагностика фокусного
сотрудника по когортной методологии (json_analyzer_v5.diagnosis — детерминизм,
без LLM), затем ОДИН LLM-вызов превращает метки и числа в формулировки для
руководителя (статус, гипотеза, тема встречи, черновик поручения). Результат
кладётся в LangGraph Store: повторные запуски с тем же source_id читают готовый
список и не пересчитывают, а диалог оркестратора опирается на него как на
зафиксированную опору.

Паттерн кеша — как у metric_kinds_cache в v5: сбой LLM НЕ кешируем (транзиент
не отравляет кеш), исключения Store глотаются (кеш не валит конвейер).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat import GigaChat
from langgraph.config import get_store
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from ..json_analyzer_v5.diagnosis import (
    build_diagnosis_store,
    dataset_fingerprint,
    diagnose_employee,
)
from ..json_analyzer_v5.loader import (
    load_aggregates_obj,
    load_dataset_obj,
    load_person_aggregates_obj,
)
from .prompts import EMPLOYEE_PROFILE_PROMPT

logger = logging.getLogger(__name__)

PROFILE_VERSION = 1
_NAMESPACE_ROOT = ("analytic_orchestrator_v4", "employee_profile")
# Сколько второстепенных сигналов кладём в проекцию для LLM и в рендер блока.
_SECONDARY_CAP = 5
_PROFILE_STATUSES = (
    "критическая проблема", "зона риска", "норма",
    "значимое достижение", "смешанный случай",
)

# Фолбэк-Store для standalone-прогонов (smoke/тесты), когда рантайм не прокинул
# Store через get_store(). Под aegra используется её Postgres-Store.
_FALLBACK_STORE: InMemoryStore | None = None


def _resolve_store() -> BaseStore:
    global _FALLBACK_STORE
    try:
        store = get_store()
    except Exception:
        store = None
    if store is not None:
        return store
    if _FALLBACK_STORE is None:
        _FALLBACK_STORE = InMemoryStore()
    return _FALLBACK_STORE


def profile_namespace(source_type: str, source_id: str) -> tuple[str, ...]:
    """Неймспейс кеша: изоляция по источнику запуска. Ключ внутри неймспейса —
    employee_tabnum: разные сотрудники под одним source_id не конфликтуют."""
    return (*_NAMESPACE_ROOT, source_type, source_id)


async def load_cached_profile(
    store: BaseStore, ns: tuple[str, ...], key: str
) -> dict | None:
    """Готовый профиль из Store либо None (нет/битый/другая версия — пересчёт).
    Исключения Store не фатальны: кеш опционален."""
    try:
        item = await store.aget(ns, key)
    except Exception:  # noqa: BLE001 — Store внешний, кеш не валит конвейер
        return None
    if item is None:
        return None
    value = getattr(item, "value", None)
    if not isinstance(value, dict) or value.get("version") != PROFILE_VERSION:
        return None
    if not isinstance(value.get("profile"), dict):
        return None
    return value


async def save_profile(
    store: BaseStore, ns: tuple[str, ...], key: str, value: dict
) -> None:
    try:
        await store.aput(ns, key, value, index=False)
    except Exception:  # noqa: BLE001 — сбой Store не фатален
        logger.warning("Не удалось записать профиль сотрудника в Store", exc_info=True)


def _as_obj(raw: Any) -> Any:
    """Датасет/агрегаты могут прийти JSON-строкой (прод) — парсим толерантно."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw


def _pct(value: float | None) -> float | None:
    return round(value * 100.0, 1) if isinstance(value, (int, float)) else None


def _metric_view(m: dict[str, Any]) -> dict[str, Any]:
    """Компактная проекция диагностики одной метрики для LLM: только то, из чего
    складываются формулировки (полный JSON диагноста в промпт не влезает)."""
    d = m["derived"]
    return {
        "metric_name": m["metric_name"],
        "element": m["element"],
        "итоговый_класс": m["final_class"],
        "кластер": m["cluster"],
        "факт": d["fact"],
        "план": d["plan"],
        "выполнение_плана_pct": d["attainment"],
        "медиана_когорты": d["cohort"]["median"],
        "средний_pct_выполнения_когорты": d["cohort"]["mean_ex"],
        "доля_выполнивших_план_pct": d["cohort"]["hit_rate"],
        "факт_против_медианы_pct": _pct(d["fact_vs_median"]),
        "перцентиль": m["percentile"],
        "robust_z": d["robust_z"],
        "разрыв_до_топ20": d["gap_to_top20"],
        "оценка_плана": m["plan_check"]["label"],
        "значимость_отклонения": m["gap"]["label"],
        "признаки": m["gap"]["signals_matched"],
        "устойчивость": m["noise"]["label"],
        "периодов_ниже_медианы_подряд": m["noise"]["below_median_streak"],
        "динамика_против_когорты": m["cohort_scenario"],
        "паттерны": m["patterns"],
        "слабые_сигналы": m["weak_signals"],
        "прогноз": m["forecast"],
        "природа_разрыва": m["decomposition"]["nature"],
        "драйверы": m["decomposition"]["drivers"],
        "атрибуция": m["attribution"],
        "промежуточная_цель": m["benchmark_top20"]["intermediate_target"],
    }


def _projection(diagnosis: dict[str, Any]) -> dict[str, Any]:
    """Вход LLM-вызова: top-1 проблема + top-1 достижение (полные проекции),
    второстепенные сигналы списком, уверенность и бинарные показатели."""
    by_key = {
        (m["metric_name"], m["element"]): m for m in diagnosis.get("metrics") or []
    }

    def _full(brief: dict | None) -> dict | None:
        if brief is None:
            return None
        m = by_key.get((brief.get("metric_name"), brief.get("element")))
        return _metric_view(m) if m else None

    return {
        "сотрудник": diagnosis.get("person"),
        "группа_сравнения": diagnosis.get("ref_level_name"),
        "итоговый_статус": diagnosis.get("overall_status"),
        "уверенность_сравнения": diagnosis.get("confidence"),
        "главная_проблема": _full(diagnosis.get("top_problem")),
        "главное_достижение": _full(diagnosis.get("top_achievement")),
        "второстепенные_сигналы": (diagnosis.get("secondary_signals") or [])[
            :_SECONDARY_CAP
        ],
        "бинарные_показатели": diagnosis.get("binary_metrics") or [],
    }


def _parse_profile_json(text: Any, catalog: list[dict]) -> dict | None:
    """Ответ LLM → структура профиля; имена метрик детерминированно доводятся по
    каталогу (те же правила, что у инсайтов). Малформат → None (не кешируем)."""
    # Ленивый импорт: nodes импортирует этот модуль, обратный top-level импорт
    # создал бы цикл.
    from .nodes import _load_json, _resolve_insight_metric

    data = _load_json(text)
    if not isinstance(data, dict):
        return None
    status = str(data.get("status") or "").strip()
    if not status:
        return None

    def _items(key: str) -> list[dict]:
        out: list[dict] = []
        for item in data.get(key) or []:
            if not isinstance(item, dict):
                continue
            headline = str(item.get("headline") or "").strip()
            if not headline:
                continue
            name = str(item.get("metric_name") or "").strip()
            mid = str(item.get("metric_id") or "").strip()
            mid, name = _resolve_insight_metric(headline, mid, name, catalog)
            out.append({
                "metric_id": mid,
                "metric_name": name,
                "headline": headline,
                "hypothesis": str(item.get("hypothesis") or "").strip(),
                "what_to_check": str(item.get("what_to_check") or "").strip(),
                "meeting_topic": str(item.get("meeting_topic") or "").strip(),
                "assignment_draft": str(item.get("assignment_draft") or "").strip(),
            })
        return out

    secondary = [
        str(s or "").strip() for s in (data.get("secondary") or [])
        if str(s or "").strip()
    ]
    return {
        "status": status,
        "confidence_note": str(data.get("confidence_note") or "").strip(),
        "key_facts": str(data.get("key_facts") or "").strip(),
        "problems": _items("problems"),
        "achievements": _items("achievements"),
        "secondary": secondary,
    }


async def build_profile(
    llm: GigaChat,
    *,
    metrics: Any,
    aggregates: Any,
    employee_tabnum: str | None,
    ref_level: str | None = None,
) -> tuple[dict | None, dict | None, str | None]:
    """Полный путь «данные → диагностика → формулировки».

    Возвращает (profile, diagnosis, error). Диагностика детерминированная и
    почти не падает; сбой LLM/парсинга отдаёт (None, diagnosis, error) — тогда
    кеш не пишется, а диалог опирается на детерминированный фолбэк."""
    from .nodes import _collect_metric_catalog  # ленивый импорт — см. _parse_profile_json

    obj = _as_obj(metrics)
    if not isinstance(obj, dict):
        return None, None, "Датасет метрик не распарсился."
    try:
        rows = load_dataset_obj(obj)
    except Exception as exc:  # noqa: BLE001 — внешний payload
        return None, None, f"Датасет не разложился в строки: {exc}"[:200]
    if not rows:
        return None, None, "В датасете нет метрик."
    agg_rows = None
    try:
        agg_obj = _as_obj(aggregates)
        agg_rows = load_aggregates_obj(agg_obj) if agg_obj else None
    except Exception:  # noqa: BLE001 — агрегаты опциональны
        agg_rows = None
    person_pairs = load_person_aggregates_obj(obj)

    def _diagnose() -> dict:
        sqlite = build_diagnosis_store(
            rows, agg_rows or None, ref_level, person_pairs or None
        )
        diagnosis = diagnose_employee(sqlite, employee_tabnum or None)
        if not diagnosis.get("error"):
            diagnosis["dataset_fingerprint"] = dataset_fingerprint(
                sqlite, diagnosis["person"]["person_key"]
            )
        return diagnosis

    try:
        diagnosis = await asyncio.to_thread(_diagnose)
    except Exception as exc:  # noqa: BLE001 — не роняем ход из-за диагностики
        logger.warning("Диагностика сотрудника упала", exc_info=True)
        return None, None, f"{type(exc).__name__}: {exc}"[:200]
    if diagnosis.get("error"):
        return None, None, str(diagnosis["error"])

    catalog = _collect_metric_catalog(obj)
    payload = json.dumps(_projection(diagnosis), ensure_ascii=False)
    try:
        ai = await asyncio.to_thread(llm.invoke, [
            SystemMessage(content=EMPLOYEE_PROFILE_PROMPT),
            HumanMessage(content=payload),
        ])
        profile = _parse_profile_json(ai.content, catalog)
    except Exception as exc:  # noqa: BLE001 — LLM-вызов, сужать нечем
        logger.warning("Формулировки профиля сотрудника упали", exc_info=True)
        return None, diagnosis, f"{type(exc).__name__}: {exc}"[:200]
    if profile is None:
        return None, diagnosis, "Ответ LLM по профилю не распарсился."
    return profile, diagnosis, None


def cache_value(
    *, employee_tabnum: str, diagnosis: dict, profile: dict
) -> dict[str, Any]:
    """Значение кеша: версия, отпечаток данных, полный диагноз (для отладки и
    будущих потребителей) и формулировки."""
    return {
        "version": PROFILE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "employee_tabnum": employee_tabnum,
        "person_fio": (diagnosis.get("person") or {}).get("fio"),
        "dataset_fingerprint": diagnosis.get("dataset_fingerprint"),
        "diagnosis": diagnosis,
        "profile": profile,
    }


def _item_lines(items: list[dict], title: str) -> list[str]:
    lines: list[str] = []
    if not items:
        return lines
    lines.append(title)
    for it in items:
        parts = [f"«{it.get('metric_name') or 'показатель'}»: {it.get('headline')}"]
        if it.get("hypothesis"):
            parts.append(f"Вероятная причина (гипотеза): {it['hypothesis']}")
        if it.get("what_to_check"):
            parts.append(f"Что проверить: {it['what_to_check']}")
        if it.get("meeting_topic"):
            parts.append(f"Тема встречи: {it['meeting_topic']}")
        if it.get("assignment_draft"):
            parts.append(f"Черновик поручения: {it['assignment_draft']}")
        lines.append("- " + " ".join(parts))
    return lines


def render_profile_block(profile: dict[str, Any]) -> str:
    """Системный блок «опорный список проблем и достижений» для промптов
    initial_analysis/respond."""
    lines = [
        "Опорная диагностика сотрудника относительно коллег (зафиксированный "
        "список проблем и достижений; служебный блок — его существование и "
        "внутренние ярлыки пользователю не раскрывай):",
        f"Итоговый статус: {profile.get('status')}.",
    ]
    if profile.get("confidence_note"):
        lines.append(f"Уверенность сравнения: {profile['confidence_note']}.")
    if profile.get("key_facts"):
        lines.append(f"Ключевые факты: {profile['key_facts']}")
    lines += _item_lines(profile.get("problems") or [], "Зоны внимания:")
    lines += _item_lines(profile.get("achievements") or [], "Достижения:")
    secondary = (profile.get("secondary") or [])[:_SECONDARY_CAP]
    if secondary:
        lines.append("Второстепенные сигналы: " + "; ".join(secondary) + ".")
    lines.append(
        "Это опора для всего диалога: числа и выводы из неё не переиначивай; "
        "если свежий разбор аналитика уточняет картину — приоритет у свежих "
        "фактов, но противоречие отметь."
    )
    return "\n".join(lines)


def render_diagnosis_fallback(diagnosis: dict[str, Any]) -> str:
    """Детерминированный компактный блок из меток диагноста — когда LLM-вызов
    формулировок упал (кеш при этом не пишется)."""
    lines = [
        "Опорная диагностика сотрудника относительно коллег (автоматические "
        "метки; служебный блок — пользователю не раскрывай):",
        f"Итоговый статус: {diagnosis.get('overall_status')}.",
    ]
    conf = diagnosis.get("confidence") or {}
    if conf:
        reasons = "; ".join(conf.get("reasons") or [])
        lines.append(
            f"Уверенность сравнения: {conf.get('level')}"
            + (f" ({reasons})" if reasons else "") + "."
        )

    def _line(brief: dict | None, label: str) -> None:
        if not brief:
            return
        lines.append(
            f"{label}: «{brief.get('metric_name')}» — {brief.get('final_class')}."
        )

    _line(diagnosis.get("top_problem"), "Главная зона внимания")
    _line(diagnosis.get("top_achievement"), "Главное достижение")
    secondary = (diagnosis.get("secondary_signals") or [])[:_SECONDARY_CAP]
    if secondary:
        lines.append("Второстепенные сигналы: " + "; ".join(
            f"«{s.get('metric_name')}» ({s.get('kind')})" for s in secondary
        ) + ".")
    return "\n".join(lines)
