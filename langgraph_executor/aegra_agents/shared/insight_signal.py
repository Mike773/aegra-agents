"""Сигнальный режим инсайтов (``configurable.run_mode = "signal"``).

Общая логика для analytic_orchestrator_v4 и analyst_agent: в этом режиме инсайт
уходит в сервис с дополнительными полями — ``author``, ``confirmed``,
``signal`` (есть ли проблема относительно входного запроса) и
``signal_description`` (текст ответа агента). Вердикт ``signal`` даёт
отдельный короткий LLM-вызов по вопросу руководителя и ответу агента; при сбое
модели берётся fallback, который вычисляет вызывающая сторона по типу инсайта.

Поля кладутся внутрь словаря инсайта: контракт ``SendAssignmentsComponent`` не
меняется, сервис получает их в ``content.insight``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

RUN_MODE_SIGNAL = "signal"
DEFAULT_INSIGHT_AUTHOR = "Agent"
EMPTY_INSIGHT_TEXT = "Существенных отклонений по запросу не выявлено."

SIGNAL_VERDICT_PROMPT = (
    "Ты — контролёр качества аналитики. Тебе даны запрос руководителя и ответ "
    "аналитика по показателям сотрудника. Определи, выявил ли ответ ПРОБЛЕМУ, "
    "относящуюся к запросу: невыполнение плана, негативную динамику, отставание "
    "от коллег, риск по задаче — то, что требует внимания руководителя.\n\n"
    "Верни строго JSON без пояснений: {\"signal\": true} — если проблема по "
    "запросу найдена, {\"signal\": false} — если показатели в норме, есть только "
    "достижения или данных недостаточно для вывода о проблеме."
)

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_TRUE_WORDS = ("true", "да")
_FALSE_WORDS = ("false", "нет")


def is_signal_mode(config: Any) -> bool:
    """``run_mode`` сравнивается без учёта регистра и пробелов по краям."""
    cfg = (config or {}).get("configurable") or {}
    return str(cfg.get("run_mode") or "").strip().lower() == RUN_MODE_SIGNAL


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in _TRUE_WORDS:
            return True
        if low in _FALSE_WORDS:
            return False
    return None


def parse_signal_verdict(content: Any) -> bool | None:
    """JSON ``{"signal": ...}`` из ответа модели; иначе — первое слово-вердикт.

    ``None`` — разобрать не удалось, вызывающая сторона берёт fallback.
    """
    text = content if isinstance(content, str) else str(content or "")
    for match in _JSON_RE.finditer(text):
        try:
            data = json.loads(match.group(0))
        except ValueError:
            continue
        if isinstance(data, dict) and "signal" in data:
            verdict = _coerce_bool(data["signal"])
            if verdict is not None:
                return verdict
    for word in re.findall(r"[a-zа-яё]+", text.lower()):
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    return None


async def detect_signal(
    llm: Any, question: str, answer: str, *, fallback: bool
) -> bool:
    """Вердикт «проблема по запросу найдена». Любой сбой → ``fallback``."""
    parts = []
    if (question or "").strip():
        parts.append(f"Запрос руководителя:\n{question.strip()}")
    parts.append(f"Ответ аналитика:\n{answer}")
    try:
        ai = await asyncio.to_thread(
            llm.invoke,
            [
                SystemMessage(content=SIGNAL_VERDICT_PROMPT),
                HumanMessage(content="\n\n".join(parts)),
            ],
        )
        verdict = parse_signal_verdict(getattr(ai, "content", ai))
    except Exception:  # noqa: BLE001 — LLM-вызов, сужать нечем
        logger.warning("Вердикт signal не получен, беру fallback", exc_info=True)
        return fallback
    if verdict is None:
        logger.warning("Вердикт signal не разобран, беру fallback")
        return fallback
    return verdict


def empty_norm_insight() -> dict[str, Any]:
    """Инсайт-заглушка «всё в норме», когда главный вывод не выделился."""
    return {
        "type": "norm",
        "metric_id": None,
        "metric_name": None,
        "text": EMPTY_INSIGHT_TEXT,
    }


def signal_insight(
    insight: dict[str, Any],
    *,
    signal: bool,
    description: str,
    author: str = DEFAULT_INSIGHT_AUTHOR,
) -> dict[str, Any]:
    """Копия инсайта с полями сигнального режима; исходный словарь не трогаем."""
    return {
        **insight,
        "author": author,
        "confirmed": True,
        "signal": bool(signal),
        "signal_description": description,
    }


__all__ = [
    "DEFAULT_INSIGHT_AUTHOR",
    "EMPTY_INSIGHT_TEXT",
    "RUN_MODE_SIGNAL",
    "SIGNAL_VERDICT_PROMPT",
    "detect_signal",
    "empty_norm_insight",
    "is_signal_mode",
    "parse_signal_verdict",
    "signal_insight",
]
