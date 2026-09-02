"""Планировщик составного разбора: брифинг → список задач.

Включается флагом ``dashboard_mode``. Модель раскладывает запрос на несколько
последовательных аналитических задач; результат каждой доступен следующим.
Разбор ответа толерантный: не разобрали — работаем обычным ходом, а не падаем.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

PLANNER_PROMPT = (
    "Ты — планировщик аналитического разбора. Тебе дан запрос руководителя и "
    "состав доступных данных. Разложи запрос на ПОСЛЕДОВАТЕЛЬНЫЕ аналитические "
    "задачи: каждая задача — самостоятельный вопрос по данным, задачи идут в том "
    "порядке, в котором их надо выполнять, и следующая может опираться на "
    "результат предыдущей.\n\n"
    "Правила:\n"
    "- Задач столько, сколько реально просит запрос: обычно 2–5, максимум 8. "
    "Если запрос простой и на него отвечают одним разбором — верни пустой список.\n"
    "- Задача формулируется как вопрос по ДОСТУПНЫМ данным (показатели из состава "
    "данных ниже). Не придумывай показатели, которых нет.\n"
    "- depends_on — идентификаторы задач, без результата которых эту делать "
    "бессмысленно.\n"
    "- metric_hints — названия показателей из состава данных, о которых задача.\n\n"
    "Ответ — СТРОГО JSON без пояснений и без markdown:\n"
    '{"tasks": [{"id": "t1", "title": "…", "question": "…", '
    '"depends_on": [], "metric_hints": ["…"]}]}'
)

_MAX_TASKS = 8
_TITLE_CAP = 200
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[-1] if "\n" in raw else raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_RE.search(raw)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def validate_tasks(payload: Any, *, max_tasks: int = _MAX_TASKS) -> list[dict[str, Any]]:
    """Приводит ответ модели к списку задач: дедуп, нормализация id, кап."""
    if not isinstance(payload, dict):
        return []
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        return []
    out: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for item in raw_tasks:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("question") or "").strip()[:_TITLE_CAP]
        if not title or title.casefold() in seen_titles:
            continue
        seen_titles.add(title.casefold())
        hints = [
            str(h).strip()
            for h in (item.get("metric_hints") or [])
            if str(h or "").strip()
        ]
        out.append(
            {
                "id": f"t{len(out) + 1}",
                "title": title,
                "question": str(item.get("question") or title).strip()[:1000],
                "metric_names": hints,
                "target_value": None,
                "target_kind": None,
                "depends_on": [str(d).strip() for d in (item.get("depends_on") or [])],
            }
        )
        if len(out) >= max_tasks:
            break
    known = {t["id"] for t in out}
    for task in out:
        task["depends_on"] = [d for d in task["depends_on"] if d in known and d != task["id"]]
    return out


def plan_tasks(
    llm: Any, *, briefing: str | None, data_summary: str, max_tasks: int = _MAX_TASKS
) -> list[dict[str, Any]]:
    """Один вызов модели. Любая ошибка → пустой список (обычный ход)."""
    text = (briefing or "").strip()
    if not text:
        return []
    try:
        ai = llm.invoke(
            [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(
                    content=f"Запрос руководителя:\n{text}\n\nСостав данных:\n{data_summary}"
                ),
            ]
        )
    except Exception:  # noqa: BLE001 — LLM-вызов
        logger.warning("planner: вызов модели упал", exc_info=True)
        return []
    content = ai.content if isinstance(ai.content, str) else str(ai.content)
    return validate_tasks(_extract_json(content), max_tasks=max_tasks)


__all__ = ["PLANNER_PROMPT", "plan_tasks", "validate_tasks"]
