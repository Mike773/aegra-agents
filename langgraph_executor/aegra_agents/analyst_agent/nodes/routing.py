"""Маршрутизация графа — чистые функции от состояния, без LLM.

Роутера-модели больше нет: ветка хода определяется тем, загружены ли данные и
включён ли составной разбор, а не классификацией реплики.
"""
from __future__ import annotations

from typing import Any


def need_load(state: Any) -> str:
    """Первый ход треда грузит данные; дальше состояние берётся из чекпойнтера."""
    return "begin_turn" if (state or {}).get("loaded") else "load_data"


def after_prepare(state: Any) -> str:
    """После подготовки: составной разбор, обычный ход или сразу конец."""
    state = state or {}
    if state.get("metrics_error"):
        # Данных нет — ответ про ошибку уже положен, идём к выходу.
        return "auto_insight"
    if state.get("dashboard_mode"):
        return "plan_tasks"
    return "agent"


def after_agent(state: Any) -> str:
    """После ответа агента: следующая задача, сводка, инсайт или память."""
    state = state or {}
    if state.get("turn_kind") == "dashboard":
        tasks = state.get("tasks") or []
        if int(state.get("task_idx") or 0) < len(tasks):
            return "agent"
        return "summarize"
    if state.get("turn_kind") == "followup":
        return "save_memory"
    return "auto_insight"


__all__ = ["after_agent", "after_prepare", "need_load"]
