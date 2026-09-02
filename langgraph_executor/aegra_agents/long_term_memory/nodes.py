"""Заглушка промового механизма долгосрочной памяти.

На проме одноимённый граф загружает контекст предыдущих диалогов и сохраняет
итоги текущего. Здесь — тихие no-op узлы с тем же интерфейсом (имена фабрик,
сигнатуры, поля стейта), чтобы структура оркестратора совпадала с промом и
реальная реализация подставлялась без перепайки графа.
"""
from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

# Фильтр в оркестраторе не пропускает контекст со словом «отсутствует» в промпт
# — заглушка ходит по промовому пути «память загружена, но пустая».
EMPTY_MEMORY_CONTEXT = "Долгосрочная память отсутствует."


def make_load_memory_node(llm: Any):
    """Фабрика узла load_memory. ``llm`` не используется заглушкой,
    но принимается для совместимости интерфейса с промом."""

    async def load_memory(state: Any, config: RunnableConfig) -> dict:
        return {
            "memory_context": EMPTY_MEMORY_CONTEXT,
            "memory_error": None,
            "memory_loaded": True,
        }

    return load_memory


def make_save_memory_node():
    """Фабрика узла save_memory: ничего не пишет, зеркалит успешный путь прома."""

    async def save_memory(state: Any, config: RunnableConfig) -> dict:
        return {
            "memory_saved": True,
            "memory_save_error": None,
        }

    return save_memory
