"""Короткие служебные тексты, перенесённые из v4."""
from __future__ import annotations

LOAD_ERROR_PROMPT = 'Не удалось загрузить метрики сотрудника. Проверьте, что в configurable переданы boss_tabnum и employee_tabnum.'

SAVE_INSIGHT_AUTO_FIXED = 'Главный вывод по сотруднику фиксируется автоматически при старте разбора (при запуске из карточки встречи). Отдельно сохранять ничего не нужно — можем продолжить разбор показателей.'

__all__ = ["LOAD_ERROR_PROMPT", "SAVE_INSIGHT_AUTO_FIXED"]
