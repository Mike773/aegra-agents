"""Заглушка сервиса фиксации поручений сотрудникам.

Публичный API совпадает с реальным клиентом из рабочего langgraph_executor,
где этот файл вытесняется настоящей реализацией.
"""
from __future__ import annotations

from typing import Any


class SendAssignmentsComponent:
    """Отправка списка поручений в сервис управления задачами.

    Args:
        boss_tabnum:     Табельный номер руководителя (автор поручения).
        employee_tabnum: Табельный номер сотрудника-исполнителя.
        assignments:     Список поручений; каждое — dict с ключами
                         ``title`` / ``problem`` / ``action``.
    """

    def __init__(
        self,
        boss_tabnum: str,
        employee_tabnum: str,
        assignments: list[dict[str, Any]],
    ) -> None:
        self.boss_tabnum = boss_tabnum
        self.employee_tabnum = employee_tabnum
        self.assignments = assignments

    def submit(self) -> dict[str, Any]:
        return {
            "boss_tabnum": self.boss_tabnum,
            "employee_tabnum": self.employee_tabnum,
            "assignments": self.assignments,
            "accepted": len(self.assignments),
            "_stub": True,
        }
