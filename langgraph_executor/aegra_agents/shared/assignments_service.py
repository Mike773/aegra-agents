"""Заглушка сервиса фиксации поручений сотрудникам.

Публичный API совпадает с реальным клиентом из рабочего langgraph_executor,
где этот файл вытесняется настоящей реализацией.
"""
from __future__ import annotations

from typing import Any


class SendAssignmentsComponent:
    """Отправка списка поручений (insights) в сервис управления задачами.

    Args:
        employee_tabnum: Табельный номер сотрудника-исполнителя.
        direction_key:   Ключ направления сотрудника.
        insights:        Список наблюдений; каждое — dict с ключами
                         ``type`` / ``metric_id`` / ``metric_name`` / ``text``.
                         ``type`` ∈ {main_problem, problem, norm, achievement}.
    """

    def __init__(
        self,
        employee_tabnum: str,
        direction_key: str,
        insights: list[dict[str, Any]],
    ) -> None:
        self.employee_tabnum = employee_tabnum
        self.direction_key = direction_key
        self.insights = insights

    def submit(self) -> dict[str, Any]:
        return {
            "employee_tabnum": self.employee_tabnum,
            "direction_key": self.direction_key,
            "insights": self.insights,
            "accepted": len(self.insights),
            "_stub": True,
        }
