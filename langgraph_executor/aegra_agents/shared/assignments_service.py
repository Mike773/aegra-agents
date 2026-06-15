"""Заглушка сервиса фиксации поручений сотрудникам.

Публичный API совпадает с реальным клиентом из рабочего langgraph_executor,
где этот файл вытесняется настоящей реализацией.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


class SendAssignmentsComponent:
    """Отправка списка поручений (insights) в сервис управления задачами.

    Args:
        boss_tabnum:     Табельный номер руководителя (в payload — ``subject_id``).
        employee_tabnum: Табельный номер сотрудника-исполнителя (``object_id``).
        direction_key:   Ключ направления сотрудника (для роутинга на стороне сервиса).
        thread_id:       Идентификатор треда aegra из ``/threads`` (``session_id``).
        insights:        Список наблюдений; каждое — dict с ключами
                         ``type`` / ``metric_id`` / ``metric_name`` / ``text``.
                         ``type`` ∈ {main_problem, problem, norm, achievement}.
    """

    def __init__(
        self,
        boss_tabnum: str,
        employee_tabnum: str,
        direction_key: str,
        thread_id: str,
        insights: list[dict[str, Any]],
    ) -> None:
        self.boss_tabnum = boss_tabnum
        self.employee_tabnum = employee_tabnum
        self.direction_key = direction_key
        self.thread_id = thread_id
        self.insights = insights

    def submit(self) -> dict[str, Any]:
        return {
            "title": "agent_analyst_insights",
            "content": {"insights": self.insights},
            "object_id": self.employee_tabnum,
            "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
            "session_id": self.thread_id,
            "subject_id": self.boss_tabnum,
        }
