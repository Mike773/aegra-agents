"""Заглушка сервиса фиксации поручений сотрудникам.

Публичный API совпадает с реальным клиентом из рабочего langgraph_executor,
где этот файл вытесняется настоящей реализацией.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


class SendAssignmentsComponent:
    """Отправка поручений (инсайтов) в сервис управления задачами.

    Args:
        boss_tabnum:     Табельный номер руководителя (в payload — ``subject_id``).
        employee_tabnum: Табельный номер сотрудника-исполнителя (``object_id``).
        direction_key:   Ключ направления сотрудника (для роутинга на стороне сервиса).
        thread_id:       Идентификатор треда aegra из ``/threads`` (``session_id``).
        insight:         ОДИН главный инсайт — dict с ключами
                         ``type`` / ``metric_id`` / ``metric_name`` / ``text``,
                         ``type`` ∈ {main_problem, problem, norm, achievement}.
                         Новый контракт сервиса (v4): в payload идёт объект
                         ``content.insight``, не массив.
        insights:        Легаси-список наблюдений той же формы (v1–v3): в payload
                         идёт массив ``content.insights``. Используется, только
                         когда ``insight`` не передан.
        source_type:     Тип сущности вызывающей системы, к которой привязан инсайт.
        source_id:       Её идентификатор. Оба опциональны и попадают в payload,
                         только когда заданы: старые оркестраторы их не передают.
    """

    def __init__(
        self,
        boss_tabnum: str,
        employee_tabnum: str,
        direction_key: str,
        thread_id: str,
        insights: list[dict[str, Any]] | None = None,
        source_type: str | None = None,
        source_id: str | None = None,
        insight: dict[str, Any] | None = None,
    ) -> None:
        self.boss_tabnum = boss_tabnum
        self.employee_tabnum = employee_tabnum
        self.direction_key = direction_key
        self.thread_id = thread_id
        self.insights = insights
        self.insight = insight
        self.source_type = source_type
        self.source_id = source_id

    def submit(self) -> dict[str, Any]:
        content = (
            {"insight": self.insight} if self.insight is not None
            else {"insights": self.insights}
        )
        payload = {
            "title": "agent_analyst_insights",
            "content": content,
            "object_id": self.employee_tabnum,
            "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
            "session_id": self.thread_id,
            "subject_id": self.boss_tabnum,
        }
        if self.source_type:
            payload["source_type"] = self.source_type
        if self.source_id:
            payload["source_id"] = self.source_id
        return payload
