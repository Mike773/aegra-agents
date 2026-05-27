"""Заглушка ISU-сервиса оргструктуры.

Публичный API совпадает с реальным клиентом из рабочего langgraph_executor,
где этот файл вытесняется настоящей реализацией.
"""
from __future__ import annotations

from typing import Any


class IsuEmployeeOrgstructureInfo:
    """Информация об оргструктуре пары сотрудник/руководитель.

    Args:
        manager_id:  Табельный номер руководителя.
        position:    Позиция сотрудника (опционально).
        employee_id: Табельный номер сотрудника.
    """

    def __init__(
        self,
        manager_id: str,
        position: str | None,
        employee_id: str,
    ) -> None:
        self.manager_id = manager_id
        self.position = position
        self.employee_id = employee_id

    def employee_description(self) -> str:
        return (
            f"[stub] сотрудник {self.employee_id}, "
            f"позиция {self.position or '-'}, "
            f"руководитель {self.manager_id}"
        )

    def combined_json(self) -> dict[str, Any]:
        return {
            "manager_id": self.manager_id,
            "position": self.position,
            "employee_id": self.employee_id,
            "_stub": True,
        }

    def direction_key(self) -> str:
        # Заглушка: реальная реализация вытесняется в проде.
        # Детерминированный ключ направления — нужен подагентам (например, easyrag)
        # для фильтрации wiki по направлению сотрудника.
        normalized = (self.position or "").strip().lower().replace(" ", "_")
        return normalized or "default"
