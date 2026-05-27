"""Заглушка агентского dataset-сервиса.

Публичный API совпадает с реальным клиентом из рабочего langgraph_executor,
где этот файл вытесняется настоящей реализацией.
"""
from __future__ import annotations

from typing import Any


class GetBatchAgentDatasetByFiltersComponent:
    """Запрос датасета `dataset_name` с фильтрами `filters`.

    Args:
        dataset_name: Имя датасета (например, `metrics_for_agent_analyst`).
        filters:      JSON-фильтры (обычно — результат
                      `IsuEmployeeOrgstructureInfo.combined_json()`).
    """

    def __init__(
        self,
        dataset_name: str,
        filters: dict[str, Any] | str,
    ) -> None:
        self.dataset_name = dataset_name
        self.filters = filters

    def build_json_output(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "filters": self.filters,
            "_stub": True,
            "rows": [],
        }
