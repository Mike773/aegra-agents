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


def _demo_aggregates(dt: str, mean_fact: float) -> dict[str, Any]:
    return {
        "cv": 75.5,
        "dt": dt,
        "iqr": 8442485,
        "median": 14894246,
        "mean_ex": 102.1,
        "hit_rate": 43.8,
        "mean_fact": mean_fact,
        "mean_plan": 15995040,
        "calc_period": "Месяц",
        "total_objects": 496,
        "top20_mean_fact": 30322068,
    }


class GetBatchAgentAggregateDatasetByFiltersComponent:
    """Запрос batch-агрегатов peer-групп по уровням ORG/TERR/OFFICE.

    Args: как у ``GetBatchAgentDatasetByFiltersComponent`` — те же
    ``dataset_name`` и ``filters``.

    Ответ: список ``{"dataset": {"level", "metrics": [...]}}``, у метрики —
    ``aggregates`` (текущий срез + ``history``) и дети в ``children_metrics``.
    """

    def __init__(
        self,
        dataset_name: str,
        filters: dict[str, Any] | str,
    ) -> None:
        self.dataset_name = dataset_name
        self.filters = filters

    def build_json_output(self) -> list[dict[str, Any]]:
        # Демо-ответ в форме прод-сервиса, чтобы загрузку peer_aggregates можно
        # было прогнать end-to-end без реального клиента.
        metric = {
            "metric_id": "377722562",
            "metric_name": "Метрика тест",
            "aggregates": {
                **_demo_aggregates("2026-07-30", 16335807),
                "history": [
                    _demo_aggregates("2026-05-31", 33313488),
                    _demo_aggregates("2026-04-30", 38507555),
                ],
            },
            "children_metrics": [
                {
                    "metric_id": "350022538",
                    "metric_name": "Пример дочерней метрики",
                    "aggregates": {
                        **_demo_aggregates("2026-07-30", 7331432),
                        "history": [
                            _demo_aggregates("2026-05-31", 12108343),
                            _demo_aggregates("2026-04-30", 15034855),
                        ],
                    },
                }
            ],
        }
        return [
            {"dataset": {"level": level, "metrics": [metric]}}
            for level in ("ORG", "TERR", "OFFICE")
        ]
