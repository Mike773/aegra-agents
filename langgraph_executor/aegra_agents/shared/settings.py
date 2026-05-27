"""Конфигурация внешних сервисов. Заглушка — реальные значения подставляет
окружение рабочего langgraph_executor.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    agent_dataset_url: str = ""
    isu_orgstructure_url: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            agent_dataset_url=os.environ.get("AGENT_DATASET_URL", ""),
            isu_orgstructure_url=os.environ.get("ISU_ORGSTRUCTURE_URL", ""),
        )


settings = Settings.from_env()
