"""Параметры запуска из ``config["configurable"]``.

Набор ключей и дефолты повторяют analytic_orchestrator_v4 — прод-клиент уже их
шлёт. Новых ровно три: dashboard_mode (составной разбор), text2sql_enabled
(рубильник свободного SQL) и debug_dump (выгрузка базы для отладки).
``format_first_answer`` принимается ради совместимости, но ничего не делает:
правила формата первого ответа теперь в самом промпте. ``run_mode="signal"``
включает сигнальный режим инсайтов (см. ``shared/insight_signal.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig

from ..shared.insight_signal import RUN_MODE_SIGNAL
from .contract.messages import config_flag, config_value

DEFAULT_DATASET = "metrics_for_agent_analyst"
DEFAULT_EASYRAG_TOP_K = 5
DEFAULT_TOOL_BUDGET = 18
# В составном разборе бюджет на КАЖДУЮ задачу меньше: задач несколько.
DEFAULT_TASK_TOOL_BUDGET = 10
MAX_TASKS = 8
# Сколько недостающих трактовок добираем прямо в ходе (остальное — фоновый граф)
DEFAULT_KNOWLEDGE_MAX_NEW = 10
DEFAULT_KNOWLEDGE_TIMEOUT = 25.0


@dataclass
class RunConfig:
    boss_tabnum: str = ""
    employee_tabnum: str = ""
    position: str | None = None
    dataset_name: str = DEFAULT_DATASET
    direction_key: str | None = None
    source_type: str | None = None
    source_id: str | None = None
    thread_id: str | None = None
    run_mode: str | None = None

    easyrag_enabled: bool = True
    easyrag_top_k: int = DEFAULT_EASYRAG_TOP_K
    gap_on_unanswered: bool = True
    use_peer_aggregates: bool = True
    text2sql_enabled: bool = True
    knowledge_enabled: bool = True
    knowledge_max_new: int = DEFAULT_KNOWLEDGE_MAX_NEW
    knowledge_timeout: float = DEFAULT_KNOWLEDGE_TIMEOUT

    emit_progress_messages: bool = True
    describe_answer: bool = False
    answer_html: bool = True
    system_prompt_override: str | None = None

    dashboard_mode: bool = False
    tool_budget: int = DEFAULT_TOOL_BUDGET
    task_tool_budget: int = DEFAULT_TASK_TOOL_BUDGET
    max_tasks: int = MAX_TASKS
    debug_dump: bool = False

    @classmethod
    def from_config(cls, config: RunnableConfig | None) -> "RunConfig":
        def text(key: str) -> str | None:
            value = config_value(config, key)
            if value is None:
                return None
            value = str(value).strip()
            return value or None

        def number(key: str, default: int) -> int:
            value = config_value(config, key)
            try:
                return int(value) if value is not None else default
            except (TypeError, ValueError):
                return default

        return cls(
            boss_tabnum=(text("boss_tabnum") or ""),
            employee_tabnum=(text("employee_tabnum") or ""),
            position=config_value(config, "position"),
            dataset_name=text("dataset_name") or DEFAULT_DATASET,
            direction_key=text("direction_key"),
            source_type=text("source_type"),
            source_id=text("source_id"),
            thread_id=text("thread_id"),
            run_mode=text("run_mode"),
            easyrag_enabled=config_flag(config, "easyrag_enabled", default=True),
            easyrag_top_k=number("easyrag_top_k", DEFAULT_EASYRAG_TOP_K),
            gap_on_unanswered=config_flag(config, "gap_on_unanswered", default=True),
            use_peer_aggregates=config_flag(config, "use_peer_aggregates", default=True),
            text2sql_enabled=config_flag(config, "text2sql_enabled", default=True),
            knowledge_enabled=config_flag(config, "knowledge_enabled", default=True),
            knowledge_max_new=number("knowledge_max_new", DEFAULT_KNOWLEDGE_MAX_NEW),
            knowledge_timeout=float(
                number("knowledge_timeout", int(DEFAULT_KNOWLEDGE_TIMEOUT))
            ),
            emit_progress_messages=config_flag(
                config, "emit_progress_messages", default=True
            ),
            describe_answer=config_flag(config, "describe_answer", default=False),
            answer_html=config_flag(config, "answer_html", default=True),
            system_prompt_override=text("system_prompt_override"),
            dashboard_mode=config_flag(config, "dashboard_mode", default=False),
            tool_budget=number("tool_budget", DEFAULT_TOOL_BUDGET),
            task_tool_budget=number("task_tool_budget", DEFAULT_TASK_TOOL_BUDGET),
            max_tasks=number("max_tasks", MAX_TASKS),
            debug_dump=config_flag(config, "debug_dump", default=False),
        )

    def has_source(self) -> bool:
        """Инсайты пишем только когда известны и тип, и идентификатор источника."""
        return bool(self.source_type and self.source_id)

    def is_signal_mode(self) -> bool:
        """Сигнальный режим: инсайт с вердиктом о проблеме и текстом ответа."""
        return (self.run_mode or "").strip().lower() == RUN_MODE_SIGNAL


def configurable(config: RunnableConfig | None) -> dict[str, Any]:
    return (config or {}).get("configurable") or {}


__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_KNOWLEDGE_MAX_NEW",
    "DEFAULT_KNOWLEDGE_TIMEOUT",
    "DEFAULT_TASK_TOOL_BUDGET",
    "DEFAULT_TOOL_BUDGET",
    "MAX_TASKS",
    "RunConfig",
    "configurable",
]
