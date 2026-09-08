"""Загрузка датасета и предагрегатов — перенос узла из v4.

Клиенты данных синхронные (на проме ходят по HTTP), поэтому весь блокирующий
блок уводится в выделенный пул с таймаутом (shared/offload.run_blocking): иначе
зависший сервис данных подвесил бы весь ход.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from ...shared.agent_dataset import (
    GetBatchAgentAggregateDatasetByFiltersComponent,
    GetBatchAgentDatasetByFiltersComponent,
)
from ...shared.offload import run_blocking
from ...shared.orgstructure import IsuEmployeeOrgstructureInfo
from ..config import RunConfig
from ..contract.messages import last_user_text

logger = logging.getLogger(__name__)


def metrics_obj(metrics: Any) -> dict | None:
    """Датасет как dict: прод-компонент может отдать JSON строкой."""
    if isinstance(metrics, dict):
        return metrics
    if isinstance(metrics, str):
        text = metrics.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def collect_aggregate_ids(metrics: Any) -> list[str]:
    """Уникальные aggregates_ids по всем персонам, в порядке первого появления."""
    data = metrics_obj(metrics)
    if data is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for person in [data.get("me"), *(data.get("employees") or [])]:
        if not isinstance(person, dict):
            continue
        for agg_id in person.get("aggregates_ids") or []:
            if agg_id is None:
                continue
            text = str(agg_id).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


def make_load_data_node():
    async def load_data(state: Any, config: RunnableConfig) -> dict:
        cfg = RunConfig.from_config(config)
        # Первое сообщение треда — брифинг: роль, формат, иногда список задач.
        briefing = (last_user_text(state) or "").strip() or None
        base = {
            "briefing": briefing,
            "source_type": cfg.source_type,
            "source_id": cfg.source_id,
            "dashboard_mode": cfg.dashboard_mode,
            "reasoning_trace": [],
            "turn_no": 1,
            "loaded": True,
        }

        if not cfg.boss_tabnum and not cfg.employee_tabnum:
            return {
                **base,
                "metrics": None,
                "metrics_error": (
                    "В configurable нет boss_tabnum/employee_tabnum: "
                    f"boss_tabnum={cfg.boss_tabnum!r}, "
                    f"employee_tabnum={cfg.employee_tabnum!r}."
                ),
            }

        def _fetch() -> tuple[str, Any, str | None, Any, str | None]:
            orgstructure = IsuEmployeeOrgstructureInfo(
                manager_id=cfg.boss_tabnum,
                position=cfg.position,
                employee_id=cfg.employee_tabnum,
            )
            dkey = orgstructure.direction_key()
            filters = orgstructure.combined_json()
            try:
                metrics = GetBatchAgentDatasetByFiltersComponent(
                    dataset_name=cfg.dataset_name, filters=filters
                ).build_json_output()
            except Exception as exc:  # noqa: BLE001 — внешний клиент
                return (
                    dkey, None,
                    f"Ошибка при загрузке датасета {cfg.dataset_name!r}: {exc}",
                    None, None,
                )
            if not cfg.use_peer_aggregates:
                return dkey, metrics, None, None, None

            # Предагрегаты индивидуальны: какие грузить, говорит сам датасет.
            # Сбой по одному id не роняет ни загрузку, ни остальные.
            aggregates: list | None = None
            agg_errors: list[str] = []
            for agg_id in collect_aggregate_ids(metrics):
                try:
                    batch = GetBatchAgentAggregateDatasetByFiltersComponent(
                        dataset_name=cfg.dataset_name,
                        filters=[{"object_type": "aggregate_id", "object_id": agg_id}],
                    ).build_json_output()
                except Exception as exc:  # noqa: BLE001 — внешний клиент
                    agg_errors.append(f"{agg_id}: {exc}")
                    continue
                if isinstance(batch, list):
                    for rec in batch:
                        if isinstance(rec, dict):
                            rec["aggregate_id"] = agg_id
                    aggregates = (aggregates or []) + batch
            agg_error = (
                f"Ошибка при загрузке агрегатов {cfg.dataset_name!r}: "
                + "; ".join(agg_errors)[:300]
                if agg_errors
                else None
            )
            return dkey, metrics, None, aggregates, agg_error

        try:
            direction_key, metrics, error, aggregates, agg_error = await run_blocking(_fetch)
        except asyncio.TimeoutError:
            direction_key, metrics, aggregates, agg_error = "", None, None, None
            error = "Сервис данных (оргструктура/датасет) не ответил за отведённое время."
        except Exception as exc:  # noqa: BLE001 — сбой оргструктуры до датасета
            logger.warning("load_data: внешний сервис упал", exc_info=True)
            direction_key, metrics, aggregates, agg_error = "", None, None, None
            error = f"Ошибка внешнего сервиса данных: {type(exc).__name__}: {exc}"[:300]

        return {
            **base,
            "boss_tabnum": cfg.boss_tabnum,
            "employee_tabnum": cfg.employee_tabnum,
            "position": cfg.position,
            "direction_key": cfg.direction_key or direction_key,
            "metrics": metrics_obj(metrics) or metrics,
            "metrics_error": error,
            "aggregates": aggregates,
            "aggregates_error": agg_error,
        }

    return load_data


__all__ = ["collect_aggregate_ids", "make_load_data_node", "metrics_obj"]
