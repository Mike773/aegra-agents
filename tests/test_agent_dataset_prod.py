"""Промовая реализация agent_dataset.py (корень репо): aggregates_ids.

Структура данных: _process_person возвращает датасет персоны, где
aggregates_ids — массив id предагрегатов — лежит на верхнем уровне, рядом с
metrics (в самих метриках поля нет). build_json_output нормализует список
(дедуп, строки) и отдаёт у персоны рядом со сконвертированными metrics.

Файл в корне — прод-код с импортом промового `settings`, которого в репо нет:
подставляем модуль-заглушку до импорта.
"""
from __future__ import annotations

import json
import os
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
_settings_stub = types.ModuleType("settings")
_settings_stub.settings = SimpleNamespace(agent_dataset_url="http://test")
sys.modules.setdefault("settings", _settings_stub)

import agent_dataset  # noqa: E402


def _raw_metric(metric_id, children=None):
    return {
        "metric_id": metric_id,
        "metric_name": f"Метрика {metric_id}",
        "dt": "2026-07-30",
        "calc_period": "Месяц",
        "fact": 10,
        "plan": 12,
        "children_metrics": children or [],
    }


def test_normalize_aggregates_ids_dedupes_and_cleans():
    assert agent_dataset.normalize_aggregates_ids(
        ["182_m_777", "182_m_888", "182_m_777", None, " ", 5, "182_m_888"]
    ) == ["182_m_777", "182_m_888", "5"]
    assert agent_dataset.normalize_aggregates_ids([]) == []
    assert agent_dataset.normalize_aggregates_ids(None) == []


def test_build_json_output_returns_ids_at_person_level(monkeypatch):
    input_data = {
        "me": {
            "title": "Босс Борис",
            "addFiltersObjects": [{"object_type": "USER", "object_id": "1"}],
        },
        "employees": [{
            "tabNum": "111",
            "userName": "Иванов Иван",
            "addFiltersObjects": [{"object_type": "USER", "object_id": "2"}],
        }],
    }
    # aggregates_ids — на верхнем уровне датасета, рядом с metrics; дубли и
    # мусор в списке чистятся при выдаче.
    datasets = {
        "me": {
            "metrics": [_raw_metric("1")],
            "aggregates_ids": ["182_m_777", "182_m_777", None, " "],
        },
        "111": {
            "metrics": [_raw_metric("1"), _raw_metric("2")],
            "aggregates_ids": ["182_m_777", "182_m_888"],
        },
    }
    monkeypatch.setattr(
        agent_dataset.GetBatchAgentDatasetByFiltersComponent,
        "_process_person",
        lambda self, client, person_id, filters: datasets.get(person_id),
    )
    out = json.loads(
        agent_dataset.GetBatchAgentDatasetByFiltersComponent(
            "metrics_for_agent_analyst", json.dumps(input_data)
        ).build_json_output()
    )
    # ids — уникальным списком у персоны, рядом с metrics.
    assert out["me"]["aggregates_ids"] == ["182_m_777"]
    assert out["employees"][0]["aggregates_ids"] == ["182_m_777", "182_m_888"]
    # Метрики при этом сконвертированы как раньше (loader-узлы, не raw).
    me_metric = out["me"]["metrics"][0]
    assert me_metric["metric_name"] == "Метрика 1"
    assert me_metric["fact"] == 10
    assert "aggregates_ids" not in me_metric


def test_build_json_output_without_ids_field_gives_empty_list(monkeypatch):
    # Датасет без aggregates_ids (старый инстанс) — пустой список, не падение.
    input_data = {
        "me": {
            "title": "Босс Борис",
            "addFiltersObjects": [{"object_type": "USER", "object_id": "1"}],
        },
    }
    monkeypatch.setattr(
        agent_dataset.GetBatchAgentDatasetByFiltersComponent,
        "_process_person",
        lambda self, client, person_id, filters: {"metrics": [_raw_metric("1")]},
    )
    out = json.loads(
        agent_dataset.GetBatchAgentDatasetByFiltersComponent(
            "metrics_for_agent_analyst", json.dumps(input_data)
        ).build_json_output()
    )
    assert out["me"]["aggregates_ids"] == []
