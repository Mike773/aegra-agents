"""Промовая реализация agent_dataset.py (корень репо): aggregates_ids.

Структура данных поменялась: у сырой метрики есть aggregates_ids — массив id
предагрегатов, которые надо загрузить индивидуально. convert_metric поле
отбрасывает, поэтому build_json_output собирает его ДО конвертации и отдаёт
уникальным списком на уровне персоны (рядом с metrics).

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


def _raw_metric(metric_id, agg_ids, children=None):
    return {
        "metric_id": metric_id,
        "metric_name": f"Метрика {metric_id}",
        "dt": "2026-07-30",
        "calc_period": "Месяц",
        "fact": 10,
        "plan": 12,
        "aggregates_ids": agg_ids,
        "children_metrics": children or [],
    }


def test_collect_aggregates_ids_walks_tree_and_dedupes():
    metrics = [
        _raw_metric("1", ["182_m_777", "182_m_888"], children=[
            # Дочерний повторяет id родителя и добавляет свой.
            _raw_metric("2", ["182_m_888", "182_m_999"]),
        ]),
        _raw_metric("3", None),          # поля нет/None — не падаем
        _raw_metric("4", [None, " ", 5]),  # мусор чистим, числа приводим к str
    ]
    assert agent_dataset.collect_aggregates_ids(metrics) == [
        "182_m_777", "182_m_888", "182_m_999", "5",
    ]
    assert agent_dataset.collect_aggregates_ids([]) == []


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
    datasets = {
        "me": {"metrics": [_raw_metric("1", ["182_m_777", "182_m_777"])]},
        "111": {"metrics": [
            _raw_metric("1", ["182_m_777"]),
            _raw_metric("2", ["182_m_888"]),
        ]},
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
