"""Фоновый обогатитель трактовок: топология и поведение узлов.

Postgres и модель не поднимаем — проверяем маршруты и то, что сбои изолированы.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from langgraph_executor.aegra_agents.metric_enricher import nodes  # noqa: E402
from langgraph_executor.aegra_agents.metric_enricher.graph import build_graph  # noqa: E402


def _edges():
    compiled = build_graph()
    return {(e.source, e.target) for e in compiled.get_graph().edges}


def test_topology():
    edges = _edges()
    assert ("__start__", "load_catalog") in edges
    assert ("load_catalog", "enrich") in edges
    assert ("load_catalog", "finalize") in edges
    assert ("enrich", "finalize") in edges
    assert ("finalize", "__end__") in edges


def test_after_load_skips_enrich_when_nothing_pending():
    assert nodes.after_load({"pending": []}) == "finalize"
    assert nodes.after_load({"pending": [{"metric_name": "AHT"}]}) == "enrich"


def test_load_catalog_requires_direction_key():
    out = asyncio.run(nodes.load_catalog({}, {"configurable": {}}))
    assert out["pending"] == []
    assert any("direction_key" in e for e in out["errors"])


def test_load_catalog_survives_db_failure(monkeypatch):
    class BrokenScope:
        async def __aenter__(self):
            raise RuntimeError("нет базы")

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(nodes, "session_scope", lambda: BrokenScope())
    out = asyncio.run(
        nodes.load_catalog({}, {"configurable": {"direction_key": "test-dir"}})
    )
    assert out["pending"] == []
    assert out["errors"]


def test_finalize_reports_nothing_to_do():
    out = asyncio.run(nodes.finalize({"pending": []}, {}))
    assert "нечего" in out["report"]
    assert out["messages"][0].content == out["report"]


def test_finalize_reports_counts():
    out = asyncio.run(
        nodes.finalize(
            {"pending": [{}, {}, {}], "found": 2, "unknown": 1, "errors": []}, {}
        )
    )
    assert "Взято в работу: 3" in out["report"]
    assert "найдено трактовок: 2" in out["report"]
    assert "в базе знаний нет: 1" in out["report"]


def test_config_resolution_order():
    state = {"top_k": 9}
    config = {"configurable": {"top_k": 3}}
    assert nodes._cfg(config, state, "top_k", 6) == 3        # configurable важнее
    assert nodes._cfg({}, state, "top_k", 6) == 9            # затем state
    assert nodes._cfg({}, {}, "top_k", 6) == 6               # затем дефолт


def test_registered_in_link_and_aegra_json():
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    assert "metric_enricher_graph" in (repo / "link.py").read_text(encoding="utf-8")
    aegra = json.loads((repo / "aegra.json").read_text(encoding="utf-8"))
    assert aegra["graphs"]["metric_enricher"] == "./link.py:metric_enricher_graph"


def test_migration_is_idempotent_by_construction():
    from pathlib import Path

    sql = (
        Path(__file__).resolve().parents[2]
        / "migrations" / "wiki_rag" / "0003_metric_knowledge.sql"
    ).read_text(encoding="utf-8")
    creates = [
        ln.strip()
        for ln in sql.splitlines()
        if ln.strip().upper().startswith(("CREATE TABLE", "CREATE INDEX", "CREATE UNIQUE"))
    ]
    assert creates
    for line in creates:
        assert "IF NOT EXISTS" in line.upper(), line
