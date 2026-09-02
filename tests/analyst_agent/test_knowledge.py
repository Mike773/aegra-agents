"""Кэш трактовок показателей: репозиторий, поиск в wiki, ленивое дозаполнение.

Постгрес в тестах не поднимаем — сессия фейковая: проверяем, ЧТО именно мы
записываем и как ведём себя при пустом поиске и падении модели.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from _fixtures import make_dataset_obj, make_metric  # noqa: F401

from langgraph_executor.aegra_agents.analyst_agent.db import core, loader
from langgraph_executor.aegra_agents.metric_knowledge import lookup, repository
from langgraph_executor.aegra_agents.metric_knowledge.types import CatalogEntry, KnowledgeRow


class FakeSession:
    """Сессия, записывающая выполненные операции и отдающая заготовленные строки."""

    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = []
        self.flushed = 0

    async def execute(self, statement, *args, **kwargs):
        self.executed.append(statement)

        class Result:
            def __init__(self, rows):
                self._rows = rows

            def all(self):
                return self._rows

            def scalars(self):
                return self

            def first(self):
                return self._rows[0] if self._rows else None

        return Result(self.rows)

    async def flush(self):
        self.flushed += 1


def _entry(name="AHT", description="Среднее время обработки"):
    return CatalogEntry(
        metric_key=loader.metric_key(name, description),
        name_key=loader.name_key(name),
        metric_name=name,
        metric_description=description,
        direction="обратная",
        unit="сек",
        depth=1,
    )


# --- каталог и репозиторий -------------------------------------------------

def test_catalog_entries_from_db():
    db = core.build_run_db(make_dataset_obj([
        make_metric("AHT", fact=10.0, description="Среднее время"),
        make_metric("Звонки", fact=5.0, description="Число звонков"),
    ]))
    entries = repository.catalog_entries(db)
    names = {e.metric_name for e in entries}
    assert names == {"AHT", "Звонки"}
    assert all(e.metric_key and e.name_key for e in entries)


def test_upsert_catalog_writes_all_entries():
    session = FakeSession()
    asyncio.run(
        repository.upsert_catalog(session, direction_key="d", entries=[_entry(), _entry("Звонки")])
    )
    assert len(session.executed) == 2
    assert session.flushed == 1


def test_get_knowledge_prefers_exact_key_over_name_fallback():
    exact = KnowledgeRow(
        metric_key="k1", name_key="n1", metric_name="AHT", summary="точное",
        confidence=0.9, status="found",
    )
    by_name = KnowledgeRow(
        metric_key="other", name_key="n1", metric_name="AHT", summary="по имени",
        confidence=0.9, status="found",
    )
    merged = repository.merge_rows(
        exact_rows=[exact], name_rows=[by_name], metric_keys=["k1"], name_keys=["n1"]
    )
    assert merged["k1"].summary == "точное"
    assert merged["k1"].confidence == 0.9


def test_name_fallback_lowers_confidence():
    by_name = KnowledgeRow(
        metric_key="other", name_key="n1", metric_name="AHT", summary="по имени",
        confidence=1.0, status="found",
    )
    merged = repository.merge_rows(
        exact_rows=[], name_rows=[by_name], metric_keys=["k1"], name_keys=["n1"]
    )
    # Описание показателя изменилось — доверие к найденной по имени трактовке ниже.
    assert merged["k1"].confidence < 1.0
    assert merged["k1"].summary == "по имени"


def test_missing_entries_detect_absent_and_stale():
    fresh = datetime.now(timezone.utc)
    old = fresh - timedelta(days=30)
    known = {
        "k_found": KnowledgeRow(metric_key="k_found", name_key="n", metric_name="A",
                                status="found", version=1, updated_at=fresh),
        "k_unknown_old": KnowledgeRow(metric_key="k_unknown_old", name_key="n",
                                      metric_name="B", status="unknown",
                                      version=1, updated_at=old),
        "k_unknown_new": KnowledgeRow(metric_key="k_unknown_new", name_key="n",
                                      metric_name="C", status="unknown",
                                      version=1, updated_at=fresh),
        "k_old_version": KnowledgeRow(metric_key="k_old_version", name_key="n",
                                      metric_name="D", status="found",
                                      version=0, updated_at=fresh),
    }
    catalog = [
        CatalogEntry(metric_key=key, name_key="n", metric_name=key)
        for key in ("k_found", "k_unknown_old", "k_unknown_new", "k_old_version", "k_absent")
    ]
    missing = {e.metric_key for e in repository.missing_entries(catalog, known, version=1)}
    assert missing == {"k_absent", "k_unknown_old", "k_old_version"}


# --- поиск трактовки -------------------------------------------------------

def test_lookup_queries_include_name_and_abbreviations():
    queries = lookup.build_queries(_entry("AHT и FCR", "Среднее время обработки"))
    assert any("AHT и FCR" in q for q in queries)
    assert any("AHT" in q and "что такое" in q.lower() for q in queries)
    assert any("FCR" in q and "что такое" in q.lower() for q in queries)
    assert len(queries) <= 4


def test_lookup_writes_unknown_when_nothing_found():
    row = asyncio.run(
        lookup.build_row(
            _entry(), snippets=[], llm=None, model_name="test",
        )
    )
    assert row is not None
    assert row.status == "unknown"
    assert row.confidence == 0.0
    # Вид показателя проставляется даже без находок — он выводится из названия.
    assert row.rules.get("kind") in ("уровень", "вклад", "индекс")


def test_lookup_returns_none_when_llm_fails():
    class BrokenLLM:
        async def call_json(self, **kwargs):
            raise RuntimeError("нет сети")

    row = asyncio.run(
        lookup.build_row(
            _entry(),
            snippets=[{"section_id": "s1", "body_md": "AHT — среднее время обработки",
                       "similarity": 0.9}],
            llm=BrokenLLM(),
            model_name="test",
        )
    )
    # Транзиентная ошибка не должна отравлять кэш: не пишем ничего.
    assert row is None


def test_lookup_builds_row_from_llm_answer():
    class FakeLLM:
        async def call_json(self, **kwargs):
            return {
                "summary": "Среднее время обработки обращения",
                "aliases": ["AHT", "среднее время"],
                "kind": "уровень",
                "cumulative": False,
                "plan_semantics": "план задаётся сверху вниз",
                "processes": ["обслуживание"],
                "confidence": 0.8,
            }

    row = asyncio.run(
        lookup.build_row(
            _entry(),
            snippets=[{"section_id": "11111111-1111-1111-1111-111111111111",
                       "body_md": "текст", "similarity": 0.8}],
            llm=FakeLLM(),
            model_name="test",
        )
    )
    assert row.status == "found"
    assert row.summary.startswith("Среднее время")
    assert "AHT" in row.aliases
    assert row.rules["kind"] == "уровень"
    assert row.rules["cumulative"] is False
    assert row.confidence == pytest.approx(0.8)
    assert row.source_section_ids


def test_lookup_validates_kind_enum():
    class WeirdLLM:
        async def call_json(self, **kwargs):
            return {"summary": "…", "kind": "непонятно", "confidence": 5}

    row = asyncio.run(
        lookup.build_row(_entry(), snippets=[{"section_id": "s", "body_md": "t",
                                              "similarity": 0.9}],
                         llm=WeirdLLM(), model_name="test")
    )
    assert row.rules["kind"] == "уровень"     # неизвестное значение → дефолт
    assert 0.0 <= row.confidence <= 1.0


# --- применение к базе -----------------------------------------------------

def test_knowledge_applied_to_metric_table():
    db = core.build_run_db(make_dataset_obj([
        make_metric("AHT", fact=10.0, description="Среднее время")
    ]))
    key = loader.metric_key("AHT", "Среднее время")
    rows = {
        key: KnowledgeRow(
            metric_key=key, name_key=loader.name_key("AHT"), metric_name="AHT",
            summary="Среднее время обработки", aliases=["AHT"],
            rules={"kind": "уровень", "cumulative": True, "plan_semantics": "сверху"},
            status="found", confidence=0.8,
        )
    }
    repository.apply_knowledge(db, rows)
    row = db.conn.execute(
        "SELECT kn_summary, kn_aliases, kn_cumulative, kn_status, kind FROM metric"
    ).fetchone()
    assert row["kn_summary"] == "Среднее время обработки"
    assert "AHT" in row["kn_aliases"]
    assert row["kn_cumulative"] == 1
    assert row["kn_status"] == "found"
    assert row["kind"] == "уровень"


def test_knowledge_block_for_prompt():
    db = core.build_run_db(make_dataset_obj([
        make_metric("AHT", fact=10.0, description="Среднее время")
    ]))
    key = loader.metric_key("AHT", "Среднее время")
    repository.apply_knowledge(db, {
        key: KnowledgeRow(metric_key=key, name_key=loader.name_key("AHT"),
                          metric_name="AHT", summary="Среднее время обработки",
                          aliases=["AHT"], rules={"cumulative": True},
                          status="found", confidence=0.8)
    })
    text = repository.knowledge_block(db)
    assert "AHT" in text
    assert "накопительн" in text.lower()
    # Без знаний блок пустой — в промпт ничего лишнего не идёт.
    empty = core.build_run_db(make_dataset_obj([make_metric("Другая", fact=1.0)]))
    assert repository.knowledge_block(empty) == ""
