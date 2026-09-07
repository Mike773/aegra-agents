"""gap_resolver: разбор пробелов по всем чанкам, без векторного поиска.

Эмбеддинг чанков убран: отбирать кандидатов больше нечем, поэтому судья
проверяет КАЖДЫЙ чанк направления отдельным вызовом. Отсюда три ограничителя —
ранняя остановка на найденном ответе, кап чанков на пробел и общий кап вызовов
судьи на прогон: без них один неотвечаемый вопрос стоил бы столько вызовов,
сколько чанков в направлении.

Postgres и модель не поднимаем: сессия и судья подменяются.
"""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from uuid import uuid4

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

import pytest  # noqa: E402

from langgraph_executor.aegra_agents.gap_resolver import nodes  # noqa: E402
from langgraph_executor.aegra_agents.gap_resolver.judge import Verdict  # noqa: E402
from langgraph_executor.aegra_agents.gap_resolver.retrieval import SourceMatch  # noqa: E402

DIRECTION = "test-dir"


@dataclass
class FakeGap:
    id: object
    direction_key: str = DIRECTION
    resolved_at: object = None


class FakeSession:
    """Сессия, отдающая заготовленный QueryGap и глотающая остальное."""

    def __init__(self, gap: FakeGap | None):
        self.gap = gap

    async def get(self, _model, _pk):
        return self.gap

    async def execute(self, *_args, **_kwargs):
        class Result:
            def all(self):
                return []

            def scalars(self):
                return self

            def first(self):
                return None

        return Result()

    async def flush(self):
        return None


class FakeScope:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *args):
        return False


def _chunk(ord_: int, text: str, uri: str = "doc.md") -> SourceMatch:
    return SourceMatch(
        chunk_id=uuid4(), doc_id=uuid4(), uri=uri, ord=ord_, text=text
    )


def _setup(
    monkeypatch,
    *,
    chunks: list[SourceMatch],
    verdicts,
    gap: FakeGap | None = None,
    topics=("тема",),
):
    """Подменяет всё внешнее: сессию, загрузку чанков, судью, темы, заглушки."""
    calls: list[str] = []
    gap = gap if gap is not None else FakeGap(id=uuid4())

    monkeypatch.setattr(nodes, "session_scope", lambda: FakeScope(FakeSession(gap)))
    monkeypatch.setattr(nodes, "UUID", lambda value: value)

    async def fake_load(session, *, direction_key, limit):
        return chunks[:limit]

    async def fake_judge(query, texts, *, llm=None):
        assert len(texts) == 1, "судья должен получать по одному чанку за вызов"
        calls.append(texts[0])
        verdict = verdicts(texts[0]) if callable(verdicts) else verdicts
        return verdict

    async def fake_topics(queries, *, llm=None):
        return list(topics)[: len(queries)] or ["тема"]

    async def fake_stub(session, *, direction_key, name):
        return None

    marked: list[list[str]] = []

    async def fake_mark(session, ids):
        marked.append(list(ids))

    monkeypatch.setattr(nodes, "load_source_chunks", fake_load)
    monkeypatch.setattr(nodes, "answer_in_sources", fake_judge)
    monkeypatch.setattr(nodes, "extract_topics", fake_topics)
    monkeypatch.setattr(nodes, "ensure_stub_page", fake_stub)
    monkeypatch.setattr(nodes, "mark_groups_resolved", fake_mark)
    monkeypatch.setattr(nodes, "LLMClient", lambda *a, **kw: object())
    return calls, marked


def _state(n_gaps: int = 1) -> dict:
    return {
        "direction_key": DIRECTION,
        "gaps": [
            {"id": str(uuid4()), "query": f"вопрос {i}", "ids": [str(uuid4())]}
            for i in range(n_gaps)
        ],
    }


def _cfg(**kw) -> dict:
    return {"configurable": {"direction_key": DIRECTION, **kw}}


# --- судья идёт по одному чанку ------------------------------------------

def test_judge_called_once_per_chunk(monkeypatch):
    chunks = [_chunk(i, f"текст {i}") for i in range(5)]
    calls, _ = _setup(monkeypatch, chunks=chunks, verdicts=Verdict(found=False))
    out = asyncio.run(nodes.investigate(_state(), _cfg()))
    assert len(calls) == 5
    assert calls == [c.text for c in chunks]
    assert len(out["unresolved"]) == 1


def test_no_embedding_client_used(monkeypatch):
    """Векторов больше нет: клиент эмбеддингов в узле не создаётся."""
    assert not hasattr(nodes, "EmbeddingClient")


def test_early_stop_after_answer_found(monkeypatch):
    chunks = [_chunk(i, f"текст {i}") for i in range(20)]

    def verdicts(text):
        return Verdict(found=True, quote="цитата") if text == "текст 2" else Verdict(found=False)

    calls, marked = _setup(monkeypatch, chunks=chunks, verdicts=verdicts)
    out = asyncio.run(nodes.investigate(_state(), _cfg(judge_concurrency=1)))
    assert len(out["resolved"]) == 1
    # После находки оставшиеся чанки не проверяются.
    assert len(calls) <= 4, calls
    assert marked, "решённый пробел должен помечаться"


def test_evidence_points_to_source_not_similarity(monkeypatch):
    chunks = [_chunk(0, "нужный текст", uri="справочник.md")]
    _setup(monkeypatch, chunks=chunks,
           verdicts=Verdict(found=True, quote="дословная цитата"))
    out = asyncio.run(nodes.investigate(_state(), _cfg()))
    ev = out["resolved"][0]["evidence"]
    assert ev["uri"] == "справочник.md"
    assert ev["ord"] == 0
    assert "дословная цитата" in ev["preview"]
    assert "similarity" not in ev


# --- ограничители ---------------------------------------------------------

def test_max_chunks_per_gap_caps_scan(monkeypatch):
    chunks = [_chunk(i, f"текст {i}") for i in range(50)]
    calls, _ = _setup(monkeypatch, chunks=chunks, verdicts=Verdict(found=False))
    out = asyncio.run(nodes.investigate(_state(), _cfg(max_chunks_per_gap=10)))
    assert len(calls) == 10
    assert out["unresolved"][0]["chunks_checked"] == 10


def test_max_judge_calls_stops_run_and_reports(monkeypatch):
    chunks = [_chunk(i, f"текст {i}") for i in range(10)]
    calls, _ = _setup(monkeypatch, chunks=chunks, verdicts=Verdict(found=False),
                      topics=("тема1", "тема2", "тема3"))
    out = asyncio.run(
        nodes.investigate(_state(n_gaps=3), _cfg(max_judge_calls=12, judge_concurrency=1))
    )
    assert len(calls) <= 12
    # Прогон честно сообщает, что дошёл не до всех пробелов.
    assert out.get("budget_exhausted") is True
    assert len(out["resolved"]) + len(out["unresolved"]) < 3


def test_judge_unavailable_keeps_gap_open(monkeypatch):
    chunks = [_chunk(0, "текст")]
    _setup(monkeypatch, chunks=chunks, verdicts=Verdict(found=None))
    out = asyncio.run(nodes.investigate(_state(), _cfg()))
    assert out["unresolved"] and not out["resolved"]


def test_no_chunks_means_unresolved(monkeypatch):
    _setup(monkeypatch, chunks=[], verdicts=Verdict(found=False))
    out = asyncio.run(nodes.investigate(_state(), _cfg()))
    assert out["unresolved"][0]["chunks_checked"] == 0
    assert not out["resolved"]


# --- порядок обхода -------------------------------------------------------

def test_chunks_with_question_words_checked_first(monkeypatch):
    """Ни один чанк не выбрасывается — меняется только очередь."""
    chunks = [
        _chunk(0, "совершенно посторонний текст"),
        _chunk(1, "ещё один посторонний"),
        _chunk(2, "здесь есть отпуска и график"),
    ]
    calls, _ = _setup(monkeypatch, chunks=chunks, verdicts=Verdict(found=False))
    state = _state()
    state["gaps"][0]["query"] = "как оформляются отпуска"
    asyncio.run(nodes.investigate(state, _cfg(judge_concurrency=1)))
    assert calls[0] == "здесь есть отпуска и график"
    assert len(calls) == 3, "остальные чанки всё равно проверяются"


# --- устойчивость ---------------------------------------------------------

def test_error_on_one_gap_does_not_kill_run(monkeypatch):
    chunks = [_chunk(0, "текст")]
    calls, _ = _setup(monkeypatch, chunks=chunks, verdicts=Verdict(found=False),
                      topics=("t1", "t2"))

    original = nodes.load_source_chunks
    seen: list[int] = []

    async def flaky(session, *, direction_key, limit):
        seen.append(1)
        if len(seen) == 1:
            raise RuntimeError("база моргнула")
        return await original(session, direction_key=direction_key, limit=limit)

    monkeypatch.setattr(nodes, "load_source_chunks", flaky)
    out = asyncio.run(nodes.investigate(_state(n_gaps=2), _cfg()))
    assert len(out["errors"]) == 1
    assert len(out["unresolved"]) == 1


def test_skips_gap_already_resolved(monkeypatch):
    import datetime

    chunks = [_chunk(0, "текст")]
    calls, _ = _setup(
        monkeypatch, chunks=chunks, verdicts=Verdict(found=True),
        gap=FakeGap(id=uuid4(), resolved_at=datetime.datetime.now()),
    )
    out = asyncio.run(nodes.investigate(_state(), _cfg()))
    assert calls == []
    assert not out["resolved"] and not out["unresolved"]


# --- отчёт ----------------------------------------------------------------

def test_report_shows_source_and_scanned_counts():
    state = {
        "direction_key": DIRECTION,
        "resolved": [{
            "query": "как оформить отпуск", "topic": "отпуск", "stub_slug": None,
            "chunks_checked": 3,
            "evidence": {"uri": "регламент.md", "ord": 7, "preview": "цитата"},
        }],
        "unresolved": [{
            "query": "что-то ещё", "topic": "прочее", "stub_slug": None,
            "chunks_checked": 40, "capped": True,
        }],
    }
    out = asyncio.run(nodes.finalize(state))
    text = out["report"]
    assert "регламент.md" in text
    assert "близость" not in text
    assert "40" in text


def test_topology_unchanged():
    from langgraph_executor.aegra_agents.gap_resolver.graph import build_graph

    edges = {(e.source, e.target) for e in build_graph().get_graph().edges}
    assert ("__start__", "load_gaps") in edges
    assert ("load_gaps", "investigate") in edges
    assert ("investigate", "finalize") in edges
    assert ("finalize", "__end__") in edges


def test_chunk_embedding_column_gone():
    from langgraph_executor.aegra_agents.easyrag.models import SourceChunk

    assert "embedding" not in SourceChunk.__table__.columns
    # Остальные векторные колонки на месте — вики строится на них.
    from langgraph_executor.aegra_agents.easyrag.models import EntityCandidate, WikiSection

    assert "embedding" in WikiSection.__table__.columns
    assert "embedding" in EntityCandidate.__table__.columns


def test_migration_drops_column_idempotently():
    from pathlib import Path

    sql = (
        Path(__file__).resolve().parents[2]
        / "migrations" / "wiki_rag" / "0004_drop_chunk_embedding.sql"
    ).read_text(encoding="utf-8")
    assert "DROP COLUMN IF EXISTS embedding" in sql
    assert "source_chunk" in sql
