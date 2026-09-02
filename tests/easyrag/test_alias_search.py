"""Поиск по алиасам страницы: аббревиатуры должны находиться.

Модель эмбеддингов плохо работает с аббревиатурами, а в тексте секции короткий
алиас растворяется — вектор его почти не замечает. Поэтому у каждого алиаса свой
вектор, а перед векторами работает точное совпадение слова из запроса с алиасом:
оно ловит аббревиатуру гарантированно и без вызова модели.

Здесь проверяется чистая логика (нормализация, разбор запроса, слияние выдачи) —
без Postgres.
"""
from __future__ import annotations

import os
import sys
from uuid import uuid4

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from langgraph_executor.aegra_agents.easyrag.aliases import (  # noqa: E402
    alias_norm,
    query_alias_candidates,
)
from langgraph_executor.aegra_agents.easyrag.retrieval import (  # noqa: E402
    RetrievedSection,
    merge_hits,
)


# --- нормализация алиаса --------------------------------------------------

def test_alias_norm_is_case_and_space_insensitive():
    assert alias_norm("  AHT ") == alias_norm("aht") == "aht"
    assert alias_norm("Среднее   Время") == "среднее время"


def test_alias_norm_empty():
    assert alias_norm("") == ""
    assert alias_norm(None) == ""
    assert alias_norm("   ") == ""


# --- что из запроса сопоставляем с алиасами -------------------------------

def test_query_candidates_include_whole_query_and_words():
    cands = query_alias_candidates("Что такое AHT")
    assert "что такое aht" in cands      # весь запрос целиком
    assert "aht" in cands                # и отдельные слова


def test_query_candidates_keep_short_abbreviations():
    """«ФОТ», «NPS», «SL» — короткие, но это и есть аббревиатуры."""
    cands = query_alias_candidates("какой сейчас SL и ФОТ")
    assert "sl" in cands
    assert "фот" in cands


def test_query_candidates_drop_noise_words():
    cands = query_alias_candidates("а и в на что")
    # Односимвольные предлоги алиасами быть не могут — их отбрасываем.
    assert "а" not in cands and "и" not in cands


def test_query_candidates_include_bigrams():
    """Алиас часто из двух слов — «время обработки»."""
    cands = query_alias_candidates("какое среднее время обработки")
    assert "среднее время" in cands or "время обработки" in cands


def test_query_candidates_bounded():
    long_query = " ".join(f"слово{i}" for i in range(200))
    assert len(query_alias_candidates(long_query)) <= 64


# --- слияние выдачи -------------------------------------------------------

def _section(slug: str, anchor: str, sim: float, source: str) -> RetrievedSection:
    return RetrievedSection(
        section_id=uuid4(), page_id=uuid4(), slug=slug, anchor=anchor,
        page_title=slug, section_title=anchor, body_md="текст",
        similarity=sim, source=source,
    )


def test_alias_hits_come_first():
    alias = [_section("aht", "обзор", 1.0, "alias")]
    vector = [_section("прочее", "обзор", 0.9, "vector")]
    merged = merge_hits(alias_hits=alias, vector_hits=vector, limit=10)
    assert merged[0].source == "alias"


def test_merge_drops_duplicate_sections():
    shared = _section("aht", "обзор", 1.0, "alias")
    same_from_vector = RetrievedSection(
        section_id=shared.section_id, page_id=shared.page_id, slug=shared.slug,
        anchor=shared.anchor, page_title=shared.page_title,
        section_title=shared.section_title, body_md=shared.body_md,
        similarity=0.7, source="vector",
    )
    merged = merge_hits(alias_hits=[shared], vector_hits=[same_from_vector], limit=10)
    assert len(merged) == 1
    assert merged[0].source == "alias"


def test_merge_respects_limit():
    alias = [_section(f"a{i}", "обзор", 1.0, "alias") for i in range(3)]
    vector = [_section(f"v{i}", "обзор", 0.9, "vector") for i in range(10)]
    merged = merge_hits(alias_hits=alias, vector_hits=vector, limit=5)
    assert len(merged) == 5
    assert sum(1 for m in merged if m.source == "alias") == 3


def test_alias_hits_alone_are_enough():
    merged = merge_hits(alias_hits=[_section("aht", "обзор", 1.0, "alias")],
                        vector_hits=[], limit=8)
    assert len(merged) == 1


def test_merge_keeps_vector_order():
    vector = [
        _section("первый", "обзор", 0.9, "vector"),
        _section("второй", "обзор", 0.5, "vector"),
    ]
    merged = merge_hits(alias_hits=[], vector_hits=vector, limit=8)
    assert [m.slug for m in merged] == ["первый", "второй"]


# --- схема ----------------------------------------------------------------

def test_wiki_alias_model_exists():
    from langgraph_executor.aegra_agents.easyrag.models import WikiAlias

    cols = {c.name for c in WikiAlias.__table__.columns}
    assert {"page_id", "direction_key", "alias", "alias_norm", "embedding"} <= cols


def test_migration_creates_alias_table_idempotently():
    from pathlib import Path

    sql = (
        Path(__file__).resolve().parents[2]
        / "migrations" / "wiki_rag" / "0005_wiki_alias.sql"
    ).read_text(encoding="utf-8")
    creates = [
        ln.strip() for ln in sql.splitlines()
        if ln.strip().upper().startswith(("CREATE TABLE", "CREATE INDEX", "CREATE UNIQUE"))
    ]
    assert creates
    for line in creates:
        assert "IF NOT EXISTS" in line.upper(), line
    assert "wiki_alias" in sql


# --- синхронизация строк wiki_alias со страницей --------------------------

class FakeEmbedder:
    """Считает, сколько строк реально ушло на эмбеддинг."""

    def __init__(self):
        self.embedded: list[str] = []

    async def embed_many(self, texts):
        self.embedded.extend(texts)
        return [[0.1] * 4 for _ in texts]


class FakeSession:
    def __init__(self, existing):
        self.existing = list(existing)
        self.added = []
        self.deleted = []
        self.flushed = 0

    async def execute(self, _stmt):
        rows = self.existing

        class Result:
            def scalars(self_inner):
                return self_inner

            def all(self_inner):
                return rows

        return Result()

    def add(self, obj):
        self.added.append(obj)

    async def delete(self, obj):
        self.deleted.append(obj)

    async def flush(self):
        self.flushed += 1


class FakePage:
    def __init__(self, aliases):
        self.id = uuid4()
        self.direction_key = "test-dir"
        self.aliases = aliases


def _sync(page, existing):
    import asyncio

    from langgraph_executor.aegra_agents.wiki_ingest.merge_utils import (
        sync_alias_embeddings,
    )

    session = FakeSession(existing)
    embedder = FakeEmbedder()
    added = asyncio.run(sync_alias_embeddings(session, page, embedder))
    return session, embedder, added


class ExistingAlias:
    def __init__(self, norm):
        self.alias_norm = norm


def test_sync_embeds_each_alias_separately():
    session, embedder, added = _sync(FakePage(["AHT", "Среднее время"]), [])
    assert added == 2
    # Каждый алиас эмбеддится сам по себе, а не склеенным текстом.
    assert embedder.embedded == ["AHT", "Среднее время"]
    assert {a.alias_norm for a in session.added} == {"aht", "среднее время"}


def test_sync_skips_already_embedded():
    session, embedder, added = _sync(
        FakePage(["AHT", "SL"]), [ExistingAlias("aht")]
    )
    assert added == 1
    assert embedder.embedded == ["SL"]     # существующий вектор переиспользован


def test_sync_removes_dropped_aliases():
    session, _, _ = _sync(FakePage(["AHT"]), [ExistingAlias("aht"), ExistingAlias("устаревший")])
    assert [d.alias_norm for d in session.deleted] == ["устаревший"]


def test_sync_ignores_blank_and_duplicate_aliases():
    session, embedder, added = _sync(FakePage(["AHT", " aht ", "", "   "]), [])
    assert added == 1
    assert embedder.embedded == ["AHT"]


def test_sync_on_page_without_aliases():
    session, embedder, added = _sync(FakePage([]), [])
    assert added == 0
    assert embedder.embedded == []


# --- порог вектора алиаса -------------------------------------------------

def test_alias_vector_threshold_rejects_model_noise():
    """У этой модели эмбеддингов косинус сжат в узкую полосу.

    Замер на живой базе: настоящее совпадение алиаса даёт 0.96-1.00, а
    посторонний запрос («расписание поездов») к тому же алиасу — 0.78. Порог
    обязан лежать между: иначе один алиас притягивает любой вопрос.
    """
    from langgraph_executor.aegra_agents.easyrag.retrieval import DEFAULT_ALIAS_THRESH

    assert DEFAULT_ALIAS_THRESH > 0.79, "шум модели (0.78-0.79) не должен проходить"
    assert DEFAULT_ALIAS_THRESH <= 0.95, "настоящее совпадение (0.96+) должно проходить"


def test_alias_vector_accepted_only_above_best_section():
    """Вектор алиаса не перебивает секцию, которая по смыслу ближе."""
    from langgraph_executor.aegra_agents.easyrag.retrieval import alias_vector_accepted

    # Настоящее совпадение: и выше порога, и лучше любой секции.
    assert alias_vector_accepted(0.96, best_section=0.77, threshold=0.85)
    # Шум: выше секции, но ниже порога.
    assert not alias_vector_accepted(0.783, best_section=0.762, threshold=0.85)
    # Выше порога, но секция ближе по смыслу — пусть отвечает секция.
    assert not alias_vector_accepted(0.86, best_section=0.90, threshold=0.85)
    # Секций нет вовсе — судим только по порогу.
    assert alias_vector_accepted(0.96, best_section=None, threshold=0.85)
