-- wiki_rag — ingest-таблицы подагента wiki_ingest (порт ingest-стороны easyRag).
-- Дополняет 0001_initial.sql: источники документов и кандидаты сущностей,
-- из которых резолвер собирает wiki_page/wiki_section.
-- Размер вектора — 1024 (GigaChat embeddings), как в 0001.
--
-- Отличия от easyRag:
--   * source_doc.content — сырой текст документа лежит в БД (агент его читает);
--   * source_doc.processed_at — признак идемпотентности (NULL = не обработан);
--   * direction_key на source_doc/source_chunk/entity_candidate.
--
-- UUID-дефолты — gen_random_uuid() (встроен в ядро PG13+). Индексов пока нет:
-- добавим отдельной миграцией после наполнения данными.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS wiki_rag.source_doc (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    direction_key TEXT NOT NULL,
    uri           TEXT NOT NULL,
    mime          VARCHAR(64),
    content       TEXT NOT NULL,
    sha256        VARCHAR(64),
    domain_brief  TEXT,
    processed_at  TIMESTAMPTZ,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS wiki_rag.source_chunk (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id        UUID NOT NULL REFERENCES wiki_rag.source_doc(id) ON DELETE CASCADE,
    direction_key TEXT NOT NULL,
    ord           INT  NOT NULL,
    text          TEXT NOT NULL,
    char_start    INT  NOT NULL,
    char_end      INT  NOT NULL,
    embedding     vector(1024),
    CONSTRAINT uq_source_chunk_doc_ord UNIQUE (doc_id, ord)
);

CREATE TABLE IF NOT EXISTS wiki_rag.entity_candidate (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id           UUID NOT NULL REFERENCES wiki_rag.source_doc(id)   ON DELETE CASCADE,
    chunk_id         UUID NOT NULL REFERENCES wiki_rag.source_chunk(id) ON DELETE CASCADE,
    direction_key    TEXT NOT NULL,
    name             TEXT NOT NULL,
    descriptor       TEXT NOT NULL DEFAULT '',
    statements       TEXT[] NOT NULL DEFAULT '{}',
    resolved_page_id UUID REFERENCES wiki_rag.wiki_page(id) ON DELETE SET NULL,
    resolved_at      TIMESTAMPTZ,
    embedding        vector(1024)
);

CREATE TABLE IF NOT EXISTS wiki_rag.section_provenance (
    section_id      UUID NOT NULL REFERENCES wiki_rag.wiki_section(id) ON DELETE CASCADE,
    source_chunk_id UUID NOT NULL REFERENCES wiki_rag.source_chunk(id) ON DELETE CASCADE,
    contributed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (section_id, source_chunk_id)
);
