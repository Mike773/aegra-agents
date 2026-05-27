-- wiki_rag — RAG-хранилище подагента easyrag.
-- Подключается к той же базе, что и aegra (POSTGRES_DSN из .env),
-- но изолировано в отдельной схеме wiki_rag.
-- Размер вектора — 1024 (GigaChat embeddings).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE SCHEMA IF NOT EXISTS wiki_rag;

CREATE TABLE IF NOT EXISTS wiki_rag.wiki_page (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slug          TEXT NOT NULL,
    title         TEXT NOT NULL,
    type          TEXT,
    aliases       TEXT[] NOT NULL DEFAULT '{}',
    body_md       TEXT NOT NULL DEFAULT '',
    direction_key TEXT NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    version       INT NOT NULL DEFAULT 1,
    CONSTRAINT uq_wiki_page_direction_slug UNIQUE (direction_key, slug)
);
CREATE INDEX IF NOT EXISTS ix_wiki_page_direction ON wiki_rag.wiki_page (direction_key);
CREATE INDEX IF NOT EXISTS ix_wiki_page_type      ON wiki_rag.wiki_page (type);

CREATE TABLE IF NOT EXISTS wiki_rag.wiki_section (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    page_id       UUID NOT NULL REFERENCES wiki_rag.wiki_page(id) ON DELETE CASCADE,
    ord           INT  NOT NULL,
    anchor        TEXT NOT NULL,
    title         TEXT NOT NULL,
    body_md       TEXT NOT NULL,
    embedding     vector(1024),
    direction_key TEXT NOT NULL,
    CONSTRAINT uq_wiki_section_page_ord    UNIQUE (page_id, ord),
    CONSTRAINT uq_wiki_section_page_anchor UNIQUE (page_id, anchor)
);
CREATE INDEX IF NOT EXISTS ix_wiki_section_direction ON wiki_rag.wiki_section (direction_key);
CREATE INDEX IF NOT EXISTS ix_wiki_section_embedding ON wiki_rag.wiki_section
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS wiki_rag.wiki_link (
    from_page_id    UUID NOT NULL REFERENCES wiki_rag.wiki_page(id)    ON DELETE CASCADE,
    from_section_id UUID NOT NULL REFERENCES wiki_rag.wiki_section(id) ON DELETE CASCADE,
    to_slug         TEXT NOT NULL,
    to_page_id      UUID REFERENCES wiki_rag.wiki_page(id) ON DELETE SET NULL,
    PRIMARY KEY (from_page_id, from_section_id, to_slug)
);
CREATE INDEX IF NOT EXISTS ix_wiki_link_to_slug ON wiki_rag.wiki_link (to_slug);
CREATE INDEX IF NOT EXISTS ix_wiki_link_to_page ON wiki_rag.wiki_link (to_page_id);

CREATE TABLE IF NOT EXISTS wiki_rag.query_gap (
    id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query                TEXT NOT NULL,
    embedding            vector(1024),
    direction_key        TEXT NOT NULL,
    asked_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at          TIMESTAMPTZ,
    resolved_section_ids UUID[] NOT NULL DEFAULT '{}',
    unresolved_abbr      TEXT[] NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_query_gap_direction   ON wiki_rag.query_gap (direction_key);
CREATE INDEX IF NOT EXISTS ix_query_gap_resolved_at ON wiki_rag.query_gap (resolved_at);
