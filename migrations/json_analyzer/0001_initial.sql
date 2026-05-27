-- json_analyzer — кэш эмбеддингов метрик для подагента json_analyzer.
-- Подключается к той же базе, что и aegra (POSTGRES_DSN из .env),
-- но изолировано в отдельной схеме json_analyzer.
-- Размер вектора — 1024 (GigaChat embeddings).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS json_analyzer;

-- Кэш эмбеддингов названий и описаний метрик и значений element.
-- Изоляция между направлениями обеспечивается колонкой direction_key и
-- составным UNIQUE (direction_key, content_hash): один и тот же текст в двух
-- направлениях хранится двумя строками; поиск возвращает только записи своего
-- direction_key.
CREATE TABLE IF NOT EXISTS json_analyzer.metric_embeddings (
    id            BIGSERIAL PRIMARY KEY,
    direction_key TEXT NOT NULL,
    kind          TEXT NOT NULL,
    content       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    canonical     TEXT NOT NULL,
    embedding     vector(1024) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_metric_embeddings_dir_hash UNIQUE (direction_key, content_hash)
);
CREATE INDEX IF NOT EXISTS ix_metric_embeddings_direction
    ON json_analyzer.metric_embeddings (direction_key);
CREATE INDEX IF NOT EXISTS ix_metric_embeddings_dir_kind
    ON json_analyzer.metric_embeddings (direction_key, kind);
CREATE INDEX IF NOT EXISTS ix_metric_embeddings_embedding
    ON json_analyzer.metric_embeddings
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
