-- Кэш трактовок показателей из базы знаний (для analyst_agent).
--
-- Зачем: названия показателей на проде содержат аббревиатуры и внутренние
-- термины, а правила их чтения (накопительный ли, как считается план, в каком
-- процессе участвует) лежат в wiki. Искать это на каждый запрос дорого, поэтому
-- найденное складывается сюда и переиспользуется всеми диалогами направления.
--
-- Идемпотентно (IF NOT EXISTS), как 0001/0002. Индексов ровно столько, сколько
-- нужно выборкам: по направлению, по имени (фолбэк при дрейфе описания) и по
-- статусу (что дозаполнять фоновому графу).

-- Каталог показателей, которые видел основной граф. Источник работы для
-- фонового обогатителя: он не зависит от того, какой датасет грузили последним.
CREATE TABLE IF NOT EXISTS wiki_rag.metric_catalog (
    direction_key      TEXT NOT NULL,
    metric_key         VARCHAR(64) NOT NULL,   -- sha256(имя + описание)
    name_key           VARCHAR(64) NOT NULL,   -- sha256(имя)
    metric_name        TEXT NOT NULL,
    metric_description TEXT NOT NULL DEFAULT '',
    metric_ext_id      TEXT,
    direction          TEXT,
    unit               TEXT,
    parent_name        TEXT,
    depth              INT,
    first_seen_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    seen_count         INT NOT NULL DEFAULT 1,
    PRIMARY KEY (direction_key, metric_key)
);

CREATE INDEX IF NOT EXISTS ix_metric_catalog_seen
    ON wiki_rag.metric_catalog (direction_key, last_seen_at);

CREATE TABLE IF NOT EXISTS wiki_rag.metric_knowledge (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    direction_key      TEXT NOT NULL,
    metric_key         VARCHAR(64) NOT NULL,
    name_key           VARCHAR(64) NOT NULL,
    metric_name        TEXT NOT NULL,
    metric_description TEXT NOT NULL DEFAULT '',
    aliases            TEXT[] NOT NULL DEFAULT '{}',   -- аббревиатуры и синонимы
    summary            TEXT NOT NULL DEFAULT '',       -- одна-две фразы для промпта
    -- rules: {kind, cumulative, plan_semantics, fact_semantics, processes[], notes}
    rules              JSONB NOT NULL DEFAULT '{}'::jsonb,
    status             VARCHAR(16) NOT NULL DEFAULT 'found',  -- found | unknown
    confidence         REAL NOT NULL DEFAULT 0,
    source_section_ids UUID[] NOT NULL DEFAULT '{}',
    model              TEXT,
    version            INT NOT NULL DEFAULT 1,         -- версия схемы/промпта
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_metric_knowledge UNIQUE (direction_key, metric_key)
);

CREATE INDEX IF NOT EXISTS ix_metric_knowledge_direction
    ON wiki_rag.metric_knowledge (direction_key);
CREATE INDEX IF NOT EXISTS ix_metric_knowledge_name_key
    ON wiki_rag.metric_knowledge (direction_key, name_key);
CREATE INDEX IF NOT EXISTS ix_metric_knowledge_status
    ON wiki_rag.metric_knowledge (direction_key, status, updated_at);
