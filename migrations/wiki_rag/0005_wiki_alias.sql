-- Алиасы страниц как отдельная поисковая сущность.
--
-- Зачем: аббревиатуры («AHT», «SL», «ФОТ») лежат в wiki_page.aliases, но в
-- поиске не участвовали вовсе — вектор секции собирается из заголовков и тела,
-- где короткий алиас растворяется, а сам массив aliases в поиск не входил.
-- Теперь у каждого алиаса своя строка: нормализованная форма для точного
-- совпадения со словом из запроса и собственный вектор для близких форм.
--
-- Строки синхронизируются с wiki_page.aliases при каждом upsert страницы и
-- каскадно удаляются вместе со страницей.

CREATE TABLE IF NOT EXISTS wiki_rag.wiki_alias (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id       UUID NOT NULL REFERENCES wiki_rag.wiki_page(id) ON DELETE CASCADE,
    direction_key TEXT NOT NULL,
    alias         TEXT NOT NULL,          -- как записано на странице
    alias_norm    TEXT NOT NULL,          -- регистр и пробелы схлопнуты, для точного матча
    embedding     vector(1024),
    CONSTRAINT uq_wiki_alias UNIQUE (page_id, alias_norm)
);

-- Точное совпадение идёт по (направление, нормализованный алиас) — основной путь.
CREATE INDEX IF NOT EXISTS ix_wiki_alias_lookup
    ON wiki_rag.wiki_alias (direction_key, alias_norm);
CREATE INDEX IF NOT EXISTS ix_wiki_alias_page
    ON wiki_rag.wiki_alias (page_id);
-- Векторный индекс — как у wiki_section: списков меньше, алиасов на порядок меньше секций.
CREATE INDEX IF NOT EXISTS ix_wiki_alias_embedding
    ON wiki_rag.wiki_alias USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
