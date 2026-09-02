-- name: enrich_catalog
-- summary: Каталог показателей 1-2 уровня со счётчиками метрик нижних слоёв
-- params: person_key:str, max_level:int = 2, row_limit:int = 1000
-- returns: metric_id, name, depth, parent_name, unit, direction, kind, has_plan, n_elements, has_aggregate, influent_percent
-- render: table
--
-- Глубже второго уровня имена в промпт не идут: для каждой метрики 2-го уровня
-- отдельно считаются счётчики потомков по слоям (см. enrich_deeper_counts).
--
-- Разрезы: в промпт идёт только их КОЛИЧЕСТВО и признак наличия собственного
-- итога (строки без разреза) — имён разрезов здесь нет, их бывают сотни.
-- Оба поля берутся на фокус-человека: у разных людей один и тот же показатель
-- может быть представлен по-разному.
SELECT m.metric_id,
       m.name,
       m.depth,
       m.parent_name,
       m.unit,
       m.direction,
       m.kind,
       COALESCE(mp.has_plan, 0) AS has_plan,
       COALESCE(mp.n_elements, 0) AS n_elements,
       COALESCE(mp.has_aggregate, 1) AS has_aggregate,
       m.influent_percent
FROM metric m
LEFT JOIN v_metric_person mp
       ON mp.metric_id = m.metric_id AND mp.person_key = :person_key
WHERE m.depth <= :max_level AND m.is_star = 0
ORDER BY m.depth, COALESCE(m.parent_name, m.name), m.influent_percent DESC, m.name
LIMIT :row_limit;
