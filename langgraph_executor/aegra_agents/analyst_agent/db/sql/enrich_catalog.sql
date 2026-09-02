-- name: enrich_catalog
-- summary: Каталог показателей 1-2 уровня со счётчиками метрик нижних слоёв
-- params: max_level:int = 2, row_limit:int = 1000
-- returns: metric_id, name, depth, parent_name, unit, direction, kind, has_plan, n_elements, influent_percent
-- render: table
--
-- Глубже второго уровня имена в промпт не идут: для каждой метрики 2-го уровня
-- отдельно считаются счётчики потомков по слоям (см. enrich_deeper_counts).
SELECT m.metric_id,
       m.name,
       m.depth,
       m.parent_name,
       m.unit,
       m.direction,
       m.kind,
       (SELECT COUNT(*) FROM fact f
         WHERE f.metric_id = m.metric_id AND f.plan IS NOT NULL AND f.plan <> 0) > 0 AS has_plan,
       (SELECT COUNT(DISTINCT f.element) FROM fact f
         WHERE f.metric_id = m.metric_id AND f.element IS NOT NULL) AS n_elements,
       m.influent_percent
FROM metric m
WHERE m.depth <= :max_level AND m.is_star = 0
ORDER BY m.depth, COALESCE(m.parent_name, m.name), m.influent_percent DESC, m.name
LIMIT :row_limit;
