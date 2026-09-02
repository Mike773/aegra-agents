-- name: enrich_deeper_counts
-- summary: Сколько потомков каждой метрики лежит на каждом уровне ниже неё
-- params: max_level:int = 2, row_limit:int = 5000
-- returns: ancestor, descendant_depth, n
-- render: table
--
-- Рекурсивный обход дерева вниз от каждой метрики уровня <= :max_level.
-- Нужен для строк каталога вида «(10 метрик 3-го, 20 4-го)».
WITH RECURSIVE down(ancestor_id, node_id) AS (
    SELECT m.metric_id, m.metric_id FROM metric m WHERE m.depth <= :max_level
    UNION
    SELECT d.ancestor_id, e.child_id
    FROM down d JOIN metric_edge e ON e.parent_id = d.node_id
)
SELECT a.name AS ancestor, n.depth AS descendant_depth, COUNT(DISTINCT n.metric_id) AS n
FROM down d
JOIN metric a ON a.metric_id = d.ancestor_id
JOIN metric n ON n.metric_id = d.node_id
WHERE n.depth > a.depth
GROUP BY a.name, n.depth
ORDER BY a.name, n.depth
LIMIT :row_limit;
