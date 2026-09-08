-- name: enrich_deeper_counts
-- summary: Сколько потомков каждой метрики лежит на каждом уровне ниже неё
-- params: person_key:str, max_level:int = 2, row_limit:int = 5000
-- returns: ancestor, descendant_depth, n
-- render: table
--
-- Рекурсивный обход дерева ЭТОГО человека вниз от каждой его метрики уровня
-- <= :max_level. Нужен для строк каталога вида «(10 метрик 3-го, 20 4-го)».
WITH RECURSIVE me(person_id) AS (
    SELECT person_id FROM person WHERE person_key = :person_key
),
down(ancestor_id, node_id) AS (
    SELECT t.metric_id, t.metric_id FROM metric_tree t
     WHERE t.person_id = (SELECT person_id FROM me) AND t.depth <= :max_level
    UNION
    SELECT d.ancestor_id, e.child_id
    FROM down d JOIN metric_edge e ON e.parent_id = d.node_id
     AND e.person_id = (SELECT person_id FROM me)
)
SELECT a.name AS ancestor, tn.depth AS descendant_depth, COUNT(DISTINCT tn.metric_id) AS n
FROM down d
JOIN metric a ON a.metric_id = d.ancestor_id
JOIN metric_tree ta ON ta.person_id = (SELECT person_id FROM me) AND ta.metric_id = a.metric_id
JOIN metric_tree tn ON tn.person_id = (SELECT person_id FROM me) AND tn.metric_id = d.node_id
WHERE tn.depth > ta.depth
GROUP BY a.name, tn.depth
ORDER BY a.name, tn.depth
LIMIT :row_limit;
