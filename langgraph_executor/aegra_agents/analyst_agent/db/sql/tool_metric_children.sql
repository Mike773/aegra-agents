-- name: tool_metric_children
-- summary: Состав показателя — влияющие показатели с весами и их последними значениями
-- params: person_key:str, metric:str, max_depth:int = 2, row_limit:int = 40
-- returns: child, child_depth, influent_percent, share_pct, date, fact, plan, plan_status, plan_dev_pct, pop_status
-- render: table
--
-- Дерево берётся у ЭТОГО человека (metric_edge.person_id): состав показателя
-- у руководителя и сотрудника может отличаться.
--
-- Даты у детей могут отставать от родительской: берём последнее значение
-- КАЖДОЙ серии, поэтому ребёнок не пропадает из состава.
WITH RECURSIVE me(person_id) AS (
    SELECT person_id FROM person WHERE person_key = :person_key
),
down(root_id, parent_id, node_id, lvl) AS (
    SELECT m.metric_id, NULL, m.metric_id, 0 FROM metric m
     WHERE ru_lower(m.name) = ru_lower(:metric)
    UNION
    SELECT d.root_id, e.parent_id, e.child_id, d.lvl + 1
      FROM down d
      JOIN metric_edge e ON e.parent_id = d.node_id
       AND e.person_id = (SELECT person_id FROM me)
     WHERE d.lvl < :max_depth
)
SELECT c.name AS child, d.lvl AS child_depth, e.influent_percent, t.share_pct,
       f.date, f.fact, f.plan, f.plan_status, f.plan_dev_pct, f.pop_status
FROM down d
JOIN metric c ON c.metric_id = d.node_id
LEFT JOIN metric_edge e ON e.person_id = (SELECT person_id FROM me)
       AND e.parent_id = d.parent_id AND e.child_id = c.metric_id
LEFT JOIN metric pm ON pm.metric_id = d.parent_id
LEFT JOIN v_tree t ON t.person_key = :person_key
       AND t.parent = pm.name AND t.child = c.name
LEFT JOIN v_fact_latest f ON f.metric = c.name AND f.person_key = :person_key
       AND f.element IS NULL
WHERE d.lvl > 0
ORDER BY d.lvl, COALESCE(t.share_pct, 0) DESC, c.name
LIMIT :row_limit;
