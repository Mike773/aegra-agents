-- name: tool_metric_children
-- summary: Состав показателя — влияющие показатели с весами и их последними значениями
-- params: person_key:str, metric:str, max_depth:int = 2, row_limit:int = 40
-- returns: child, child_depth, influent_percent, share_pct, date, fact, plan, plan_status, plan_dev_pct, pop_status
-- render: table
--
-- Даты у детей могут отставать от родительской: берём последнее значение
-- КАЖДОЙ серии, поэтому ребёнок не пропадает из состава.
WITH RECURSIVE down(root_id, node_id, lvl) AS (
    SELECT m.metric_id, m.metric_id, 0 FROM metric m
     WHERE ru_lower(m.name) = ru_lower(:metric)
    UNION
    SELECT d.root_id, e.child_id, d.lvl + 1
      FROM down d JOIN metric_edge e ON e.parent_id = d.node_id
     WHERE d.lvl < :max_depth
)
SELECT c.name AS child, d.lvl AS child_depth, e.influent_percent, t.share_pct,
       f.date, f.fact, f.plan, f.plan_status, f.plan_dev_pct, f.pop_status
FROM down d
JOIN metric c ON c.metric_id = d.node_id
LEFT JOIN metric_edge e ON e.child_id = c.metric_id
LEFT JOIN v_tree t ON t.child = c.name
LEFT JOIN v_fact_latest f ON f.metric = c.name AND f.person_key = :person_key
       AND f.element IS NULL
WHERE d.lvl > 0
ORDER BY d.lvl, COALESCE(t.share_pct, 0) DESC, c.name
LIMIT :row_limit;
