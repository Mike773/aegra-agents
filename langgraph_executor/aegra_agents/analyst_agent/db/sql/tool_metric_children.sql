-- name: tool_metric_children
-- summary: Состав показателя деревом — влияющие показатели с весами и их последними значениями
-- params: person_key:str, metric:str, max_depth:int = 2, row_limit:int = 60
-- returns: child, child_depth, parent, is_repeat, influent_percent, share_pct, date, fact, plan, ex, rr, plan_status, plan_dev_pct, pop_status, n_total
-- render: table
--
-- Дерево берётся у ЭТОГО человека (metric_edge.person_id): состав показателя
-- у руководителя и сотрудника может отличаться.
--
-- Порядок — обход в глубину: каждый ребёнок стоит сразу под своим родителем,
-- сиблинги — по убыванию доли влияния, затем по имени (sort_path).
--
-- Один показатель может влиять на нескольких родителей. Полностью он
-- раскрывается один раз — под «домашним» родителем: на самой мелкой глубине,
-- где показатель встречается (чтобы глубина не обрезала его потомков), среди
-- родителей этой глубины — с наибольшей долей, затем по имени. Остальные
-- вхождения идут строкой is_repeat = 1 (свой вес в этом месте, без значений)
-- и без потомков — иначе общий показатель и всё его поддерево повторялись по
-- разу на каждого родителя. Дерево строится только по домашним рёбрам, поэтому
-- запрос линеен по числу рёбер, а не по числу путей, и не зацикливается.
--
-- n_total — сколько строк в составе до LIMIT: по нему видно, что выдача
-- обрезана.
--
-- Даты у детей могут отставать от родительской: берём последнее значение
-- КАЖДОЙ серии, поэтому ребёнок не пропадает из состава.
WITH RECURSIVE me(person_id) AS (
    SELECT person_id FROM person WHERE person_key = :person_key
),
root AS (
    SELECT metric_id FROM metric WHERE ru_lower(name) = ru_lower(:metric)
),
edges AS (                          -- рёбра человека с долей среди сиблингов
    SELECT e.parent_id, e.child_id, e.influent_percent,
           CASE WHEN SUM(e.influent_percent) OVER (PARTITION BY e.parent_id) > 0
                THEN ROUND(e.influent_percent * 100.0
                           / SUM(e.influent_percent) OVER (PARTITION BY e.parent_id), 2)
           END AS share_pct,
           printf('%09.3f', 1000 - COALESCE(
               e.influent_percent * 100.0
               / NULLIF(SUM(e.influent_percent) OVER (PARTITION BY e.parent_id), 0), 0))
               || c.name || char(31) AS sort_key
    FROM metric_edge e
    JOIN metric c ON c.metric_id = e.child_id
    WHERE e.person_id = (SELECT person_id FROM me)
      AND e.child_id NOT IN (SELECT metric_id FROM root)
),
reach(node_id, lvl) AS (            -- UNION: пара (показатель, глубина) один раз
    SELECT metric_id, 0 FROM root
    UNION
    SELECT e.child_id, r.lvl + 1
      FROM reach r JOIN edges e ON e.parent_id = r.node_id
     WHERE r.lvl < :max_depth
),
mind AS (
    SELECT node_id, MIN(lvl) AS lvl FROM reach GROUP BY node_id
),
home AS (                           -- домашний родитель: на глубине на 1 мельче
    SELECT child_id, parent_id FROM (
        SELECT e.child_id, e.parent_id,
               ROW_NUMBER() OVER (PARTITION BY e.child_id ORDER BY e.sort_key) AS rn
          FROM edges e
          JOIN mind mc ON mc.node_id = e.child_id
          JOIN mind mp ON mp.node_id = e.parent_id AND mp.lvl = mc.lvl - 1
    ) WHERE rn = 1
),
tree(node_id, parent_id, lvl, sort_path, influent_percent, share_pct) AS (
    SELECT metric_id, NULL, 0, '', NULL, NULL FROM root
    UNION ALL
    SELECT e.child_id, t.node_id, t.lvl + 1, t.sort_path || e.sort_key,
           e.influent_percent, e.share_pct
      FROM tree t
      JOIN edges e ON e.parent_id = t.node_id
      JOIN home h ON h.child_id = e.child_id AND h.parent_id = e.parent_id
     WHERE t.lvl < :max_depth
),
shown AS (
    SELECT node_id, parent_id, lvl, sort_path, influent_percent, share_pct,
           0 AS is_repeat
      FROM tree WHERE lvl > 0
    UNION ALL                       -- остальные вхождения — строкой-повтором
    SELECT e.child_id, e.parent_id, t.lvl + 1, t.sort_path || e.sort_key,
           e.influent_percent, e.share_pct, 1
      FROM tree t
      JOIN edges e ON e.parent_id = t.node_id
      JOIN home h ON h.child_id = e.child_id AND h.parent_id <> e.parent_id
     WHERE t.lvl < :max_depth
)
SELECT c.name AS child, s.lvl AS child_depth, pm.name AS parent, s.is_repeat,
       s.influent_percent, s.share_pct,
       f.date, f.fact, f.plan, f.ex, f.rr, f.plan_status, f.plan_dev_pct, f.pop_status,
       COUNT(*) OVER () AS n_total
FROM shown s
JOIN metric c ON c.metric_id = s.node_id
LEFT JOIN metric pm ON pm.metric_id = s.parent_id
LEFT JOIN v_fact_latest f ON s.is_repeat = 0 AND f.metric = c.name
       AND f.person_key = :person_key AND f.element IS NULL
ORDER BY s.sort_path
LIMIT :row_limit;
