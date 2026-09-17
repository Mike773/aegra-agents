-- name: enrich_profile
-- summary: Состав данных: строка на человека — роль, должность, показатели первого уровня с числом потомков, даты периодов, группы сравнения
-- params:
-- returns: person_key, fio, post, is_me, roots, dates, n_periods, groups
-- render: table
WITH RECURSIVE
roots AS (
    -- Корни дерева ЭТОГО человека: показатель с фактами, который ни у кого
    -- не числится ребёнком в его дереве.
    SELECT DISTINCT f.person_id, f.metric_id
    FROM fact f
    WHERE NOT EXISTS (
        SELECT 1 FROM metric_edge e
        WHERE e.person_id = f.person_id AND e.child_id = f.metric_id
    )
),
descendants(person_id, root_id, metric_id, lvl) AS (
    SELECT e.person_id, e.parent_id, e.child_id, 2
    FROM metric_edge e
    JOIN roots r ON r.person_id = e.person_id AND r.metric_id = e.parent_id
    UNION
    SELECT e.person_id, d.root_id, e.child_id, d.lvl + 1
    FROM metric_edge e
    JOIN descendants d ON d.person_id = e.person_id AND d.metric_id = e.parent_id
),
root_lines AS (
    SELECT r.person_id, m.name,
           (SELECT COUNT(DISTINCT d.metric_id) FROM descendants d
             WHERE d.person_id = r.person_id AND d.root_id = r.metric_id AND d.lvl = 2) AS n_l2,
           (SELECT COUNT(DISTINCT d.metric_id) FROM descendants d
             WHERE d.person_id = r.person_id AND d.root_id = r.metric_id) AS n_total
    FROM roots r
    JOIN metric m ON m.metric_id = r.metric_id
)
SELECT p.person_key, p.fio, p.post, p.is_me,
       (SELECT group_concat(name || '|' || n_l2 || '|' || n_total, ';;')
          FROM (SELECT name, n_l2, n_total FROM root_lines rl
                 WHERE rl.person_id = p.person_id ORDER BY name)) AS roots,
       (SELECT group_concat(date, ',')
          FROM (SELECT DISTINCT pe.date FROM fact f
                  JOIN period pe ON pe.period_id = f.period_id
                 WHERE f.person_id = p.person_id ORDER BY pe.date DESC)) AS dates,
       (SELECT COUNT(DISTINCT f.period_id) FROM fact f
         WHERE f.person_id = p.person_id) AS n_periods,
       (SELECT group_concat(level_name, ', ')
          FROM (SELECT DISTINCT pa.level_name, pa.level_order
                  FROM person_peer pp
                  JOIN peer_aggregate pa ON pa.aggregate_id = pp.aggregate_id
                 WHERE pp.person_id = p.person_id AND pa.level_name IS NOT NULL
                 ORDER BY pa.level_order)) AS groups
FROM person p
ORDER BY p.is_me DESC, p.person_id;
