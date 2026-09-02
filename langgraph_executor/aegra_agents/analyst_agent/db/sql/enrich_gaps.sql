-- name: enrich_gaps
-- summary: Пробелы данных: показатели без плана, с единственным периодом, без разрезов
-- params: person_key:str
-- returns: n_no_plan, names_no_plan, n_single_period, names_single_period, n_no_knowledge
-- render: kv
WITH mine AS (
    SELECT DISTINCT m.metric_id, m.name, m.kn_status
    FROM fact f JOIN metric m ON m.metric_id = f.metric_id
    JOIN person p ON p.person_id = f.person_id
    WHERE p.person_key = :person_key
),
no_plan AS (
    SELECT name FROM mine WHERE NOT EXISTS (
        SELECT 1 FROM fact f WHERE f.metric_id = mine.metric_id
          AND f.plan IS NOT NULL AND f.plan <> 0)
),
single AS (
    SELECT name FROM mine WHERE (
        SELECT COUNT(DISTINCT f.period_id) FROM fact f WHERE f.metric_id = mine.metric_id) = 1
)
SELECT (SELECT COUNT(*) FROM no_plan) AS n_no_plan,
       (SELECT group_concat(name, ', ') FROM (SELECT name FROM no_plan LIMIT 8)) AS names_no_plan,
       (SELECT COUNT(*) FROM single) AS n_single_period,
       (SELECT group_concat(name, ', ') FROM (SELECT name FROM single LIMIT 8)) AS names_single_period,
       (SELECT COUNT(*) FROM mine WHERE kn_status IS NULL) AS n_no_knowledge;
