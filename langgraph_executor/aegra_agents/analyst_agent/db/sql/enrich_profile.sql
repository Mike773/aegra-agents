-- name: enrich_profile
-- summary: Состав датасета одной строкой: люди, показатели, периоды, разрезы
-- params:
-- returns: n_people, n_employees, n_metrics, n_roots, max_depth, n_periods, first_date, last_date, calc_periods, n_elements, n_with_plan
-- render: kv
SELECT
    (SELECT COUNT(*) FROM person) AS n_people,
    (SELECT COUNT(*) FROM person WHERE is_me = 0) AS n_employees,
    (SELECT COUNT(*) FROM metric) AS n_metrics,
    (SELECT COUNT(*) FROM metric WHERE depth = 1) AS n_roots,
    (SELECT MAX(depth) FROM metric) AS max_depth,
    (SELECT COUNT(*) FROM period) AS n_periods,
    (SELECT MIN(date) FROM period) AS first_date,
    (SELECT MAX(date) FROM period) AS last_date,
    (SELECT group_concat(cp, ', ') FROM
        (SELECT DISTINCT calc_period AS cp FROM fact WHERE calc_period IS NOT NULL)) AS calc_periods,
    (SELECT COUNT(DISTINCT element) FROM fact WHERE element IS NOT NULL) AS n_elements,
    (SELECT COUNT(*) FROM metric m WHERE EXISTS (
        SELECT 1 FROM fact f WHERE f.metric_id = m.metric_id AND f.plan IS NOT NULL AND f.plan <> 0
     )) AS n_with_plan;
