-- name: enrich_zones
-- summary: Счётчики зон внимания (последние значения серий) до заданной глубины
-- params: person_key:str, max_depth:int = 1
-- returns: n_below_plan, n_declining, n_anomaly, n_peer_worse, n_group_worse, n_above_plan, n_improving
-- render: kv
SELECT
    SUM(plan_status = 'хуже_плана') AS n_below_plan,
    SUM(pop_status = 'ухудшение' OR trend_status = 'ухудшение') AS n_declining,
    SUM(is_anomaly = 1) AS n_anomaly,
    SUM(peer_status = 'хуже_коллег') AS n_peer_worse,
    SUM(rel_status = 'хуже_группы') AS n_group_worse,
    SUM(plan_status = 'лучше_плана') AS n_above_plan,
    SUM(pop_status = 'улучшение') AS n_improving
FROM v_fact_latest
WHERE person_key = :person_key AND element IS NULL AND depth <= :max_depth;
