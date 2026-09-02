-- name: enrich_zones
-- summary: Счётчики зон внимания (последние значения серий) до заданной глубины
-- params: person_key:str, max_depth:int = 1
-- returns: n_below_plan, n_declining, n_anomaly, n_peer_worse, n_group_worse, n_above_plan, n_improving
-- render: kv
--
-- Считаем по одной строке на показатель: у обычных — его собственный итог
-- (строка без разреза), у показателей без итога — их разрезы, свёрнутые в один
-- признак (иначе такой показатель либо пропадал бы из счётчиков, либо считался
-- бы столько раз, сколько у него разрезов).
WITH per_metric AS (
    SELECT metric,
           MAX(plan_status = 'хуже_плана') AS below_plan,
           MAX(pop_status = 'ухудшение' OR trend_status = 'ухудшение') AS declining,
           MAX(is_anomaly = 1) AS anomaly,
           MAX(peer_status = 'хуже_коллег') AS peer_worse,
           MAX(rel_status = 'хуже_группы') AS group_worse,
           MAX(plan_status = 'лучше_плана') AS above_plan,
           MAX(pop_status = 'улучшение') AS improving
    FROM v_fact_latest v
    WHERE v.person_key = :person_key
      AND v.depth <= :max_depth
      AND (
            v.element IS NULL
            OR NOT EXISTS (SELECT 1 FROM v_metric_person mp
                            WHERE mp.person_key = v.person_key AND mp.metric = v.metric
                              AND mp.has_aggregate = 1)
          )
    GROUP BY metric
)
SELECT SUM(below_plan) AS n_below_plan,
       SUM(declining) AS n_declining,
       SUM(anomaly) AS n_anomaly,
       SUM(peer_worse) AS n_peer_worse,
       SUM(group_worse) AS n_group_worse,
       SUM(above_plan) AS n_above_plan,
       SUM(improving) AS n_improving
FROM per_metric;
