-- name: enrich_scoreboard
-- summary: Показатели верхнего уровня человека — последнее значение КАЖДОЙ серии
-- params: person_key:str, max_depth:int = 1, row_limit:int = 200
-- returns: metric, unit, date, fact, plan, ex, plan_status, plan_dev_pct, pop_status, pop_change_pct, trend_status, rel_status, kind, n_children, n_elements, zone, signal
-- render: table
--
-- Дата берётся своя у каждой серии (is_last_of_series), поэтому показатель с
-- отстающими данными не пропадает; сортировка: сначала зоны внимания, внутри —
-- по силе сигнала.
SELECT v.metric,
       v.unit,
       v.date,
       v.fact,
       v.plan,
       v.ex,
       v.plan_status,
       v.plan_dev_pct,
       v.pop_status,
       v.pop_change_pct,
       v.trend_status,
       v.rel_status,
       v.kind,
       m.n_children,
       m.n_elements,
       CASE
           WHEN v.plan_status = 'хуже_плана' OR v.pop_status = 'ухудшение'
                OR v.trend_status = 'ухудшение' OR v.is_anomaly = 1 THEN 'внимание'
           WHEN v.plan_status = 'лучше_плана' OR v.pop_status = 'улучшение' THEN 'достижение'
           ELSE 'стабильно'
       END AS zone,
       COALESCE(ABS(v.pop_change_pct), ABS(v.plan_dev_pct), 0) AS signal
FROM v_fact_latest v
JOIN v_metric m ON m.name = v.metric
WHERE v.person_key = :person_key
  AND v.element IS NULL
  AND v.depth <= :max_depth
  AND v.star_received IS NULL
ORDER BY CASE zone WHEN 'внимание' THEN 0 WHEN 'достижение' THEN 1 ELSE 2 END,
         signal DESC,
         v.metric
LIMIT :row_limit;
