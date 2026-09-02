-- name: enrich_scoreboard
-- summary: Показатели верхнего уровня человека — последнее значение КАЖДОЙ серии
-- params: person_key:str, max_depth:int = 1, row_limit:int = 200
-- returns: metric, unit, date, fact, plan, ex, plan_status, plan_dev_pct, pop_status, pop_change_pct, trend_status, rel_status, kind, n_children, n_elements, has_aggregate, n_bad_elements, worst_element, zone, signal
-- render: table
--
-- Дата берётся своя у каждой серии (is_last_of_series), поэтому показатель с
-- отстающими данными не пропадает; сортировка: сначала зоны внимания, внутри —
-- по силе сигнала.
--
-- Две ветки. Первая — обычные показатели: берём строку без разреза, это
-- собственный итог. Вторая — показатели, у которых собственного итога НЕТ и
-- есть только разрезы: их нельзя терять, но и итог им считать нельзя (сложить
-- проценты, средние или ранги бессмысленно), поэтому вместо факта отдаём
-- количество разрезов, сколько из них хуже плана и худший разрез как иллюстрацию.
SELECT * FROM (
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
       COALESCE(mp.n_elements, 0) AS n_elements,
       1 AS has_aggregate,
       NULL AS n_bad_elements,
       NULL AS worst_element,
       CASE
           WHEN v.plan_status = 'хуже_плана' OR v.pop_status = 'ухудшение'
                OR v.trend_status = 'ухудшение' OR v.is_anomaly = 1 THEN 'внимание'
           WHEN v.plan_status = 'лучше_плана' OR v.pop_status = 'улучшение' THEN 'достижение'
           ELSE 'стабильно'
       END AS zone,
       COALESCE(ABS(v.pop_change_pct), ABS(v.plan_dev_pct), 0) AS signal
FROM v_fact_latest v
JOIN v_metric m ON m.name = v.metric
LEFT JOIN v_metric_person mp
       ON mp.metric = v.metric AND mp.person_key = v.person_key
WHERE v.person_key = :person_key
  AND v.element IS NULL
  AND v.depth <= :max_depth
  AND v.star_received IS NULL

UNION ALL

SELECT mp.metric,
       mp.unit,
       (SELECT MAX(e.date) FROM v_fact_latest e
         WHERE e.person_key = mp.person_key AND e.metric = mp.metric
           AND e.element IS NOT NULL) AS date,
       NULL AS fact,
       NULL AS plan,
       NULL AS ex,
       NULL AS plan_status,
       NULL AS plan_dev_pct,
       NULL AS pop_status,
       NULL AS pop_change_pct,
       NULL AS trend_status,
       NULL AS rel_status,
       mp.kind,
       m.n_children,
       mp.n_elements,
       0 AS has_aggregate,
       (SELECT COUNT(*) FROM v_fact_latest e
         WHERE e.person_key = mp.person_key AND e.metric = mp.metric
           AND e.element IS NOT NULL AND e.plan_status = 'хуже_плана') AS n_bad_elements,
       (SELECT e.element FROM v_fact_latest e
         WHERE e.person_key = mp.person_key AND e.metric = mp.metric
           AND e.element IS NOT NULL
         ORDER BY CASE WHEN e.plan_dev_pct IS NOT NULL
                       THEN CASE WHEN e.direction = 'обратная'
                                 THEN -e.plan_dev_pct ELSE e.plan_dev_pct END
                       ELSE CASE WHEN e.direction = 'обратная'
                                 THEN -e.fact ELSE e.fact END END
         LIMIT 1) AS worst_element,
       CASE WHEN EXISTS (SELECT 1 FROM v_fact_latest e
                          WHERE e.person_key = mp.person_key AND e.metric = mp.metric
                            AND e.element IS NOT NULL
                            AND e.plan_status = 'хуже_плана')
            THEN 'внимание' ELSE 'стабильно' END AS zone,
       0 AS signal
FROM v_metric_person mp
JOIN v_metric m ON m.name = mp.metric
WHERE mp.person_key = :person_key
  AND mp.has_aggregate = 0
  AND mp.n_elements > 0
  AND mp.depth <= :max_depth
  AND m.is_star = 0

)
ORDER BY CASE zone WHEN 'внимание' THEN 0 WHEN 'достижение' THEN 1 ELSE 2 END,
         signal DESC,
         metric
LIMIT :row_limit;
