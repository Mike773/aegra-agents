-- name: tool_metric_elements
-- summary: Разрезы показателя за последнее значение каждой серии
-- params: person_key:str, metric:str, row_limit:int = 20
-- returns: element, date, fact, plan, plan_dev_pct, plan_status, pop_change_pct, pop_status
-- render: table
SELECT element, date, fact, plan, plan_dev_pct, plan_status, pop_change_pct, pop_status
FROM v_fact_latest
WHERE person_key = :person_key
  AND ru_lower(metric) = ru_lower(:metric)
  AND element IS NOT NULL
ORDER BY CASE WHEN direction = 'обратная' THEN -COALESCE(plan_dev_pct, 0)
              ELSE COALESCE(plan_dev_pct, 0) END
LIMIT :row_limit;
