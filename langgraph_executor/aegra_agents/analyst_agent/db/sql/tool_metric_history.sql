-- name: tool_metric_history
-- summary: Ряд значений показателя по периодам у человека (агрегат или разрез)
-- params: person_key:str, metric:str, element:str? = NULL, row_limit:int = 60
-- returns: date, fact, plan, ex, rr, plan_status, pop_change_pct, pop_status, trend_status
-- render: table
SELECT date, fact, plan, ex, rr, plan_status, pop_change_pct, pop_status, trend_status
FROM v_fact
WHERE person_key = :person_key
  AND ru_lower(metric) = ru_lower(:metric)
  AND COALESCE(element, '') = COALESCE(:element, '')
ORDER BY period_idx
LIMIT :row_limit;
