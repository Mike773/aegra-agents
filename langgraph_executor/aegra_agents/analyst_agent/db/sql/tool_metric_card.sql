-- name: tool_metric_card
-- summary: Значения показателя по периодам у человека (агрегатная строка)
-- params: person_key:str, metric:str, row_limit:int = 40
-- returns: date, fact, plan, ex, rr, plan_status, plan_dev_pct, pop_change_pct, pop_status, trend_status, rel_status, is_anomaly, star_received
-- render: table
SELECT date, fact, plan, ex, rr, plan_status, plan_dev_pct, pop_change_pct,
       pop_status, trend_status, rel_status, is_anomaly, star_received
FROM v_fact
WHERE person_key = :person_key
  AND ru_lower(metric) = ru_lower(:metric)
  AND element IS NULL
ORDER BY period_idx DESC
LIMIT :row_limit;
