-- name: tool_metric_elements
-- summary: Разрезы показателя за последнее значение каждой серии, худшие сверху
-- params: person_key:str, metric:str, worst_first:bool = 1, row_limit:int = 8
-- returns: element, date, fact, plan, plan_dev_pct, plan_status, pop_change_pct, pop_status
-- render: table
--
-- Разрезов бывают сотни, поэтому инструмент отдаёт только край выборки:
-- худшие (worst_first=1) или лучшие (worst_first=0). Сортировка по отклонению
-- от плана, а если плана нет — по факту с учётом направления показателя, иначе
-- у беспланового показателя порядок вырождается в нули.
SELECT element, date, fact, plan, plan_dev_pct, plan_status, pop_change_pct, pop_status
FROM v_fact_latest
WHERE person_key = :person_key
  AND ru_lower(metric) = ru_lower(:metric)
  AND element IS NOT NULL
ORDER BY CASE WHEN :worst_first THEN 1 ELSE -1 END *
         CASE
             WHEN plan_dev_pct IS NOT NULL
                  THEN CASE WHEN direction = 'обратная' THEN -plan_dev_pct ELSE plan_dev_pct END
             ELSE CASE WHEN direction = 'обратная' THEN -fact ELSE fact END
         END
LIMIT :row_limit;
