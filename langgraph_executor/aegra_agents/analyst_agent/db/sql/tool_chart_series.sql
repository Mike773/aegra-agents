-- name: tool_chart_series
-- summary: Ряд показателя для графика: последние N периодов — факт, план, медиана ближайшей группы
-- params: person_key:str, metric:str, element:str? = NULL, periods:int = 6
-- returns: date, fact, plan, median, group_name
-- render: table
--
-- Числа графика берёт код, а не модель: на большом объёме GigaChat путает
-- значения, поэтому ряд ограничен последними periods точками серии.
-- Медиана — ближайшая (самая узкая) группа человека, срез того же месяца
-- (агрегаты помесячные, как в analytics.py); у разреза медианы нет.
WITH me AS (
    SELECT person_id FROM person WHERE person_key = :person_key
),
m AS (
    SELECT metric_id FROM metric WHERE ru_lower(name) = ru_lower(:metric)
),
pts AS (
    SELECT f.fact, f.plan, d.date, d.ym, d.period_idx
    FROM fact f
    JOIN me ON me.person_id = f.person_id
    JOIN m ON m.metric_id = f.metric_id
    JOIN period d ON d.period_id = f.period_id
    WHERE COALESCE(f.element, '') = COALESCE(:element, '')
    ORDER BY d.period_idx DESC
    LIMIT :periods
),
agg AS (                            -- срезы групп, доступные человеку
    SELECT g.* FROM peer_aggregate g
    JOIN m ON m.metric_id = g.metric_id
    WHERE :element IS NULL
      AND (g.aggregate_id IS NULL
           OR NOT EXISTS (SELECT 1 FROM person_peer pp JOIN me ON me.person_id = pp.person_id)
           OR EXISTS (SELECT 1 FROM person_peer pp JOIN me ON me.person_id = pp.person_id
                       WHERE pp.aggregate_id = g.aggregate_id))
),
near AS (
    SELECT level, aggregate_id, level_name FROM agg
    WHERE is_current = 1
    ORDER BY level_order, agg_id
    LIMIT 1
)
SELECT p.date, p.fact, p.plan,
       (SELECT h.median FROM agg h JOIN near n
           ON h.level = n.level AND COALESCE(h.aggregate_id, '') = COALESCE(n.aggregate_id, '')
         WHERE h.ym = p.ym
         ORDER BY h.dt DESC LIMIT 1) AS median,
       (SELECT level_name FROM near) AS group_name
FROM pts p
ORDER BY p.period_idx;
