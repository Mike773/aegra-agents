-- name: enrich_stars
-- summary: Звёзды человека за последний и предыдущий период — строка на (звезда, дата) с влияющими показателями
-- params: person_key:str
-- returns: star, star_date, period_rank, received, n_metrics, n_below_plan, metrics
-- render: table
--
-- Звезда приходит копией узла на каждый период; берём два последних периода
-- каждой звезды (period_rank 1 и 2). Влияющие показатели (is_star_metric) —
-- из дерева ЭТОГО человека за период звезды, одной строкой: «имя: факт X при
-- плане Y, выполнение Z %, вердикт; …». Порядок: последний период первым,
-- внутри периода неполученные звёзды выше полученных.
SELECT star, star_date, period_rank, received, n_metrics, n_below_plan, metrics
FROM v_star
WHERE person_key = :person_key AND period_rank <= 2
ORDER BY period_rank, received, star;
