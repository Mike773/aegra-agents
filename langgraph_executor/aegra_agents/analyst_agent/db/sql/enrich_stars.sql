-- name: enrich_stars
-- summary: Звёзды человека — строка на звезду с её влияющими показателями
-- params: person_key:str
-- returns: star, received, star_date, n_metrics, n_below_plan, metrics
-- render: table
--
-- Влияющие показатели (is_star_metric) берутся из дерева ЭТОГО человека и
-- идут одной строкой: «имя: факт X при плане Y, выполнение Z %, вердикт; …».
SELECT star, received, star_date, n_metrics, n_below_plan, metrics
FROM v_star
WHERE person_key = :person_key
ORDER BY received, star;
