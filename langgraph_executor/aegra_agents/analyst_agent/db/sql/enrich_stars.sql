-- name: enrich_stars
-- summary: Звёзды человека и их влияющие показатели за последние даты серий
-- params: person_key:str
-- returns: star, received, star_date, child, child_fact, child_plan, child_plan_status, child_date
-- render: table
SELECT star, received, star_date, child, child_fact, child_plan, child_plan_status, child_date
FROM v_star
WHERE person_key = :person_key
ORDER BY star, child_plan_status DESC, child;
