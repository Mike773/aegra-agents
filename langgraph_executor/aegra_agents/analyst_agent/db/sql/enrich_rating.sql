-- name: enrich_rating
-- summary: Рейтинг по звёздам — место человека по уровням за каждый квартал
-- params: person_key:str, year:int? = NULL, quarter:int? = NULL, row_limit:int = 40
-- returns: year, quarter, period, level, level_name, place, staff, is_latest
-- render: table
--
-- Уровни идут от узкой группы к широкой (level_order из агрегатов), кварталы —
-- от свежего к старому. year/quarter ограничивают выборку одним кварталом.
SELECT year, quarter, period, level, level_name, place, staff, is_latest
FROM v_rating
WHERE person_key = :person_key
  AND (:year IS NULL OR year = :year)
  AND (:quarter IS NULL OR quarter = :quarter)
ORDER BY year DESC, quarter DESC, level_order, level
LIMIT :row_limit;
