-- name: enrich_peer
-- summary: Позиция человека в группах сравнения по уровням (от узкой к широкой)
-- params: person_key:str, row_limit:int = 40
-- returns: level_name, total_objects, metric, mean_fact, median, top20_mean_fact, hit_rate, rank_raw, percentile
-- render: table
SELECT level_name, total_objects, metric, mean_fact, median, top20_mean_fact,
       hit_rate, rank_raw, percentile
FROM v_peer_latest
WHERE person_key = :person_key AND level_name IS NOT NULL
ORDER BY level_order, metric
LIMIT :row_limit;
