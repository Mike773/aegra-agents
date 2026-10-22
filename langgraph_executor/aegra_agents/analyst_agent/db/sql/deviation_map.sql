-- name: deviation_map
-- summary: Карта отклонений человека: план, динамика, группа, динамика к группе — строка на показатель
-- params: person_key:str, max_depth:int = 9
-- returns: metric, depth, parent_name, influent_percent, kind, direction, scope, date, fact, plan, basis, completion, plan_gap_pp, plan_gap_abs, prev_date, prev_completion, dyn_pp, dyn_abs, group_level, group_size, group_date, median, top20, median_gap_pct, top20_gap_pct, group_zone, fact_change_pct, group_prev_date, group_change_pct, rel_change_pp, n_bad, n_good
-- render: table
--
-- Строка на показатель человека: последний срез серии без разреза
-- (element IS NULL). Показатели без плана в карту не попадают.
--
-- Знак: у всех *_pp / *_pct «минус» = хуже, «плюс» = лучше, с учётом
-- direction (у обратных показателей рост — это ухудшение).
--
-- 1. Выполнение плана (completion, %): rr, если он есть, иначе ex, иначе
--    факт / план. basis — что взято. plan_gap_pp = completion − 100.
-- 2. Динамика (dyn_pp): completion сейчас минус completion прошлого периода
--    ЭТОЙ серии, по тому же основанию (rr с rr, ex с ex).
-- 3. Группа: самая узкая из групп человека, где есть этот показатель
--    (минимальный level_order текущего среза). Факт против медианы и
--    среднего топ-20 %. group_zone: топ-20 / выше медианы / ниже медианы.
-- 4. Динамика к группе (rel_change_pp): изменение факта человека, % минус
--    изменение медианы группы, % (прошлый срез группы — предыдущий dt).
--
-- Индексы (kind = 'индекс': CSI, ранги, места) процентов не получают —
-- деление на базу у них бессмысленно. План и динамика у них — в единицах
-- показателя (plan_gap_abs, dyn_abs), в группе — только зона.
--
-- Звёзды: если у человека есть звёзды, в карту идут ТОЛЬКО влияющие
-- показатели (is_star_metric) звёзд, не полученных в последнем периоде.
-- scope = 'звезда: <имя>'. Все звёзды получены — карта пустая.
WITH me AS (
    SELECT person_id FROM person WHERE person_key = :person_key
),
open_star AS (                      -- не полученные звёзды последнего периода
    SELECT s.metric_id AS star_id, m.name AS star
    FROM fact s
    JOIN me ON me.person_id = s.person_id
    JOIN metric m ON m.metric_id = s.metric_id AND m.is_star = 1
    WHERE s.star_received IS NOT NULL AND s.is_last_of_series = 1
      AND s.star_received = 0
),
has_stars AS (
    SELECT EXISTS (SELECT 1 FROM fact s JOIN me ON me.person_id = s.person_id
                    WHERE s.star_received IS NOT NULL) AS yes
),
star_scope AS (                     -- влияющие показатели открытых звёзд
    SELECT e.child_id AS metric_id,
           'звезда: ' || group_concat(o.star, ', ') AS scope
    FROM open_star o
    JOIN me
    JOIN metric_edge e ON e.person_id = me.person_id AND e.parent_id = o.star_id
    JOIN metric cm ON cm.metric_id = e.child_id AND cm.is_star_metric = 1
    GROUP BY e.child_id
),
cur AS (
    SELECT f.fact_id, f.person_id, f.metric_id, f.period_id, f.fact, f.plan, f.ex, f.rr,
           m.name AS metric, m.kind, m.direction,
           CASE WHEN m.direction = 'обратная' THEN -1 ELSE 1 END AS sgn,
           COALESCE(t.depth, m.depth) AS depth,
           CASE WHEN t.metric_id IS NULL THEN m.parent_name ELSE t.parent_name END AS parent_name,
           CASE WHEN t.metric_id IS NULL THEN m.influent_percent ELSE t.influent_percent END
               AS influent_percent,
           d.date,
           CASE WHEN f.rr IS NOT NULL THEN 'rr' WHEN f.ex IS NOT NULL THEN 'ex'
                ELSE 'факт/план' END AS basis,
           CASE WHEN (SELECT yes FROM has_stars) = 1 THEN ss.scope ELSE 'показатель' END AS scope,
           (SELECT MAX(p2.period_id) FROM fact p2
             WHERE p2.person_id = f.person_id AND p2.metric_id = f.metric_id
               AND p2.element IS NULL AND p2.period_id < f.period_id) AS prev_period_id
    FROM fact f
    JOIN me ON me.person_id = f.person_id
    JOIN metric m ON m.metric_id = f.metric_id AND m.is_star = 0
    JOIN period d ON d.period_id = f.period_id
    LEFT JOIN metric_tree t ON t.person_id = f.person_id AND t.metric_id = f.metric_id
    LEFT JOIN star_scope ss ON ss.metric_id = f.metric_id
    WHERE f.is_last_of_series = 1 AND f.element IS NULL
      AND f.plan IS NOT NULL AND f.plan <> 0
      AND COALESCE(t.depth, m.depth) <= :max_depth
      AND ((SELECT yes FROM has_stars) = 0 OR ss.metric_id IS NOT NULL)
),
-- rr — прогноз результата к концу периода в единицах факта, поэтому в
-- процент выполнения он переводится делением на план, как и факт.
pct AS (
    SELECT c.*, pd.date AS prev_date, pf.fact AS prev_fact,
           CASE c.basis WHEN 'rr' THEN c.rr * 100.0 / c.plan
                        WHEN 'ex' THEN c.ex
                        ELSE c.fact * 100.0 / c.plan END AS completion,
           CASE WHEN pf.plan IS NULL OR pf.plan = 0 THEN NULL
                WHEN c.basis = 'rr' THEN pf.rr * 100.0 / pf.plan
                WHEN c.basis = 'ex' THEN pf.ex
                ELSE pf.fact * 100.0 / pf.plan END AS prev_completion
    FROM cur c
    LEFT JOIN fact pf ON pf.person_id = c.person_id AND pf.metric_id = c.metric_id
          AND pf.element IS NULL AND pf.period_id = c.prev_period_id
    LEFT JOIN period pd ON pd.period_id = c.prev_period_id
),
-- Срезы групп, доступные человеку (та же логика, что в v_peer_latest).
agg AS (
    SELECT g.* FROM peer_aggregate g
    WHERE g.metric_id IS NOT NULL
      AND (g.aggregate_id IS NULL
           OR NOT EXISTS (SELECT 1 FROM person_peer pp JOIN me ON me.person_id = pp.person_id)
           OR EXISTS (SELECT 1 FROM person_peer pp JOIN me ON me.person_id = pp.person_id
                       WHERE pp.aggregate_id = g.aggregate_id))
),
grp AS (                            -- ближайшая группа: узкая, текущий срез
    SELECT p.metric_id,
           (SELECT g.agg_id FROM agg g
             WHERE g.metric_id = p.metric_id AND g.is_current = 1
             ORDER BY g.level_order, g.agg_id LIMIT 1) AS agg_id
    FROM pct p
),
grp2 AS (
    SELECT gr.metric_id, g.level_name, g.total_objects, g.dt, g.median,
           g.top20_mean_fact AS top20,
           (SELECT h.median FROM agg h
             WHERE h.metric_id = g.metric_id AND h.level = g.level
               AND COALESCE(h.aggregate_id, '') = COALESCE(g.aggregate_id, '')
               AND h.dt < g.dt
             ORDER BY h.dt DESC LIMIT 1) AS prev_median,
           (SELECT h.dt FROM agg h
             WHERE h.metric_id = g.metric_id AND h.level = g.level
               AND COALESCE(h.aggregate_id, '') = COALESCE(g.aggregate_id, '')
               AND h.dt < g.dt
             ORDER BY h.dt DESC LIMIT 1) AS prev_dt
    FROM grp gr JOIN agg g ON g.agg_id = gr.agg_id
),
sig AS (
    SELECT p.metric, p.depth, p.parent_name, p.influent_percent, p.kind, p.direction,
           p.scope, p.date, p.fact, p.plan, p.basis,
           CASE WHEN p.kind = 'индекс' THEN NULL ELSE ROUND(p.completion, 1) END AS completion,
           CASE WHEN p.kind = 'индекс' THEN NULL
                ELSE ROUND((p.completion - 100) * p.sgn, 1) END AS plan_gap_pp,
           ROUND((p.fact - p.plan) * p.sgn, 2) AS plan_gap_abs,
           p.prev_date,
           CASE WHEN p.kind = 'индекс' THEN NULL ELSE ROUND(p.prev_completion, 1) END
               AS prev_completion,
           CASE WHEN p.kind = 'индекс' THEN NULL
                ELSE ROUND((p.completion - p.prev_completion) * p.sgn, 1) END AS dyn_pp,
           ROUND((p.fact - p.prev_fact) * p.sgn, 2) AS dyn_abs,
           g.level_name AS group_level, g.total_objects AS group_size, g.dt AS group_date,
           g.median, g.top20,
           CASE WHEN p.kind = 'индекс' OR g.median IS NULL OR g.median = 0 THEN NULL
                ELSE ROUND((p.fact - g.median) * 100.0 / ABS(g.median) * p.sgn, 1) END
               AS median_gap_pct,
           CASE WHEN p.kind = 'индекс' OR g.top20 IS NULL OR g.top20 = 0 THEN NULL
                ELSE ROUND((p.fact - g.top20) * 100.0 / ABS(g.top20) * p.sgn, 1) END
               AS top20_gap_pct,
           -- Сначала медиана: если top20 у обратного показателя посчитан без
           -- учёта направления, хуже медианы всё равно не станет «топ-20».
           CASE WHEN g.median IS NULL OR p.fact IS NULL THEN NULL
                WHEN (p.fact - g.median) * p.sgn < 0 THEN 'ниже медианы'
                WHEN g.top20 IS NOT NULL AND (p.fact - g.top20) * p.sgn >= 0 THEN 'топ-20'
                ELSE 'выше медианы' END AS group_zone,
           CASE WHEN p.prev_fact IS NULL OR p.prev_fact = 0 OR p.kind = 'индекс' THEN NULL
                ELSE ROUND((p.fact - p.prev_fact) * 100.0 / ABS(p.prev_fact) * p.sgn, 1) END
               AS fact_change_pct,
           g.prev_dt AS group_prev_date,
           CASE WHEN g.prev_median IS NULL OR g.prev_median = 0 OR p.kind = 'индекс' THEN NULL
                ELSE ROUND((g.median - g.prev_median) * 100.0 / ABS(g.prev_median) * p.sgn, 1) END
               AS group_change_pct
    FROM pct p LEFT JOIN grp2 g ON g.metric_id = p.metric_id
)
SELECT s.*,
       ROUND(s.fact_change_pct - s.group_change_pct, 1) AS rel_change_pp,
       -- Сколько из четырёх сигналов в минус / в плюс (нет данных — 0).
       COALESCE(COALESCE(s.plan_gap_pp, s.plan_gap_abs) < 0, 0)
           + COALESCE(COALESCE(s.dyn_pp, s.dyn_abs) < 0, 0)
           + COALESCE(s.group_zone = 'ниже медианы', 0)
           + COALESCE(s.fact_change_pct - s.group_change_pct < 0, 0) AS n_bad,
       COALESCE(COALESCE(s.plan_gap_pp, s.plan_gap_abs) >= 0, 0)
           + COALESCE(COALESCE(s.dyn_pp, s.dyn_abs) > 0, 0)
           + COALESCE(s.group_zone = 'топ-20', 0)
           + COALESCE(s.fact_change_pct - s.group_change_pct > 0, 0) AS n_good
FROM sig s
ORDER BY s.depth, n_bad DESC, s.plan_gap_pp IS NULL, s.plan_gap_pp, s.metric;
