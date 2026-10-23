-- name: deviation_problems
-- summary: Карта отклонений текстом: строка на показатель — план, динамика, группы с местом, вывод
-- uses: deviation_map
-- params: hard_hit_rate:float = 33.3, rel_tol:float = 1, top_share:float = 20
-- returns: signal, metric, depth, problems
-- render: table
--
-- Читает deviation_map (те же числа, те же пороги) и пишет по каждой
-- строке связный текст для агента:
--   1. План: факт, план, выполнение (rr / ex / факт ÷ план) и сырые ex, rr.
--   2. Динамика к прошлому периоду серии.
--   3. Группы — ВСЕ группы человека от узкой к широкой: место в рейтинге
--      (ranking), медиана, топ-20, где он относительно медианы, сколько
--      коллег выполняют план, среднее выполнение группы.
--   4. Динамика группы: медиана ближайшей группы против своего факта.
--   5. Вывод: как читать отклонение на фоне группы — общая проблема,
--      индивидуальная, мягкий план, в числе лучших и т. п.
--
-- Знак у всех чисел из deviation_map уже учитывает direction: «ухудшение»
-- у обратного показателя — это рост.
--
-- Пороги вывода: hard_hit_rate — план выполняют меньше этой доли коллег
-- (порог «жёсткого» плана, как у plan_rigidity); rel_tol — разница динамик,
-- п.п., в пределах которой сотрудник «идёт вместе с группой»; top_share —
-- какой верх рейтинга, %, считается «в числе лучших».
WITH d AS ({{deviation_map}}),
me AS (
    SELECT person_id FROM person WHERE person_key = :person_key
),
agg AS (                            -- срезы групп человека (как в deviation_map)
    SELECT g.* FROM peer_aggregate g
    WHERE g.metric_id IS NOT NULL AND g.is_current = 1
      AND (g.aggregate_id IS NULL
           OR NOT EXISTS (SELECT 1 FROM person_peer pp JOIN me ON me.person_id = pp.person_id)
           OR EXISTS (SELECT 1 FROM person_peer pp JOIN me ON me.person_id = pp.person_id
                       WHERE pp.aggregate_id = g.aggregate_id))
      AND g.agg_id = (SELECT MIN(g2.agg_id) FROM peer_aggregate g2
                       WHERE g2.metric_id = g.metric_id AND g2.level = g.level
                         AND g2.is_current = 1
                         AND COALESCE(g2.aggregate_id, '') = COALESCE(g.aggregate_id, ''))
),
rnk AS (                            -- место в рейтинге на последнем срезе
    SELECT f.metric_id, r.level, lower(trim(r.level)) AS lkey,
           r.rank_pos, r.rank_total, r.rank_raw, r.percentile
    FROM ranking r
    JOIN fact f ON f.fact_id = r.fact_id
    JOIN me ON me.person_id = f.person_id
    WHERE f.is_last_of_series = 1 AND f.element IS NULL
),
lv_keys AS (                        -- уровни: из агрегатов и из рейтинга
    SELECT metric_id, lower(trim(level)) AS lkey FROM agg
    UNION
    SELECT metric_id, lkey FROM rnk
),
lv AS (
    SELECT k.metric_id,
           COALESCE((SELECT MIN(a.level_order) FROM agg a
                      WHERE a.metric_id = k.metric_id AND lower(trim(a.level)) = k.lkey),
                    (SELECT MIN(pa.level_order) FROM peer_aggregate pa
                      WHERE lower(trim(pa.level)) = k.lkey), 99) AS level_order,
           COALESCE((SELECT a.level_name FROM agg a
                      WHERE a.metric_id = k.metric_id AND lower(trim(a.level)) = k.lkey),
                    (SELECT MIN(pa.level_name) FROM peer_aggregate pa
                      WHERE lower(trim(pa.level)) = k.lkey),
                    (SELECT r.level FROM rnk r
                      WHERE r.metric_id = k.metric_id AND r.lkey = k.lkey)) AS level_name,
           a.total_objects, a.median, a.top20_mean_fact AS top20, a.hit_rate, a.mean_ex,
           r.rank_pos, r.rank_total, r.rank_raw, r.percentile
    FROM lv_keys k
    LEFT JOIN agg a ON a.metric_id = k.metric_id AND lower(trim(a.level)) = k.lkey
    LEFT JOIN rnk r ON r.metric_id = k.metric_id AND r.lkey = k.lkey
),
lv_text AS (                        -- одна группа одной фразой
    SELECT lv.metric_id, lv.level_order, lv.rank_pos, lv.rank_total, lv.level_name,
           lv.level_name
           || COALESCE(' (' || lv.total_objects || ' чел.)', '') || ': '
           || substr(
                COALESCE(', место ' || lv.rank_pos || ' из ' || lv.rank_total,
                         ', место ' || lv.rank_raw, '')
                || CASE WHEN lv.rank_pos IS NULL AND lv.rank_raw IS NULL
                             AND lv.percentile IS NOT NULL
                        THEN ', выше ' || fmt_num(lv.percentile) || ' % коллег' ELSE '' END
                || CASE WHEN lv.median IS NULL OR dm.fact IS NULL THEN ''
                        WHEN (dm.fact - lv.median) * dm.sgn < 0
                        THEN ', ниже медианы ' || fmt_num(lv.median)
                        WHEN lv.top20 IS NOT NULL AND (dm.fact - lv.top20) * dm.sgn >= 0
                        THEN ', на уровне лучших 20 % (их среднее ' || fmt_num(lv.top20) || ')'
                        ELSE ', выше медианы ' || fmt_num(lv.median) END
                || CASE WHEN lv.top20 IS NOT NULL AND dm.fact IS NOT NULL
                             AND (dm.fact - lv.median) * dm.sgn >= 0
                             AND (dm.fact - lv.top20) * dm.sgn < 0
                        THEN ', до лучших 20 % (их среднее ' || fmt_num(lv.top20) || ') не хватает '
                             || fmt_num(ROUND(ABS(dm.fact - lv.top20), 2))
                        ELSE '' END
                || COALESCE(', план выполняют ' || fmt_num(lv.hit_rate) || ' % коллег', '')
                || COALESCE(', среднее выполнение плана ' || fmt_num(lv.mean_ex) || ' %', ''),
                3) AS txt
    FROM lv
    JOIN (SELECT f.metric_id, f.fact,
                 CASE WHEN mm.direction = 'обратная' THEN -1 ELSE 1 END AS sgn
            FROM fact f JOIN me ON me.person_id = f.person_id
            JOIN metric mm ON mm.metric_id = f.metric_id
           WHERE f.is_last_of_series = 1 AND f.element IS NULL) dm
      ON dm.metric_id = lv.metric_id
),
groups AS (
    SELECT metric_id,
           group_concat(txt, '; ') AS groups_txt,
           -- лучшее место среди уровней, где оно в верхних top_share %
           (SELECT 'место ' || t2.rank_pos || ' из ' || t2.rank_total || ' ' || t2.level_name
              FROM lv_text t2
             WHERE t2.metric_id = x.metric_id AND t2.rank_pos IS NOT NULL
               AND t2.rank_total > 0
               AND t2.rank_pos <= MAX(1, t2.rank_total * :top_share / 100.0)
             ORDER BY t2.rank_pos * 1.0 / t2.rank_total, t2.level_order LIMIT 1) AS top_place
    FROM (SELECT * FROM lv_text ORDER BY metric_id, level_order) x
    GROUP BY metric_id
),
txt AS (
    SELECT d.*, m.metric_id, g.groups_txt, g.top_place,
           CASE WHEN d.completion IS NOT NULL OR d.plan_gap_abs IS NOT NULL
                THEN COALESCE(d.plan_gap_pp, d.plan_gap_abs) >= 0 END AS plan_ok,
           -- 1. План
           'План: факт ' || fmt_num(d.fact) || ' при плане ' || fmt_num(d.plan)
           || CASE WHEN d.direction = 'обратная' THEN ' (меньше — лучше)' ELSE '' END
           || CASE
                WHEN d.completion IS NOT NULL
                THEN CASE d.basis
                       WHEN 'rr' THEN '; прогноз на конец периода (run rate) ' || fmt_num(f.rr)
                                      || ', выполнение по прогнозу ' || fmt_num(d.completion) || ' %'
                                      || COALESCE(', на сегодня выполнено ' || fmt_num(f.ex) || ' %', '')
                       ELSE ', выполнение ' || fmt_num(d.completion) || ' %' END
                     || CASE WHEN d.plan_gap_pp >= 0
                             THEN ' — план выполнен, запас ' || fmt_num(d.plan_gap_pp) || ' п.п.'
                             ELSE ' — план не выполнен, отставание ' || fmt_num(-d.plan_gap_pp) || ' п.п.' END
                WHEN d.plan_gap_abs >= 0
                THEN ' — план выполнен, запас ' || fmt_num(d.plan_gap_abs)
                ELSE ' — план не выполнен, отставание ' || fmt_num(-d.plan_gap_abs) END
           AS plan_txt,
           -- 2. Динамика
           CASE WHEN d.prev_date IS NULL THEN 'Динамика: прошлого периода нет'
                ELSE 'Динамика к прошлому периоду (' || d.prev_date || '): '
                     || CASE WHEN COALESCE(d.dyn_pp, d.dyn_abs) > 0 THEN 'улучшение'
                             WHEN COALESCE(d.dyn_pp, d.dyn_abs) < 0 THEN 'ухудшение'
                             ELSE 'без изменений' END
                     || CASE WHEN d.dyn_pp IS NOT NULL AND d.dyn_pp <> 0
                             THEN ' на ' || fmt_num(ABS(d.dyn_pp)) || ' п.п. выполнения (было '
                                  || fmt_num(d.prev_completion) || ' %)'
                             WHEN d.dyn_pp IS NULL AND COALESCE(d.dyn_abs, 0) <> 0
                             THEN ' на ' || fmt_num(ABS(d.dyn_abs))
                             ELSE '' END
                     || COALESCE(', факт ' || CASE WHEN d.fact_change_pct >= 0 THEN 'лучше' ELSE 'хуже' END
                                 || ' на ' || fmt_num(ABS(d.fact_change_pct)) || ' %', '')
           END AS dyn_txt,
           -- 4. Динамика группы
           CASE WHEN d.group_change_pct IS NULL OR d.fact_change_pct IS NULL THEN NULL
                ELSE 'Динамика группы ' || d.group_level || ' к прошлому срезу (' || d.group_prev_date || ')'
                     || ': медиана ' || CASE WHEN d.group_change_pct >= 0 THEN 'лучше' ELSE 'хуже' END
                     || ' на ' || fmt_num(ABS(d.group_change_pct)) || ' %, у сотрудника '
                     || CASE WHEN d.fact_change_pct >= 0 THEN 'лучше' ELSE 'хуже' END
                     || ' на ' || fmt_num(ABS(d.fact_change_pct)) || ' %'
           END AS gdyn_txt
    FROM d
    JOIN metric m ON m.name = d.metric
    JOIN me
    JOIN fact f ON f.person_id = me.person_id AND f.metric_id = m.metric_id
          AND f.element IS NULL AND f.is_last_of_series = 1
    LEFT JOIN groups g ON g.metric_id = m.metric_id
),
concl AS (                          -- 5. Вывод: фразы через «; »
    SELECT t.*,
           substr(
             -- план не выполнен — на фоне группы
             CASE WHEN t.plan_ok = 0 AND t.group_hit_rate < :hard_hit_rate
                  THEN '; план не выполняет и большинство коллег (выполняют '
                       || fmt_num(t.group_hit_rate) || ' %) — похоже на общую проблему группы, а не на личную'
                  WHEN t.plan_ok = 0 AND t.group_zone = 'топ-20'
                  THEN '; план не выполнен, но результат на уровне лучших 20 % группы — план для группы высокий'
                  WHEN t.plan_ok = 0 AND t.group_zone = 'выше медианы'
                  THEN '; план не выполнен, но результат лучше медианы группы — отставание от плана, а не от коллег'
                  WHEN t.plan_ok = 0 AND t.group_zone = 'ниже медианы'
                       AND t.group_hit_rate >= :soft_hit_rate
                  THEN '; большинство коллег план выполняет (' || fmt_num(t.group_hit_rate)
                       || ' %) — отставание личное'
                  WHEN t.plan_ok = 0 AND t.group_zone = 'ниже медианы'
                  THEN '; план не выполнен и результат ниже медианы группы'
                  ELSE '' END
             -- план выполнен — на фоне группы
             || CASE WHEN t.signal = 'мягкий план'
                     THEN '; план выполнен, но он мягкий: его выполняют ' || fmt_num(t.group_hit_rate)
                          || ' % коллег, а результат ниже медианы группы'
                     WHEN t.signal = 'слабый сигнал'
                     THEN '; план выполнен, но результат заметно ниже медианы группы — слабый сигнал'
                     WHEN t.signal = 'сильный результат'
                     THEN '; результат далеко впереди группы — стоит перенять практику'
                     WHEN t.plan_ok = 1 AND t.group_zone = 'топ-20'
                     THEN '; план выполнен, результат на уровне лучших 20 % группы'
                     ELSE '' END
             || CASE WHEN t.top_place IS NOT NULL THEN '; в числе лучших: ' || t.top_place ELSE '' END
             -- устойчивость отставания
             || CASE WHEN t.n_levels > 1 AND t.n_levels_below = t.n_levels
                     THEN '; ниже медианы во всех ' || t.n_levels || ' группах' ELSE '' END
             || CASE WHEN t.periods_below >= 3
                     THEN '; ниже медианы ' || t.periods_below || ' срезов подряд' ELSE '' END
             -- динамика на фоне группы
             || CASE WHEN t.fact_change_pct IS NULL OR t.group_change_pct IS NULL THEN ''
                     WHEN t.fact_change_pct < 0 AND t.group_change_pct < 0
                          AND ABS(t.rel_change_pp) <= :rel_tol
                     THEN '; ухудшение общее для группы — сотрудник снижается вместе с коллегами'
                     WHEN t.fact_change_pct < 0 AND t.group_change_pct >= 0
                     THEN '; ухудшается, хотя группа улучшается'
                     WHEN t.fact_change_pct >= 0 AND t.group_change_pct < 0
                     THEN '; улучшается, хотя группа ухудшается'
                     WHEN t.rel_change_pp < -:rel_tol
                     THEN '; динамика хуже группы на ' || fmt_num(-t.rel_change_pp) || ' п.п.'
                     WHEN t.rel_change_pp > :rel_tol
                     THEN '; динамика лучше группы на ' || fmt_num(t.rel_change_pp) || ' п.п.'
                     ELSE '' END,
             3) AS concl_txt
    FROM txt t
)
SELECT c.signal, c.metric, c.depth,
       trim(c.metric
       || CASE WHEN c.scope <> 'показатель' THEN ' [' || c.scope || ']' ELSE '' END
       || CASE WHEN c.depth > 1 AND c.parent_name IS NOT NULL
               THEN ' (влияет на «' || c.parent_name || '»'
                    || COALESCE(', доля влияния ' || fmt_num(c.influent_percent) || ' %', '') || ')'
               ELSE '' END
       || ' — ' || c.signal || '. '
       || rtrim(c.plan_txt, '.') || '. '
       || rtrim(c.dyn_txt, '.') || '. '
       || CASE WHEN c.groups_txt IS NOT NULL THEN 'Группы: ' || rtrim(c.groups_txt, '.') || '. '
               ELSE 'Данных по группам нет. ' END
       || COALESCE(rtrim(c.gdyn_txt, '.') || '. ', '')
       || CASE WHEN c.concl_txt <> '' THEN 'Вывод: ' || rtrim(c.concl_txt, '.') || '.' ELSE '' END)
           AS problems
FROM concl c
ORDER BY CASE c.signal WHEN 'отклонение' THEN 1 WHEN 'мягкий план' THEN 2
                       WHEN 'слабый сигнал' THEN 3 WHEN 'сильный результат' THEN 4 ELSE 5 END,
         c.depth, c.n_bad DESC, c.plan_gap_pp IS NULL, c.plan_gap_pp, c.peer_dist, c.metric;
