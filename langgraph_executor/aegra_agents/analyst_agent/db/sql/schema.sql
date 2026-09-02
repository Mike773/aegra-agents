-- Нормализованная схема in-memory SQLite для analyst_agent.
--
-- Базовые таблицы служебные: агент ходит во ВЬЮ (v_fact и остальные) —
-- там уже склеены персона, метрика, период и готовые вердикты.
--
-- Даты: «последний период» считается НА СЕРИЮ (персона × метрика × разрез)
-- через fact.is_last_of_series, а НЕ по глобальной дате датасета. Дочерние
-- показатели могут отставать по датам от родительских — это нормально, они
-- всё равно попадают в v_fact_latest со своей датой. Дерево метрик
-- (metric_edge, depth, path) строится только из структуры JSON и от дат не
-- зависит вовсе.

CREATE TABLE person (
    person_id   INTEGER PRIMARY KEY,
    person_key  TEXT NOT NULL UNIQUE,   -- табельный | ФИО | personN
    tabnum      INTEGER,
    fio         TEXT,
    post        TEXT,
    depart      TEXT,
    is_me       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE metric (
    metric_id      INTEGER PRIMARY KEY,
    metric_key     TEXT NOT NULL UNIQUE,   -- sha256(norm(имя)+norm(описание))
    name_key       TEXT NOT NULL,          -- sha256(norm(имя)) — фолбэк кэша знаний
    ext_id         TEXT,                   -- поле id из JSON (для инсайтов)
    name           TEXT NOT NULL UNIQUE,
    name_norm      TEXT NOT NULL,
    description    TEXT,
    direction      TEXT,                   -- 'прямая' | 'обратная' | NULL
    unit           TEXT,
    calc_period    TEXT,
    kind           TEXT,                   -- 'уровень' | 'вклад' | 'индекс'
    is_star        INTEGER NOT NULL DEFAULT 0,
    is_star_metric INTEGER NOT NULL DEFAULT 0,
    star_of        TEXT,
    depth          INTEGER,
    root_name      TEXT,
    path           TEXT,                   -- 'A > B > C'
    parent_name    TEXT,                   -- доминирующий родитель (max influent)
    influent_percent REAL,
    -- Знания из wiki (materialized из wiki_rag.metric_knowledge); NULL = нет в кэше
    kn_status TEXT, kn_summary TEXT, kn_aliases TEXT, kn_cumulative INTEGER,
    kn_plan_semantics TEXT, kn_fact_semantics TEXT, kn_processes TEXT,
    kn_notes TEXT, kn_confidence REAL
);
CREATE INDEX ix_metric_name_norm ON metric(name_norm);

CREATE TABLE metric_edge (
    parent_id INTEGER NOT NULL REFERENCES metric(metric_id),
    child_id  INTEGER NOT NULL REFERENCES metric(metric_id),
    influent_percent REAL,
    edge_kind TEXT NOT NULL DEFAULT 'tree',
    PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE period (
    period_id  INTEGER PRIMARY KEY,
    date       TEXT NOT NULL UNIQUE,   -- ISO YYYY-MM-DD
    period_idx INTEGER NOT NULL,       -- 1..N по возрастанию даты
    ym         TEXT,                   -- 'YYYY-MM' — ключ джойна с агрегатами
    is_latest  INTEGER NOT NULL DEFAULT 0   -- последняя дата ВСЕГО датасета (справочно)
);

CREATE TABLE fact (
    fact_id   INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES person(person_id),
    metric_id INTEGER NOT NULL REFERENCES metric(metric_id),
    period_id INTEGER NOT NULL REFERENCES period(period_id),
    element   TEXT,                    -- NULL = строка без разреза (агрегат)
    fact REAL, plan REAL, benchmark REAL, ex REAL, rr REAL,
    star_received INTEGER,             -- 1/0 только у звезды
    calc_period TEXT,
    src_ord INTEGER NOT NULL,
    is_last_of_series INTEGER NOT NULL DEFAULT 0  -- последняя дата ЭТОЙ серии
);
CREATE UNIQUE INDEX ux_fact_key
    ON fact(person_id, metric_id, period_id, COALESCE(element, ''));
CREATE INDEX ix_fact_metric_period ON fact(metric_id, period_id);
CREATE INDEX ix_fact_person ON fact(person_id);

CREATE TABLE fact_analytics (
    fact_id INTEGER PRIMARY KEY REFERENCES fact(fact_id),
    plan_dev_abs REAL, plan_dev_pct REAL, plan_status TEXT,
    benchmark_dev_abs REAL, benchmark_dev_pct REAL, benchmark_status TEXT,
    pop_change_abs REAL, pop_change_pct REAL, pop_status TEXT,
    trend TEXT, trend_status TEXT,
    peer_mean REAL, peer_std REAL, peer_count INTEGER, peer_rank INTEGER,
    peer_percentile REAL, zscore REAL, peer_status TEXT, is_anomaly INTEGER,
    group_change_pct REAL, rel_change_pct REAL, rel_status TEXT,
    group_hit_rate REAL, plan_rigidity TEXT
);

CREATE TABLE ranking (
    fact_id INTEGER NOT NULL REFERENCES fact(fact_id),
    level TEXT NOT NULL,
    rank_pos INTEGER, rank_total INTEGER, rank_raw TEXT, percentile REAL,
    PRIMARY KEY (fact_id, level)
);

CREATE TABLE peer_aggregate (
    agg_id INTEGER PRIMARY KEY,
    aggregate_id TEXT,
    level TEXT NOT NULL,
    level_name TEXT,
    level_order INTEGER,               -- 1 = самая узкая группа
    metric_id INTEGER REFERENCES metric(metric_id),  -- NULL если имя не сматчилось
    metric_name TEXT NOT NULL,
    metric_ext_id TEXT,
    node_uid INTEGER,
    parent_node_uid INTEGER,
    dt TEXT, ym TEXT, calc_period TEXT,
    is_current INTEGER NOT NULL,
    mean_fact REAL, mean_plan REAL, mean_ex REAL, median REAL, hit_rate REAL,
    top20_mean_fact REAL, iqr REAL, cv REAL, total_objects INTEGER
);
CREATE INDEX ix_agg_lookup ON peer_aggregate(metric_id, level, ym, is_current);

CREATE TABLE person_peer (
    person_id INTEGER NOT NULL REFERENCES person(person_id),
    aggregate_id TEXT NOT NULL,
    PRIMARY KEY (person_id, aggregate_id)
);

-- Кэш отклонений треда: материализуется из state каждый ход, чтобы SQL мог
-- по нему ходить (v_deviation).
CREATE TABLE deviation (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    polarity TEXT NOT NULL,            -- 'negative' | 'positive'
    metric_id INTEGER REFERENCES metric(metric_id),
    metric_name TEXT NOT NULL,
    metric_ext_id TEXT,
    person_id INTEGER REFERENCES person(person_id),
    person_key TEXT NOT NULL,
    element TEXT,
    date TEXT, depth INTEGER, root_name TEXT,
    signal REAL, priority REAL NOT NULL,
    fact REAL, plan REAL, plan_dev_pct REAL, pop_change_pct REAL, zscore REAL,
    percentile REAL, benchmark_dev_pct REAL, influent_share REAL,
    source TEXT NOT NULL,              -- 'auto' | 'agent'
    status TEXT NOT NULL DEFAULT 'open',
    task_id TEXT, task_target REAL, task_dev_pct REAL, note TEXT,
    turn_created INTEGER, turn_updated INTEGER
);

-- --- ВЬЮ: то, что видит агент ------------------------------------------

CREATE VIEW v_person AS
SELECT p.person_id, p.person_key, p.tabnum, p.fio, p.post, p.depart, p.is_me,
       (SELECT group_concat(pp.aggregate_id, ',') FROM person_peer pp
         WHERE pp.person_id = p.person_id) AS aggregate_ids
FROM person p;

CREATE VIEW v_metric AS
SELECT m.metric_id, m.name, m.ext_id, m.description, m.direction, m.unit,
       m.calc_period, m.kind, m.depth, m.path, m.root_name, m.parent_name,
       m.influent_percent, m.is_star, m.is_star_metric, m.star_of,
       m.kn_status, m.kn_summary, m.kn_aliases, m.kn_cumulative,
       m.kn_plan_semantics, m.kn_fact_semantics, m.kn_processes, m.kn_notes,
       (SELECT COUNT(*) FROM metric_edge e WHERE e.parent_id = m.metric_id) AS n_children,
       (SELECT COUNT(DISTINCT f.element) FROM fact f
         WHERE f.metric_id = m.metric_id AND f.element IS NOT NULL) AS n_elements,
       (SELECT COUNT(DISTINCT f.period_id) FROM fact f
         WHERE f.metric_id = m.metric_id) AS n_periods,
       (SELECT COUNT(*) FROM fact f
         WHERE f.metric_id = m.metric_id AND f.plan IS NOT NULL AND f.plan <> 0) > 0 AS has_plan,
       (m.depth = 1) AS is_root
FROM metric m;

CREATE VIEW v_tree AS
SELECT pm.name AS parent, cm.name AS child, e.influent_percent,
       cm.depth AS child_depth, cm.path AS child_path, e.edge_kind,
       CASE WHEN (SELECT SUM(e2.influent_percent) FROM metric_edge e2
                   WHERE e2.parent_id = e.parent_id) > 0
            THEN ROUND(e.influent_percent * 100.0 /
                 (SELECT SUM(e2.influent_percent) FROM metric_edge e2
                   WHERE e2.parent_id = e.parent_id), 2)
       END AS share_pct
FROM metric_edge e
JOIN metric pm ON pm.metric_id = e.parent_id
JOIN metric cm ON cm.metric_id = e.child_id;

CREATE VIEW v_fact AS
SELECT f.fact_id, p.person_key, p.fio, p.is_me, p.post, p.depart,
       m.name AS metric, m.ext_id, m.description, m.direction, m.unit, m.kind,
       m.depth, m.path, m.root_name, m.parent_name, m.influent_percent,
       m.is_star, m.is_star_metric, m.star_of, m.kn_summary,
       f.element, d.date, d.period_idx, d.ym, f.is_last_of_series,
       d.is_latest AS is_dataset_latest,
       f.fact, f.plan, f.benchmark, f.ex, f.rr, f.star_received, f.calc_period,
       a.plan_dev_abs, a.plan_dev_pct, a.plan_status,
       a.benchmark_dev_abs, a.benchmark_dev_pct, a.benchmark_status,
       a.pop_change_abs, a.pop_change_pct, a.pop_status,
       a.trend, a.trend_status,
       a.peer_mean, a.peer_std, a.peer_count, a.peer_rank, a.peer_percentile,
       a.zscore, a.peer_status, a.is_anomaly,
       a.group_change_pct, a.rel_change_pct, a.rel_status,
       a.group_hit_rate, a.plan_rigidity
FROM fact f
JOIN person p ON p.person_id = f.person_id
JOIN metric m ON m.metric_id = f.metric_id
JOIN period d ON d.period_id = f.period_id
LEFT JOIN fact_analytics a ON a.fact_id = f.fact_id;

-- Последнее значение КАЖДОЙ серии (не глобальная дата!).
CREATE VIEW v_fact_latest AS
SELECT * FROM v_fact WHERE is_last_of_series = 1;

CREATE VIEW v_element AS
SELECT m.name AS metric, f.element,
       COUNT(DISTINCT f.person_id) AS n_persons,
       COUNT(DISTINCT f.period_id) AS n_periods
FROM fact f JOIN metric m ON m.metric_id = f.metric_id
WHERE f.element IS NOT NULL
GROUP BY m.name, f.element;

CREATE VIEW v_peer_latest AS
SELECT p.person_key, m.name AS metric, g.level, g.level_name, g.level_order,
       g.dt, g.ym, g.mean_fact, g.mean_plan, g.mean_ex, g.median, g.hit_rate,
       g.top20_mean_fact, g.iqr, g.cv, g.total_objects,
       (SELECT r.rank_raw FROM ranking r
          JOIN fact f2 ON f2.fact_id = r.fact_id
         WHERE f2.person_id = p.person_id AND f2.metric_id = m.metric_id
           AND r.level = g.level AND f2.is_last_of_series = 1
         LIMIT 1) AS rank_raw,
       (SELECT r.percentile FROM ranking r
          JOIN fact f2 ON f2.fact_id = r.fact_id
         WHERE f2.person_id = p.person_id AND f2.metric_id = m.metric_id
           AND r.level = g.level AND f2.is_last_of_series = 1
         LIMIT 1) AS percentile
FROM peer_aggregate g
JOIN metric m ON m.metric_id = g.metric_id
JOIN person p ON EXISTS (
        SELECT 1 FROM fact f WHERE f.person_id = p.person_id AND f.metric_id = m.metric_id)
   AND (g.aggregate_id IS NULL
        OR NOT EXISTS (SELECT 1 FROM person_peer pp WHERE pp.person_id = p.person_id)
        OR EXISTS (SELECT 1 FROM person_peer pp
                    WHERE pp.person_id = p.person_id AND pp.aggregate_id = g.aggregate_id))
WHERE g.is_current = 1;

-- Звёзды: строка на (персона, звезда, влияющий показатель) за последний
-- период каждой серии.
CREATE VIEW v_star AS
SELECT p.person_key, p.fio, sm.name AS star, sf.star_received AS received,
       sf.date AS star_date, cm.name AS child, cm.is_star_metric AS child_is_influencing,
       cf.fact AS child_fact, cf.plan AS child_plan, cf.date AS child_date,
       cf.plan_status AS child_plan_status, cf.plan_dev_pct AS child_plan_dev_pct
FROM v_fact sf
JOIN metric sm ON sm.name = sf.metric AND sm.is_star = 1
JOIN person p ON p.person_key = sf.person_key
LEFT JOIN metric cm ON cm.star_of = sm.name
LEFT JOIN v_fact cf ON cf.metric = cm.name AND cf.person_key = sf.person_key
       AND cf.is_last_of_series = 1 AND cf.element IS NULL
WHERE sf.is_last_of_series = 1 AND sf.star_received IS NOT NULL;

CREATE VIEW v_deviation AS
SELECT d.*, m.unit, m.path, m.direction
FROM deviation d LEFT JOIN metric m ON m.metric_id = d.metric_id
ORDER BY d.priority DESC;
