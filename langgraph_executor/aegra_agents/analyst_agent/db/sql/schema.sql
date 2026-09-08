-- Нормализованная схема in-memory SQLite для analyst_agent.
--
-- Базовые таблицы служебные: агент ходит во ВЬЮ (v_fact и остальные) —
-- там уже склеены персона, метрика, период и готовые вердикты.
--
-- Даты: «последний период» считается НА СЕРИЮ (персона × метрика × разрез)
-- через fact.is_last_of_series, а НЕ по глобальной дате датасета. Дочерние
-- показатели могут отставать по датам от родительских — это нормально, они
-- всё равно попадают в v_fact_latest со своей датой. Дерево метрик
-- (metric_edge, metric_tree) строится только из структуры JSON и от дат не
-- зависит вовсе.
--
-- Дерево у КАЖДОГО человека своё: у руководителя и сотрудников набор и
-- вложенность показателей могут отличаться. Рёбра (metric_edge) и позиция
-- показателя (metric_tree: depth, path, parent_name) привязаны к person_id, и
-- именно они попадают в v_fact и v_tree. На каталоге (metric.depth/path/
-- parent_name) лежит СВОДНОЕ дерево по всем людям — ориентир для каталога и
-- резолва имён, не истина для конкретного человека.

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
    -- Сводное дерево по всем людям (см. шапку); точное для человека — metric_tree
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

-- Рёбра дерева ЭТОГО человека: родитель → ребёнок с весом влияния.
CREATE TABLE metric_edge (
    person_id INTEGER NOT NULL REFERENCES person(person_id),
    parent_id INTEGER NOT NULL REFERENCES metric(metric_id),
    child_id  INTEGER NOT NULL REFERENCES metric(metric_id),
    influent_percent REAL,
    edge_kind TEXT NOT NULL DEFAULT 'tree',
    PRIMARY KEY (person_id, parent_id, child_id)
);
CREATE INDEX ix_edge_child ON metric_edge(person_id, child_id);

-- Место показателя в дереве ЭТОГО человека (по доминирующему родителю).
CREATE TABLE metric_tree (
    person_id INTEGER NOT NULL REFERENCES person(person_id),
    metric_id INTEGER NOT NULL REFERENCES metric(metric_id),
    depth       INTEGER NOT NULL,
    root_name   TEXT NOT NULL,
    path        TEXT NOT NULL,             -- 'A > B > C'
    parent_name TEXT,
    influent_percent REAL,
    PRIMARY KEY (person_id, metric_id)
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
       (SELECT COUNT(DISTINCT e.child_id) FROM metric_edge e
         WHERE e.parent_id = m.metric_id) AS n_children,
       (SELECT COUNT(DISTINCT f.element) FROM fact f
         WHERE f.metric_id = m.metric_id AND f.element IS NOT NULL) AS n_elements,
       (SELECT COUNT(DISTINCT f.period_id) FROM fact f
         WHERE f.metric_id = m.metric_id) AS n_periods,
       (SELECT COUNT(*) FROM fact f
         WHERE f.metric_id = m.metric_id AND f.plan IS NOT NULL AND f.plan <> 0) > 0 AS has_plan,
       (m.depth = 1) AS is_root
FROM metric m;

-- Показатель глазами конкретного человека.
--
-- has_aggregate — есть ли у ЭТОГО человека строка без разреза, то есть
-- собственный итог показателя. Считать глобально по имени нельзя: у
-- руководителя итог может быть, а у сотрудника тот же показатель представлен
-- только разрезами, и глобальный флаг увёл бы запрос в element IS NULL и
-- вернул бы пусто, скрыв данные сотрудника.
CREATE VIEW v_metric_person AS
SELECT p.person_key,
       m.name AS metric,
       m.metric_id,
       m.depth,
       m.direction,
       m.unit,
       m.kind,
       EXISTS (SELECT 1 FROM fact f
                WHERE f.person_id = p.person_id AND f.metric_id = m.metric_id
                  AND f.element IS NULL) AS has_aggregate,
       (SELECT COUNT(DISTINCT f.element) FROM fact f
         WHERE f.person_id = p.person_id AND f.metric_id = m.metric_id
           AND f.element IS NOT NULL) AS n_elements,
       (SELECT COUNT(DISTINCT f.period_id) FROM fact f
         WHERE f.person_id = p.person_id AND f.metric_id = m.metric_id) AS n_periods,
       (SELECT COUNT(*) FROM fact f
         WHERE f.person_id = p.person_id AND f.metric_id = m.metric_id
           AND f.plan IS NOT NULL AND f.plan <> 0) > 0 AS has_plan
FROM person p
JOIN metric m ON EXISTS (
        SELECT 1 FROM fact f
         WHERE f.person_id = p.person_id AND f.metric_id = m.metric_id);

-- Дерево показателей человека: строка на (person_key, родитель, ребёнок).
CREATE VIEW v_tree AS
SELECT p.person_key, pm.name AS parent, cm.name AS child, e.influent_percent,
       t.depth AS child_depth, t.path AS child_path, e.edge_kind,
       CASE WHEN (SELECT SUM(e2.influent_percent) FROM metric_edge e2
                   WHERE e2.person_id = e.person_id AND e2.parent_id = e.parent_id) > 0
            THEN ROUND(e.influent_percent * 100.0 /
                 (SELECT SUM(e2.influent_percent) FROM metric_edge e2
                   WHERE e2.person_id = e.person_id AND e2.parent_id = e.parent_id), 2)
       END AS share_pct
FROM metric_edge e
JOIN person p ON p.person_id = e.person_id
JOIN metric pm ON pm.metric_id = e.parent_id
JOIN metric cm ON cm.metric_id = e.child_id
LEFT JOIN metric_tree t ON t.person_id = e.person_id AND t.metric_id = e.child_id;

-- depth/path/parent_name — из дерева ЭТОГО человека (metric_tree); сводные
-- значения каталога подставляются только если у человека своей записи нет.
CREATE VIEW v_fact AS
SELECT f.fact_id, p.person_key, p.fio, p.is_me, p.post, p.depart,
       m.name AS metric, m.ext_id, m.description, m.direction, m.unit, m.kind,
       COALESCE(t.depth, m.depth) AS depth,
       COALESCE(t.path, m.path) AS path,
       COALESCE(t.root_name, m.root_name) AS root_name,
       CASE WHEN t.metric_id IS NULL THEN m.parent_name ELSE t.parent_name END AS parent_name,
       CASE WHEN t.metric_id IS NULL THEN m.influent_percent ELSE t.influent_percent END
           AS influent_percent,
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
LEFT JOIN metric_tree t ON t.person_id = f.person_id AND t.metric_id = f.metric_id
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

-- Влияющие показатели звёзд: строка на (персона, звезда, показатель). Берутся
-- дети звезды в дереве ЭТОГО человека с признаком is_star_metric, последнее
-- значение серии без разреза. completion_pct — выполнение плана, % (ex из
-- данных либо факт / план).
CREATE VIEW v_star_metric AS
SELECT sf.person_id, p.person_key, sf.metric_id AS star_id, sm.name AS star,
       cf.metric_id, cm.name AS metric, cf.fact, cf.plan, cf.ex,
       ROUND(COALESCE(cf.ex, CASE WHEN cf.plan IS NOT NULL AND cf.plan <> 0
                                  THEN cf.fact * 100.0 / cf.plan END), 1) AS completion_pct,
       a.plan_status, a.plan_dev_pct, d.date
FROM fact sf
JOIN person p ON p.person_id = sf.person_id
JOIN metric sm ON sm.metric_id = sf.metric_id AND sm.is_star = 1
JOIN metric_edge e ON e.person_id = sf.person_id AND e.parent_id = sf.metric_id
JOIN metric cm ON cm.metric_id = e.child_id AND cm.is_star_metric = 1
JOIN fact cf ON cf.person_id = sf.person_id AND cf.metric_id = cm.metric_id
       AND cf.is_last_of_series = 1 AND cf.element IS NULL
JOIN period d ON d.period_id = cf.period_id
LEFT JOIN fact_analytics a ON a.fact_id = cf.fact_id
WHERE sf.star_received IS NOT NULL AND sf.is_last_of_series = 1;

-- Звёзды: ОДНА строка на (персона, звезда) за последний период серии.
-- metrics — влияющие показатели одной строкой: «имя: факт X при плане Y,
-- выполнение Z %, вердикт; …». Обычные показатели сюда не попадают.
CREATE VIEW v_star AS
SELECT p.person_key, p.fio, sm.name AS star, sf.star_received AS received,
       d.date AS star_date,
       (SELECT COUNT(*) FROM v_star_metric x
         WHERE x.person_id = sf.person_id AND x.star_id = sf.metric_id) AS n_metrics,
       (SELECT COUNT(*) FROM v_star_metric x
         WHERE x.person_id = sf.person_id AND x.star_id = sf.metric_id
           AND x.plan_status = 'хуже_плана') AS n_below_plan,
       (SELECT group_concat(line, '; ') FROM (
            SELECT x.metric || ': факт ' || COALESCE(fmt_num(x.fact), '—')
                   || CASE WHEN x.plan IS NULL THEN ''
                           ELSE ' при плане ' || fmt_num(x.plan) END
                   || CASE WHEN x.completion_pct IS NULL THEN ''
                           ELSE ', выполнение ' || fmt_num(x.completion_pct) || ' %' END
                   || CASE WHEN x.plan_status IS NULL THEN ''
                           ELSE ', ' || REPLACE(x.plan_status, '_', ' ') END AS line
              FROM v_star_metric x
             WHERE x.person_id = sf.person_id AND x.star_id = sf.metric_id
             ORDER BY x.plan_status = 'хуже_плана' DESC, x.metric)) AS metrics
FROM fact sf
JOIN metric sm ON sm.metric_id = sf.metric_id AND sm.is_star = 1
JOIN person p ON p.person_id = sf.person_id
JOIN period d ON d.period_id = sf.period_id
WHERE sf.is_last_of_series = 1 AND sf.star_received IS NOT NULL;

CREATE VIEW v_deviation AS
SELECT d.*, m.unit, m.path, m.direction
FROM deviation d LEFT JOIN metric m ON m.metric_id = d.metric_id
ORDER BY d.priority DESC;
