#!/usr/bin/env python3
"""Отладка SQL агента-аналитика: те же шаблоны и та же защита, что на проме.

Смысл в воспроизводимости: инструменты агента гоняют файлы из
``analyst_agent/db/sql``, и этот скрипт гоняет ИХ ЖЕ — на дампе базы с прода
или на датасете из ``samples_v2``. Свободный SQL идёт через тот же
SafeQueryRunner, поэтому ошибка воспроизводится дословно.

Примеры::

    # список шаблонов
    python scripts/sql_debug.py --dataset samples_v2/sample_declining.json --list

    # шаблон с параметрами на дампе базы (DEBUG_DUMP_DIR на проме)
    python scripts/sql_debug.py --db dump.db --template tool_metric_history \\
        --param person_key=4032085 --param metric=AHT

    # свободный запрос и блоки обогащения промпта
    python scripts/sql_debug.py --dataset samples_v2/sample_star.json \\
        --sql "SELECT metric, date, fact FROM v_fact_latest LIMIT 5"
    python scripts/sql_debug.py --dataset samples_v2/sample_good.json --enrichment
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Клиент GigaChat создаётся на импорте графов; здесь сеть не нужна, но заглушка
# избавляет от падения, если модуль подтянется транзитивно.
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from langgraph_executor.aegra_agents.analyst_agent.db import (  # noqa: E402
    analytics,
    core,
    enrich,
    sqlrunner,
)
from langgraph_executor.aegra_agents.analyst_agent.db.text2sql import (  # noqa: E402
    QueryResult,
    SafeQueryRunner,
)


def _load_db(args: argparse.Namespace) -> core.RunDb:
    if args.db:
        conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.create_function("ru_lower", 1, core._ru_lower, deterministic=True)
        conn.create_function("pct", 2, core._pct, deterministic=True)
        return core.RunDb(conn, core.BuildReport())

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    aggregates = (
        json.loads(Path(args.aggregates).read_text(encoding="utf-8"))
        if args.aggregates
        else None
    )
    db = core.build_run_db(dataset, aggregates)
    return db


def _parse_params(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in pairs or []:
        if "=" not in raw:
            raise SystemExit(f"параметр без значения: {raw!r} (нужно имя=значение)")
        key, value = raw.split("=", 1)
        out[key.strip()] = value
    return out


def _emit(res: QueryResult, fmt: str) -> int:
    if res.error:
        print(res.error)
        return 1
    if fmt == "json":
        print(json.dumps(
            {"columns": res.columns, "rows": [list(r) for r in res.rows],
             "truncated": res.truncated},
            ensure_ascii=False, indent=2,
        ))
    elif fmt == "csv":
        print(sqlrunner.render_csv(res))
    elif fmt == "kv":
        print(sqlrunner.render_kv(res))
    else:
        print(sqlrunner.render_markdown(res))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--db", help="дамп базы запуска (.db)")
    src.add_argument("--dataset", help="JSON датасета, база собирается на лету")
    parser.add_argument("--aggregates", help="JSON предагрегатов peer-групп (с --dataset)")
    parser.add_argument("--template", help="имя шаблона из db/sql")
    parser.add_argument("--param", action="append", default=[], metavar="ИМЯ=ЗНАЧЕНИЕ")
    parser.add_argument("--sql", help="свободный SELECT (через ту же защиту, что у агента)")
    parser.add_argument("--list", action="store_true", help="показать шаблоны и выйти")
    parser.add_argument("--schema", action="store_true", help="справка по схеме для промпта")
    parser.add_argument("--enrichment", action="store_true", help="блоки обогащения промпта")
    parser.add_argument("--person", help="person_key для блоков обогащения")
    parser.add_argument("--format", choices=("md", "csv", "json", "kv"), default="md")
    parser.add_argument("--max-rows", type=int, default=200)
    args = parser.parse_args()

    if args.list:
        for tpl in sqlrunner.list_templates():
            params = ", ".join(
                f"{p.name}:{p.type}" + ("?" if p.optional else "")
                for p in tpl.params.values()
            )
            print(f"{tpl.name}({params})\n    {tpl.summary}")
        return 0

    db = _load_db(args)

    if args.schema:
        print(core.schema_doc(db))
        return 0

    if args.enrichment:
        person = args.person
        if not person:
            row = db.conn.execute(
                "SELECT person_key FROM person ORDER BY is_me, person_id LIMIT 1"
            ).fetchone()
            person = row["person_key"] if row else None
        if not person:
            print("в базе нет людей")
            return 1
        print(enrich.enrichment_block(db, person_key=person))
        print()
        print(enrich.catalog_block(db))
        return 0

    if args.template:
        try:
            res = sqlrunner.run_template(
                db.conn, args.template, max_rows=args.max_rows, **_parse_params(args.param)
            )
        except (ValueError, FileNotFoundError) as exc:
            print(str(exc))
            return 1
        return _emit(res, args.format)

    if args.sql:
        runner = SafeQueryRunner(db.conn, max_rows=args.max_rows)
        runner.install()
        return _emit(runner.run(args.sql), args.format)

    parser.error("укажи --template, --sql, --schema, --enrichment или --list")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
