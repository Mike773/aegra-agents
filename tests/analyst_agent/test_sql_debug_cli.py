"""scripts/sql_debug.py: тот же SQL, что на проме, воспроизводится локально.

Ради этого шаблоны и лежат отдельными .sql-файлами: результат CLI обязан
совпадать с тем, что получает агент в процессе.
"""
from __future__ import annotations

import json
import subprocess
import sys

from _fixtures import REPO_ROOT, SAMPLES_DIR, load_sample  # noqa: F401

from langgraph_executor.aegra_agents.analyst_agent.db import analytics, core, sqlrunner

SCRIPT = REPO_ROOT / "scripts" / "sql_debug.py"
SAMPLE = SAMPLES_DIR / "sample_declining.json"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env={"PATH": "/usr/bin:/bin", "GIGACHAT_CREDENTIALS": "test-dummy",
             "PYTHONPATH": str(REPO_ROOT)},
    )


def test_list_templates():
    proc = _run("--dataset", str(SAMPLE), "--list")
    assert proc.returncode == 0, proc.stderr
    assert "enrich_profile" in proc.stdout
    assert "tool_metric_history" in proc.stdout


def test_template_output_matches_in_process():
    db = core.build_run_db(load_sample(SAMPLE.name))
    person = db.conn.execute("SELECT person_key FROM person LIMIT 1").fetchone()[0]
    metric = db.conn.execute("SELECT name FROM metric WHERE depth = 1 LIMIT 1").fetchone()[0]
    expected = sqlrunner.render_markdown(
        sqlrunner.run_template(
            db.conn, "tool_metric_history", person_key=person, metric=metric
        )
    )

    proc = _run(
        "--dataset", str(SAMPLE), "--template", "tool_metric_history",
        "--param", f"person_key={person}", "--param", f"metric={metric}",
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == expected.strip()


def test_free_sql_goes_through_the_same_guard():
    proc = _run("--dataset", str(SAMPLE), "--sql", "DELETE FROM fact")
    assert proc.returncode != 0
    assert "ОШИБКА SQL" in proc.stdout + proc.stderr


def test_json_format():
    proc = _run(
        "--dataset", str(SAMPLE), "--sql", "SELECT COUNT(*) AS n FROM v_fact",
        "--format", "json",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["rows"][0][0] > 0
    assert payload["columns"] == ["n"]


def test_enrichment_dump():
    proc = _run("--dataset", str(SAMPLE), "--enrichment")
    assert proc.returncode == 0, proc.stderr
    assert "СОСТАВ ДАННЫХ" in proc.stdout
