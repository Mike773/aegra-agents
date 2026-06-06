"""Smoke-проверка каузального слоя json_analyzer_causal (этапы 1-3).

Гоняет детерминированный слой БЕЗ LLM/креденшелов:
  - algebraic: sample_declining.json (1 человек, 6 дат)
  - causal:    test_metrics.json (25 сотрудников) — нужен dowhy
  - anomaly:   test_metrics.json — нужен dowhy
"""
import json
import sys

from langgraph_executor.aegra_agents.json_analyzer_causal import causal
from langgraph_executor.aegra_agents.json_analyzer_causal.analytics import (
    compute_analytics,
)
from langgraph_executor.aegra_agents.json_analyzer_causal.loader import (
    load_dataset_obj,
)
from langgraph_executor.aegra_agents.json_analyzer_causal.sqlite_store import (
    SqliteStore,
)


def build(path):
    data = json.load(open(path, encoding="utf-8"))
    rows = load_dataset_obj(data)
    store = SqliteStore()
    store.load(rows)
    compute_analytics(store)
    return store


def show(title, result):
    print("\n" + "=" * 70)
    print(title)
    print("-" * 70)
    print("method:", result.get("method"), "| target:", result.get("target"),
          "| type:", result.get("target_type"))
    for k in ("date_old", "date_new", "date", "cohort_size",
              "reference_cohort_size", "parent_old", "parent_new",
              "parent_delta", "residual_unexplained", "self_attribution",
              "causal_fallback_reason", "error"):
        if k in result and result[k] is not None:
            print(f"  {k}: {result[k]}")
    for c in result.get("contributions", [])[:8]:
        print("   •", json.dumps(c, ensure_ascii=False))


def main():
    print("causal_available():", causal.causal_available())

    # --- Этап 1: algebraic, один человек ---
    store1 = build("samples/sample_declining.json")
    dates = causal._dates(store1)
    print("\nsample_declining dates:", dates)
    res = causal.attribute_change(store1, "Производительность")
    show("ЭТАП 1 — algebraic (sample_declining, 1 человек)", res)

    # --- Этап 2: causal по когорте ---
    store2 = build("samples/test_metrics.json")
    print("\ntest_metrics dates:", causal._dates(store2))
    print("employees:", len(causal._employee_tabnums(store2)))
    res2 = causal.attribute_change(store2, "Производительность")
    show("ЭТАП 2 — causal/algebraic (test_metrics, 25 сотрудников)", res2)

    # algebraic для конкретного человека на тех же данных (для сравнения)
    emp0 = causal._employee_tabnums(store2)[0]
    res2b = causal.attribute_change(store2, "Производительность", person=str(emp0))
    show(f"ЭТАП 1 — algebraic для сотрудника {emp0} (test_metrics)", res2b)

    # --- Этап 3: anomaly ---
    # ищем сотрудника с аномалией по Производительности на последней дате
    last = causal._dates(store2)[-1]
    row = store2.conn.execute(
        "SELECT m.person_tabnum, m.person_fio, a.zscore FROM metrics m "
        "JOIN metric_analytics a ON a.metric_uid = m.metric_uid "
        "WHERE m.metric_name='Производительность' AND m.element IS NULL "
        "AND m.date=? AND a.is_anomaly=1 ORDER BY ABS(a.zscore) DESC LIMIT 1",
        (last,),
    ).fetchone()
    if row is None:
        # берём максимальный |zscore| даже если флага нет
        row = store2.conn.execute(
            "SELECT m.person_tabnum, m.person_fio, a.zscore FROM metrics m "
            "JOIN metric_analytics a ON a.metric_uid = m.metric_uid "
            "WHERE m.metric_name='Производительность' AND m.element IS NULL "
            "AND m.date=? AND a.zscore IS NOT NULL ORDER BY ABS(a.zscore) DESC LIMIT 1",
            (last,),
        ).fetchone()
    if row:
        print(f"\n[anomaly target] {row['person_fio']} "
              f"(tab={row['person_tabnum']}, z={row['zscore']}) на {last}")
        res3 = causal.attribute_anomaly(
            store2, "Производительность", row["person_tabnum"], date=last
        )
        show("ЭТАП 3 — attribute_anomalies (test_metrics)", res3)
    else:
        print("\nНе нашёл подходящего сотрудника для anomaly-теста")

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
