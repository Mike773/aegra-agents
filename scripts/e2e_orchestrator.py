"""E2E прогон json_analyzer ЧЕРЕЗ analytic_orchestrator (продакшн-путь).

Первый ход оркестратора: load_data -> ground_wiki_initial -> initial_analysis,
где initial_analysis вызывает подграф json_analyzer. Источник метрик
(GetBatchAgentDatasetByFiltersComponent) здесь заглушка, поэтому подменяем его
build_json_output на выбранный файл samples_v2. Wiki-grounding отключаем
(wiki_grounding_enabled=False), чтобы прогон был сфокусирован на json_analyzer.
"""
import asyncio
import json
import os
import sys
import warnings

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

# --- подмена источника метрик на файл samples_v2 ---
from langgraph_executor.aegra_agents.shared import agent_dataset  # noqa: E402

SAMPLE = os.environ.get("E2E_SAMPLE", "samples_v2/sample_declining.json")
_DATA = json.load(open(SAMPLE, encoding="utf-8"))
agent_dataset.GetBatchAgentDatasetByFiltersComponent.build_json_output = (
    lambda self: _DATA
)


# Профили демо-групп: (код уровня, множитель среднего, доля выполняющих план,
# размер группы). Уровни РАЗЛИЧАЮТСЯ по всем трём — иначе сравнение вырождается:
# одинаковые числа на всех уровнях модели нечем прокомментировать, и в ответе
# остаётся только референсная (самая узкая) группа.
_LEVEL_PROFILES = (
    ("OFFICE", 1.00, 46.0, 40),
    ("TERR", 1.12, 52.0, 900),
    ("ORG", 1.25, 58.0, 12000),
)


def _peer_aggregates_for_sample(data):
    """Демо-предагрегаты peer-групп ПОД ИМЕНА МЕТРИК сэмпла.

    Штатная заглушка отдаёт метрику «Метрика тест», которой в сэмпле нет, — тогда
    peer_context ничего не находит и сравнение с коллегами в ответе не проверишь.
    Здесь берём корневые метрики сэмпла и строим по ним группы с фактом чуть выше
    личного, историю за два прошлых периода и топ-20% заметно выше.
    """
    people = [p for p in [data.get("me"), *(data.get("employees") or [])] if p]
    seen, roots = set(), []
    for person in people:
        for m in person.get("metrics") or []:
            name = m.get("metric_name")
            fact = m.get("fact")
            if not name or name in seen or not isinstance(fact, (int, float)):
                continue
            seen.add(name)
            roots.append(m)

    def _metrics_for(level_mult, hit_rate, objects):
        out = []
        for m in roots:
            fact = m["fact"]

            def _slice(dt, mult, m=m, fact=fact):
                scale = level_mult * mult
                return {
                    "dt": dt, "calc_period": m.get("calc_period") or "Месяц",
                    "mean_fact": round(fact * 1.1 * scale, 2),
                    "mean_plan": round((m.get("plan") or fact) * 1.0, 2),
                    "mean_ex": 98.0, "median": round(fact * 1.05 * scale, 2),
                    "hit_rate": hit_rate,
                    "top20_mean_fact": round(fact * 1.4 * scale, 2),
                    "iqr": round(abs(fact) * 0.3, 2), "cv": 42.0,
                    "total_objects": objects,
                }

            out.append({
                "metric_id": str(m.get("id") or m["metric_name"]),
                "metric_name": m["metric_name"],
                "aggregates": {
                    **_slice(m.get("date") or "2026-07-30", 1.0),
                    "history": [
                        _slice("2026-05-31", 1.08), _slice("2026-04-30", 1.12),
                    ],
                },
            })
        return out

    return [
        {"dataset": {"level": lvl, "metrics": _metrics_for(mult, hit, objs)}}
        for lvl, mult, hit, objs in _LEVEL_PROFILES
    ]


_AGGS = _peer_aggregates_for_sample(_DATA)
agent_dataset.GetBatchAgentAggregateDatasetByFiltersComponent.build_json_output = (
    lambda self: _AGGS
)


class UsageTracker(BaseCallbackHandler):
    def __init__(self):
        self.calls = []

    def on_llm_end(self, response, **kwargs):
        usage = None
        try:
            msg = response.generations[0][0].message
            usage = (getattr(msg, "response_metadata", {}) or {}).get(
                "token_usage"
            ) or getattr(msg, "usage_metadata", None)
        except Exception:
            pass
        if not usage:
            return
        self.calls.append(
            {
                "input": usage.get("prompt_tokens", usage.get("input_tokens")),
                "output": usage.get("completion_tokens", usage.get("output_tokens")),
            }
        )

    def report(self):
        print("=" * 70 + "\nТОКЕНЫ GigaChat по вызовам:")
        mx = 0
        for i, c in enumerate(self.calls, 1):
            inp = c["input"] or 0
            mx = max(mx, inp)
            print(f"  #{i:<2} вход={inp:<7} выход={c['output'] or 0}")
        print(f"  вызовов: {len(self.calls)} | ПИК входа: {mx}")


async def main():
    # E2E_ORCHESTRATOR=v2 гоняет бизнес-оркестратор (analytic_orchestrator_v2):
    # многоуровневый разбор по бизнес-методологии + блок «Что делаем дальше?» +
    # завершение через post_insights. По умолчанию — прод-оркестратор.
    which = os.environ.get("E2E_ORCHESTRATOR", "").strip().lower()
    if which in {"v4", "4"}:
        # v4: аналитик json_analyzer_v5, предагрегаты peer-групп, инсайты по
        # source_type/source_id, блок продолжения без нумерованных вариантов.
        from langgraph_executor.aegra_agents.analytic_orchestrator_v4.graph import graph
    elif which in {"v2", "2", "business"}:
        from langgraph_executor.aegra_agents.analytic_orchestrator_v2.graph import graph
    else:
        from langgraph_executor.aegra_agents.analytic_orchestrator.graph import graph

    briefing = os.environ.get(
        "E2E_BRIEFING",
        "Ты — аналитик метрик. Разбери ключевые метрики сотрудника: где он хуже "
        "плана и коллег, какая динамика, из чего складывается результат. Дай 3–5 "
        "выводов с конкретными числами.",
    )
    print(f"sample: {SAMPLE}\nbriefing: {briefing}\n")

    tracker = UsageTracker()
    state = {"messages": [HumanMessage(content=briefing)]}
    configurable = {
        "boss_tabnum": "1000",
        "employee_tabnum": "2000",
        "position": "оператор",
        "thread_id": "e2e-orch-1",
        "wiki_grounding_enabled": False,
    }
    # Только для v4: привязка инсайтов, флаг предагрегатов и режим описания хода.
    if os.environ.get("E2E_SOURCE_ID"):
        configurable["source_type"] = os.environ.get("E2E_SOURCE_TYPE", "meeting")
        configurable["source_id"] = os.environ["E2E_SOURCE_ID"]
    if os.environ.get("E2E_PEER_AGGREGATES"):
        configurable["use_peer_aggregates"] = os.environ["E2E_PEER_AGGREGATES"]
    if os.environ.get("E2E_DESCRIBE_ANSWER"):
        configurable["describe_answer"] = os.environ["E2E_DESCRIBE_ANSWER"]
    config = {"configurable": configurable, "callbacks": [tracker]}
    result = await graph.ainvoke(state, config)

    print("=" * 70 + "\nREASONING TRACE (шаги хода):")
    for s in result.get("reasoning_trace", []):
        line = f"  [{s.get('stage')}/{s.get('kind')}] {s.get('summary', '')}"
        det = s.get("detail")
        if det:
            line += f"  | detail={json.dumps(det, ensure_ascii=False)[:300]}"
        print(line)

    print("=" * 70 + "\nmetrics загружены:", result.get("metrics") is not None,
          "| direction_key:", result.get("direction_key"),
          "| metrics_error:", result.get("metrics_error"))

    # Факты аналитика, которые ушли в системный контекст ответчика: если итоговый
    # ответ расходится с ними — проблема в промпте оркестратора, а не в аналитике.
    summary = result.get("metrics_summary")
    print("=" * 70 + "\nРАЗБОР АНАЛИТИКА (вход ответчика):")
    print(summary if summary else "<пусто>")

    tracker.report()

    print("=" * 70 + "\nИТОГОВЫЙ ОТВЕТ оркестратора:")
    msgs = result.get("messages", [])
    final = next(
        (m for m in reversed(msgs)
         if isinstance(m, AIMessage)
         and (m.additional_kwargs or {}).get("orchestrator_final")),
        msgs[-1] if msgs else None,
    )
    print(getattr(final, "content", "<нет ответа>"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
