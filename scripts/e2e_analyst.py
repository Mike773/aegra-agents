"""E2E прогон analyst_agent на живом GigaChat.

Сервис данных здесь заглушка, поэтому датасет берём из ``samples_v2``, а
предагрегаты peer-групп синтезируем по демо-профилям групп. Печатаем финальный
ответ, шаги хода, трассу, карту отклонений и расход токенов — по этому выводу и
видно, как агент работает.

Примеры::

    .venv/bin/python scripts/e2e_analyst.py
    E2E_SAMPLE=samples_v2/sample_star.json .venv/bin/python scripts/e2e_analyst.py
    E2E_TURNS=2 .venv/bin/python scripts/e2e_analyst.py          # ход 1 + уточнение
    E2E_DASHBOARD=1 .venv/bin/python scripts/e2e_analyst.py      # составной разбор
    E2E_PEER=1 .venv/bin/python scripts/e2e_analyst.py           # с группами сравнения
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()
warnings.filterwarnings("ignore")

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from langgraph_executor.aegra_agents.shared import agent_dataset  # noqa: E402

SAMPLE = os.environ.get("E2E_SAMPLE", "samples_v2/sample_declining.json")
_DATA = json.loads(Path(SAMPLE).read_text(encoding="utf-8"))

# Демо-группы сравнения: (код, человеческое название, множитель среднего,
# доля выполняющих план, размер). Название приходит с данными — его и печатает
# агент; коды наружу не попадают никогда.
_LEVEL_PROFILES = (
    ("OFFICE", "по офису", 1.00, 46.0, 40),
    ("TERR", "по территории", 1.12, 52.0, 900),
    ("ORG", "по всему банку", 1.25, 58.0, 12000),
)


def _agg_id(metric: dict) -> str:
    return f"agg_{metric.get('id') or metric['metric_name']}"


def _inject_aggregates_ids(data: dict) -> None:
    """Сэмплы — старой формы, без aggregates_ids; кладём их на лету."""
    for person in [data.get("me"), *(data.get("employees") or [])]:
        if not isinstance(person, dict):
            continue
        ids = []
        for metric in person.get("metrics") or []:
            agg = _agg_id(metric)
            if agg not in ids:
                ids.append(agg)
        person["aggregates_ids"] = ids


def _facts_of(metric: dict) -> list[float]:
    out = []
    if isinstance(metric.get("fact"), (int, float)):
        out.append(float(metric["fact"]))
    for child in metric.get("child_metrics") or []:
        out.extend(_facts_of(child))
    return out


def _aggregate_payload(agg_id: str) -> list[dict]:
    """Синтетические предагрегаты по демо-профилям для одного agg_id."""
    metric = None
    for person in [_DATA.get("me"), *(_DATA.get("employees") or [])]:
        for m in (person or {}).get("metrics") or []:
            if _agg_id(m) == agg_id:
                metric = m
                break
    if metric is None:
        return []
    facts = _facts_of(metric)
    base = sum(facts) / len(facts) if facts else 100.0
    date = metric.get("date") or "2026-04-13"
    out = []
    for level, level_name, factor, hit_rate, size in _LEVEL_PROFILES:
        out.append(
            {
                "dataset": {
                    "level": level,
                    "level_name": level_name,
                    "metrics": [
                        {
                            "metric_id": metric.get("id"),
                            "metric_name": metric["metric_name"],
                            "aggregates": {
                                "dt": date,
                                "calc_period": metric.get("calc_period") or "неделя",
                                "mean_fact": round(base * factor, 2),
                                "median": round(base * factor * 0.98, 2),
                                "top20_mean_fact": round(base * factor * 1.25, 2),
                                "hit_rate": hit_rate,
                                "total_objects": size,
                            },
                        }
                    ],
                }
            }
        )
    return out


agent_dataset.GetBatchAgentDatasetByFiltersComponent.build_json_output = lambda self: _DATA
if os.environ.get("E2E_PEER"):
    _inject_aggregates_ids(_DATA)
    agent_dataset.GetBatchAgentAggregateDatasetByFiltersComponent.build_json_output = (
        lambda self: _aggregate_payload(
            (self.filters or [{}])[0].get("object_id", "") if isinstance(self.filters, list) else ""
        )
    )


class UsageTracker(BaseCallbackHandler):
    """Считает вызовы модели и токены — цена хода видна сразу."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def on_llm_end(self, response, **kwargs):  # noqa: D102
        usage = {}
        for gen in response.generations:
            for g in gen:
                meta = getattr(g.message, "response_metadata", {}) or {}
                usage = dict(meta.get("token_usage") or usage)
                um = getattr(g.message, "usage_metadata", None) or {}
                details = um.get("input_token_details") or {}
                usage["cache_read"] = int(details.get("cache_read") or 0)
        self.calls.append(usage or {})

    def report(self) -> str:
        total_in = sum(int(c.get("prompt_tokens") or 0) for c in self.calls)
        total_out = sum(int(c.get("completion_tokens") or 0) for c in self.calls)
        peak = max((int(c.get("prompt_tokens") or 0) for c in self.calls), default=0)
        cached = sum(int(c.get("cache_read") or 0) for c in self.calls)
        return (
            f"вызовов модели: {len(self.calls)}; токенов на вход: {total_in} "
            f"(из кэша префикса: {cached}), на выход: {total_out}; пик входа: {peak}"
        )


DEFAULT_BRIEFING = (
    "Разбери ключевые показатели сотрудника: где он отстаёт от плана и от коллег, "
    "какая динамика, из чего складывается результат."
)
DASHBOARD_BRIEFING = (
    "Сделай сводный разбор:\n"
    "1. Что с выполнением плана по ключевым показателям\n"
    "2. Как сотрудник смотрится на фоне коллег\n"
    "3. Что стало главной причиной просадки\n"
)


def _print_messages(result: dict) -> None:
    for m in result.get("messages") or []:
        kwargs = getattr(m, "additional_kwargs", {}) or {}
        if kwargs.get("orchestrator_step"):
            print(f"  · {m.content}")
        elif kwargs.get("orchestrator_final"):
            print("\n" + "=" * 70)
            print("ИТОГОВЫЙ ОТВЕТ:\n")
            print(kwargs.get("orchestrator_markdown") or m.content)


async def main() -> int:
    from langgraph.checkpoint.memory import MemorySaver

    from langgraph_executor.aegra_agents.analyst_agent.graph import build_graph, llm

    dashboard = bool(os.environ.get("E2E_DASHBOARD"))
    briefing = os.environ.get(
        "E2E_BRIEFING", DASHBOARD_BRIEFING if dashboard else DEFAULT_BRIEFING
    )
    print(f"датасет: {SAMPLE}")
    print(f"режим: {'составной разбор' if dashboard else 'обычный ход'}")
    print(f"брифинг: {briefing.strip()}\n")

    tracker = UsageTracker()
    app = build_graph(llm, checkpointer=MemorySaver())
    config = {
        "configurable": {
            "boss_tabnum": "1000",
            "employee_tabnum": "2000",
            "position": os.environ.get("E2E_POSITION", "оператор"),
            "thread_id": "e2e-analyst-1",
            "dashboard_mode": dashboard,
            # Базу знаний и её кэш трогаем только по явной просьбе: нужен Postgres.
            "easyrag_enabled": bool(os.environ.get("E2E_WIKI")),
            "knowledge_enabled": bool(os.environ.get("E2E_WIKI")),
            "describe_answer": bool(os.environ.get("E2E_DESCRIBE")),
            "answer_html": False,
        },
        "callbacks": [tracker],
    }

    result = await app.ainvoke({"messages": [HumanMessage(content=briefing)]}, config)
    _print_messages(result)

    if result.get("task_results"):
        print("\n" + "=" * 70 + "\nЗАДАЧИ СОСТАВНОГО РАЗБОРА:")
        for r in result["task_results"]:
            print(f"\n— {r['title']}\n{r['answer_md']}")

    print("\n" + "=" * 70 + "\nШАГИ ХОДА (трасса):")
    for i, step in enumerate(result.get("reasoning_trace") or [], 1):
        print(f"  {i}. [{step.get('kind')}] {step.get('summary')}")

    state = app.get_state(config).values
    devs = state.get("deviations") or []
    print("\n" + "=" * 70 + f"\nКАРТА ОТКЛОНЕНИЙ (top-8 из {len(devs)}):")
    for d in devs[:8]:
        print(
            f"  {d['priority']:.2f}  {d['metric_name']}"
            + (f" ({d['element']})" if d.get("element") else "")
            + f" — {d['kind']}"
        )

    turns = int(os.environ.get("E2E_TURNS", "1"))
    if turns > 1:
        follow = os.environ.get(
            "E2E_FOLLOWUP", "Разбери подробнее главную область внимания."
        )
        print("\n" + "=" * 70 + f"\nХОД 2: {follow}\n")
        result2 = await app.ainvoke({"messages": [HumanMessage(content=follow)]}, config)
        _print_messages(result2)

    print("\n" + "=" * 70)
    print(tracker.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
