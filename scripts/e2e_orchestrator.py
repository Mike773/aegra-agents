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
    config = {
        "configurable": {
            "boss_tabnum": "1000",
            "employee_tabnum": "2000",
            "position": "оператор",
            "thread_id": "e2e-orch-1",
            "wiki_grounding_enabled": False,
        },
        "callbacks": [tracker],
    }
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
