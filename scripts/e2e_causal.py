"""E2E прогон графа json_analyzer_causal через GigaChat.

Полный путь: gather (load -> SQLite -> analytics -> embeddings -> relations ->
ReAct tool-loop через GigaChat) -> synthesize. Проверяем, что граф доходит до
ответа, что агент вызывает attribute_change, и СКОЛЬКО ТОКЕНОВ ушло в GigaChat
по каждому LLM-вызову (callback-трекер по response_metadata).
"""
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# тише сторонние варнинги dowhy/sklearn
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402


class UsageTracker(BaseCallbackHandler):
    """Снимает фактический расход токенов из ответов GigaChat по каждому вызову."""

    def __init__(self):
        self.calls = []  # list[dict(input,output,total)]

    def on_llm_end(self, response, **kwargs):
        usage = None
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            if msg is not None:
                usage = (getattr(msg, "response_metadata", {}) or {}).get(
                    "token_usage"
                ) or getattr(msg, "usage_metadata", None)
        except Exception:
            pass
        if usage is None:
            usage = (getattr(response, "llm_output", None) or {}).get("token_usage")
        if not usage:
            return  # эмбеддинги и т.п. без usage
        inp = usage.get("prompt_tokens", usage.get("input_tokens"))
        out = usage.get("completion_tokens", usage.get("output_tokens"))
        tot = usage.get("total_tokens")
        self.calls.append({"input": inp, "output": out, "total": tot})

    def report(self):
        print("=" * 70)
        print("ТОКЕНЫ GigaChat по LLM-вызовам (фактические):")
        print(f"  {'#':>2}  {'вход':>8}  {'выход':>7}  {'итого':>8}")
        max_in = 0
        sum_in = 0
        for i, c in enumerate(self.calls, 1):
            inp = c["input"] or 0
            max_in = max(max_in, inp)
            sum_in += inp
            print(f"  {i:>2}  {inp:>8}  {(c['output'] or 0):>7}  {(c['total'] or 0):>8}")
        print(f"  LLM-вызовов: {len(self.calls)}  | ПИК входа: {max_in}  | "
              f"сумма входа: {sum_in}")


async def main():
    from langgraph_executor.aegra_agents.json_analyzer_causal.graph import graph

    sample = os.environ.get("E2E_SAMPLE", "samples/test_metrics.json")
    question = os.environ.get(
        "E2E_QUESTION",
        "Почему производительность команды изменилась за последнюю неделю? "
        "Что именно её потянуло — разложи по компонентам.",
    )
    data = json.load(open(sample, encoding="utf-8"))
    print(f"sample: {sample}  | question: {question}\n")

    tracker = UsageTracker()
    state = {
        "raw_json": data,
        "question": question,
        "direction_key": "e2e-causal-test",
    }
    result = await graph.ainvoke(
        state,
        {"configurable": {"direction_key": "e2e-causal-test"}, "callbacks": [tracker]},
    )

    steps = result.get("tool_steps") or []
    print("=" * 70)
    print("TOOL STEPS (что вызвал агент):")
    for s in steps:
        print(f"  - {s.get('tool')}({json.dumps(s.get('args', {}), ensure_ascii=False)})")
    used = {s.get("tool") for s in steps}
    print(f"\nattribute_change вызван: {'ДА' if 'attribute_change' in used else 'нет'}")
    print(f"attribute_anomaly вызван: {'ДА' if 'attribute_anomaly' in used else 'нет'}")

    tracker.report()

    print("=" * 70)
    print("ОТВЕТ:")
    print(result.get("answer", "<нет ответа>"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
