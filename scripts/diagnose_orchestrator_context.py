"""ДИАГНОСТИКА: откуда analytic_orchestrator берёт значения на 2-м+ ходе.

Симптом: на втором и последующих вопросах в рамках одного thread'а непонятно,
откуда берутся данные, и ассистент ошибается в значениях метрик. Этот скрипт
НИЧЕГО не правит в графе/узлах — он лишь прогоняет несколько ходов in-process
(как scripts/run_orchestrator.py, с MemorySaver вместо aegra) и после КАЖДОГО
хода реконструирует «карту источников» контекста, переиспользуя чистые хелперы
из nodes.py (_metrics_system_block / _easyrag_system_block / _history_for_llm).

Цель — глазами увидеть на реальном прогоне:
  * вызывался ли json_analyzer на этом ходе (пересчёт) или ответ из sticky;
  * какой ИМЕННО блок метрик увидел LLM и из какой ветки он взялся
    (sticky analytics_answer | metrics_error | сырой JSON);
  * на какой вопрос отвечал sticky (analytics_question) — не устарел ли;
  * какие прошлые числа притягиваются в историю (_history_for_llm).

Запуск (нужны GigaChat-креды и доступ к БД; дай ТАБЕЛЬНЫЙ С НАПОЛНЕННЫМИ
метриками, иначе значений не будет):

    DEMO_EMP_TABNUM=<наполненный> .venv/bin/python scripts/diagnose_orchestrator_context.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph_executor.aegra_agents.analytic_orchestrator import nodes
from langgraph_executor.aegra_agents.analytic_orchestrator.graph import build_graph
from langgraph_executor.aegra_agents.shared.clients import create_gigachat_client

# Табельников у нас нет — кормим граф готовым датасетом из samples/ (та же форма
# {"me", "employees":[{tabnum, fio, metrics:[…]}]}, что отдаёт реальный загрузчик).
# Подменяем ТОЛЬКО источник данных в load_data; топология/узлы не меняются, так
# что первый ход честно идёт load_data → ground_wiki_initial → initial_analysis.
_DEFAULT_SAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "samples", "sample_declining.json"
)


def _install_sample_dataset(path: str) -> None:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    class _FakeDatasetComponent:
        def __init__(self, *args, **kwargs):
            pass

        def build_json_output(self):
            return payload

    nodes.GetBatchAgentDatasetByFiltersComponent = _FakeDatasetComponent

# Сценарий ходов, провоцирующий баг: первичный анализ → узкий метрический вопрос
# (analytics перезапишет sticky) → ДРУГОЙ метрический вопрос → chat-уточнение
# (увидит уже суженный sticky). Подмени под свои реальные метрики при желании.
TURNS = [
    "Проанализируй метрики сотрудника.",
    "Какое сейчас значение факта по «Производительности»?",
    "А какой факт по «Доле переводов»?",
    "Повтори, пожалуйста, точное значение Производительности, которое ты называл.",
    "Сделай детализацию по западающим.",  # referent — проверка обогащения вопроса
]

_PREVIEW = 280


def _short(text: object, limit: int = _PREVIEW) -> str:
    s = "" if text is None else str(text)
    s = s.replace("\n", " ⏎ ")
    return s if len(s) <= limit else s[:limit] + "…"


def _metrics_source(state: dict) -> str:
    """Та же логика ветвления, что и nodes._metrics_system_block — но возвращает
    ИМЯ ветки, чтобы пометить источник блока метрик."""
    if state.get("analytics_answer") and not state.get("analytics_error"):
        return "sticky analytics_answer"
    if state.get("metrics_error"):
        return "metrics_error"
    if state.get("metrics") is None:
        return "НЕТ (metrics is None)"
    return "сырой JSON метрик"


def _final_message(state: dict) -> str:
    for m in reversed(state.get("messages") or []):
        if nodes._is_step(m):
            continue
        who = getattr(m, "type", "")
        if who == "ai":
            return _short(getattr(m, "content", ""), 400)
    return "(итогового сообщения не нашлось)"


def _print_turn_path(turn_no: int, user_text: str, events: list[dict]) -> None:
    print(f"\n{'='*78}")
    print(f"ХОД {turn_no}. Пользователь: {user_text!r}")
    print(f"{'='*78}")
    nodes_ran = [node for ev in events for node in ev.keys()]
    print(f"  Путь хода (узлы по порядку): {' → '.join(nodes_ran) or '—'}")
    called_analyzer = "call_json_analyzer" in nodes_ran
    called_extract = "extract_assignments" in nodes_ran
    print(
        "  json_analyzer вызывался на этом ходе: "
        + ("ДА" if (called_analyzer or called_extract) else "НЕТ")
        + (
            f" (через {'call_json_analyzer' if called_analyzer else 'extract_assignments'})"
            if (called_analyzer or called_extract)
            else " → ответ собирается из sticky/истории"
        )
    )


def _print_context_map(state: dict, prev_question: object) -> None:
    print("  --- карта источников контекста (что увидел LLM) ---")
    print(f"  intent: {state.get('intent')!r}")

    cur_q = state.get("analytics_question")
    fresh = cur_q != prev_question and cur_q is not None
    print(
        f"  analytics_question (на какой вопрос отвечал sticky): {_short(cur_q)}"
        + ("   [СВЕЖИЙ на этом ходе]" if fresh else "   [ПРОШЛЫЙ ход / не менялся]")
    )
    if state.get("analytics_error"):
        print(f"  analytics_error: {_short(state.get('analytics_error'))}")
    ans = state.get("analytics_answer")
    print(f"  analytics_answer (sticky): {len(ans) if ans else 0} симв. — {_short(ans)}")

    block = nodes._metrics_system_block(state)
    print(f"  ИСТОЧНИК блока метрик в системном промпте: {_metrics_source(state)}")
    print(f"  _metrics_system_block → {_short(block, 360)}")

    wiki = nodes._easyrag_system_block(state)
    snips = state.get("easyrag_snippets") or []
    print(f"  wiki в контексте: {len(snips)} сниппет(ов) — {_short(wiki, 160)}")

    history = nodes._history_for_llm(state.get("messages") or [])
    print(f"  история для LLM ({len(history)} сообщ., шаги вырезаны):")
    for m in history:
        who = getattr(m, "type", m.__class__.__name__)
        print(f"      [{who}] {_short(getattr(m, 'content', ''), 160)}")

    print(f"  ИТОГ хода (то, что видит пользователь): {_final_message(state)}")


async def main() -> int:
    sample_path = os.environ.get("SAMPLE_FILE", _DEFAULT_SAMPLE)
    _install_sample_dataset(sample_path)
    print(f"Датасет из файла: {sample_path}")

    llm = create_gigachat_client().get_llm()
    graph = build_graph(llm)
    graph.checkpointer = MemorySaver()

    config = {
        "configurable": {
            "thread_id": "diag-thread-1",
            "boss_tabnum": os.environ.get("DEMO_BOSS_TABNUM", "100001"),
            "employee_tabnum": os.environ.get("DEMO_EMP_TABNUM", "100500"),
            "position": os.environ.get("DEMO_POSITION", "Аналитик"),
            # describe_answer/emit_progress оставляем по умолчанию — диагностике
            # шаговые сообщения не мешают (_history_for_llm их вырезает).
        }
    }

    print("Диагностика контекста analytic_orchestrator (in-process, MemorySaver).")
    print(
        f"Сотрудник={config['configurable']['employee_tabnum']} "
        f"босс={config['configurable']['boss_tabnum']} "
        f"должность={config['configurable']['position']}"
    )
    print("ВНИМАНИЕ: для реальных значений нужен табельный с наполненными метриками.")

    prev_question: object = None
    prev_state: dict = {}
    for i, user_text in enumerate(TURNS, 1):
        # Что уйдёт в json_analyzer (если ход классифицируется как analytics):
        # дословный вопрос, дообогащённый контекстом прошлых взаимодействий.
        enriched = nodes._question_with_dialogue_context(prev_state, user_text)

        events: list[dict] = []
        try:
            async for ev in graph.astream(
                {"messages": [HumanMessage(content=user_text)]},
                config=config,
                stream_mode="updates",
            ):
                events.append(ev)
        except Exception as exc:  # noqa: BLE001 — диагностика не должна падать на ходе
            print(f"\nХОД {i}: ошибка прогона: {type(exc).__name__}: {exc}")
            continue

        snapshot = await graph.aget_state(config)
        state = snapshot.values if snapshot else {}

        _print_turn_path(i, user_text, events)
        if enriched != user_text:
            print("  --- обогащённый вопрос к json_analyzer (referent-контекст) ---")
            print(f"  {_short(enriched, 600)}")
        _print_context_map(state, prev_question)
        prev_question = state.get("analytics_question")
        prev_state = state

    print(f"\n{'='*78}")
    print("Готово. Смотри строки «ИСТОЧНИК блока метрик» и «analytics_question»:")
    print("  • sticky + analytics_question ПРОШЛОГО хода → значения из устаревшего")
    print("    узкого ответа (гипотезы A/B); сырой JSON при этом замаскирован.")
    print("  • число из истории, а не из метрик → гипотеза C (история тянет цифры).")
    print("  • json_analyzer=НЕТ на метрическом вопросе → роутер недоклассифицировал.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
