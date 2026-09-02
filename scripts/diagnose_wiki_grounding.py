"""Диагностика wiki-grounding на живом GigaChat — без сервиса датасета.

Отвечает на вопрос «почему нет обращений к wiki»: прогоняет реальный путь
grounding по локальному сэмплу метрик и печатает, на каком шаге рвётся.

Проверяет по порядку:
  1) direction_key (= position.lower()) и сколько секций под него в wiki-БД;
  2) _distinct_metric_specs — извлеклись ли метрики из сэмпла;
  3) _generate_wiki_queries (ЖИВОЙ GigaChat) — вернул ли запросы или [];
  4) _gather_wiki_snippets через easyrag-подграф (ЖИВЫЕ эмбеддинги) — есть ли хиты.

Запуск (нужны креды GigaChat и доступ к БД, как у боевого приложения):

    GIGACHAT_CREDENTIALS=... .venv/bin/python scripts/diagnose_wiki_grounding.py \
        --sample samples/sample_declining.json --position Аналитик

Совет: --position Аналитик → direction_key 'аналитик', под который в БД есть
данные. Любая другая должность ожидаемо даст 0 секций.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import func, select

from langgraph_executor.aegra_agents.analytic_orchestrator.nodes import (
    _DEFAULT_EASYRAG_TOP_K,
    _DEFAULT_WIKI_MAX_QUERIES,
    _distinct_metric_specs,
    _format_metric_specs,
    _gather_wiki_snippets,
    _generate_wiki_queries,
)
from langgraph_executor.aegra_agents.easyrag.db import session_scope
from langgraph_executor.aegra_agents.easyrag.graph import build_graph as build_easyrag
from langgraph_executor.aegra_agents.easyrag.models import WikiSection
from langgraph_executor.aegra_agents.shared.clients import create_gigachat_client
from langgraph_executor.aegra_agents.shared.orgstructure import (
    IsuEmployeeOrgstructureInfo,
)


def _line(title: str) -> None:
    print(f"\n=== {title} ===")


async def _count_sections(direction_key: str) -> tuple[int, int]:
    async with session_scope() as s:
        total = (await s.execute(
            select(func.count()).select_from(WikiSection)
            .where(WikiSection.direction_key == direction_key)
        )).scalar_one()
        emb = (await s.execute(
            select(func.count(WikiSection.embedding))
            .where(WikiSection.direction_key == direction_key)
        )).scalar_one()
    return total, emb


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, help="JSON-файл с датасетом метрик")
    ap.add_argument("--position", default="Аналитик", help="должность сотрудника")
    ap.add_argument("--question", default="", help="вопрос руководителя (пусто = первичный обзор)")
    ap.add_argument("--max-queries", type=int, default=_DEFAULT_WIKI_MAX_QUERIES)
    ap.add_argument("--top-k", type=int, default=_DEFAULT_EASYRAG_TOP_K)
    args = ap.parse_args()

    if not os.environ.get("GIGACHAT_CREDENTIALS"):
        print("⚠️  GIGACHAT_CREDENTIALS не задан — генерация запросов и эмбеддинги упадут.")

    with open(args.sample, encoding="utf-8") as f:
        metrics = json.load(f)

    direction_key = IsuEmployeeOrgstructureInfo(
        manager_id="0", position=args.position, employee_id="0",
    ).direction_key()

    # 1) direction_key + данные в БД
    _line("1. direction_key и данные в wiki-БД")
    total, emb = await _count_sections(direction_key)
    print(f"position={args.position!r} → direction_key={direction_key!r}")
    print(f"секций в БД под это направление: {total} (с эмбеддингом: {emb})")
    if total == 0:
        print("❌ ГЕЙТ: данных под это направление нет — wiki вернёт пусто на всех путях.")
        print("   (это не баг кода: нужно либо другая должность, либо наполнить wiki)")

    # 2) метрики из сэмпла
    _line("2. _distinct_metric_specs (извлечение метрик из сэмпла)")
    specs = _distinct_metric_specs(metrics)
    print(f"метрик распознано: {len(specs)}")
    if not specs:
        print("❌ ГЕЙТ: метрики не извлеклись — grounding выйдет в return {}.")
        return
    print(_format_metric_specs(specs[:10]))

    # 3) генерация wiki-запросов живым GigaChat
    _line("3. _generate_wiki_queries (ЖИВОЙ GigaChat)")
    llm = create_gigachat_client().get_llm()
    state = {
        "messages": [{"type": "human", "content": args.question}] if args.question else [],
        "metrics": metrics,
        "direction_key": direction_key,
    }
    # _generate_wiki_queries читает _last_user_text — подсунем простой state-объект
    from langchain_core.messages import HumanMessage
    state["messages"] = [HumanMessage(content=args.question)] if args.question else []
    queries = _generate_wiki_queries(llm, state, specs, max_n=args.max_queries)
    print(f"сгенерировано запросов: {len(queries)}")
    for q in queries:
        print(f"  • {q}")
    if not queries:
        print("❌ ГЕЙТ: модель вернула [] — easyrag НЕ вызывается (nodes.py:598).")
        print("   Это самый частый тихий обрыв. Промпт разрешает [] (prompts.py).")
        return

    # 4) retrieval через easyrag-подграф
    _line("4. _gather_wiki_snippets (easyrag-подграф, ЖИВЫЕ эмбеддинги)")
    easyrag = build_easyrag()
    snippets, err = await _gather_wiki_snippets(
        easyrag, queries, direction_key=direction_key, top_k=args.top_k
    )
    if err:
        print(f"⚠️  ошибка retrieval: {err}")
    print(f"найдено сниппетов: {len(snippets)}")
    for s in snippets[:args.top_k]:
        sim = s.get("similarity")
        sim_str = f"{sim:.3f}" if isinstance(sim, (int, float)) else "-"
        print(f"  sim={sim_str} {s.get('page_title')} / {s.get('section_title')}")

    _line("ИТОГ")
    if snippets:
        print("✅ wiki-grounding отработал полностью: запросы → сниппеты получены.")
        print("   Если в проде их не видно — проверь intent (grounding только на")
        print("   turn-1 и analytics) и что json_analyzer wiki НЕ использует by design.")
    else:
        print("⚠️  запросы были, но сниппетов нет — данных под направление мало/нерелевантны.")


if __name__ == "__main__":
    asyncio.run(main())
