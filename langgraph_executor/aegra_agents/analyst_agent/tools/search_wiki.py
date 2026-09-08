"""Поиск в базе знаний через подграф easyrag.

Это справка для интерпретации показателей (методика, нормативы, расшифровка
аббревиатур), а НЕ источник чисел: числа всегда из данных.

Найденное копится в контексте запуска и уезжает в поля easyrag_* стейта, чтобы
опись источников в describe_answer работала так же, как в v4.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

_SNIPPET_PREVIEW = 400
_MAX_SNIPPETS = 5


def _format(snippets: list[dict[str, Any]]) -> str:
    lines = ["Найдено в базе знаний:"]
    for s in snippets[:_MAX_SNIPPETS]:
        page = s.get("page_title") or s.get("slug") or "-"
        title = s.get("section_title") or s.get("anchor") or "-"
        body = " ".join(str(s.get("body_md") or "").split())
        if len(body) > _SNIPPET_PREVIEW:
            body = body[:_SNIPPET_PREVIEW] + "…"
        lines.append(f"- [{page} / {title}]: {body}")
    lines.append(
        "Это справка для интерпретации: числа и вердикты бери только из данных."
    )
    return "\n".join(lines)


def make_search_wiki(ctx: Any) -> StructuredTool:
    async def search_wiki(query: str) -> str:
        ctx.used_wiki_tool = True
        ctx.wiki_queries.append(query)
        if not ctx.direction_key:
            return "База знаний недоступна: направление не определено."
        try:
            result = await ctx.easyrag_graph.ainvoke(
                {
                    "query": query,
                    "direction_key": ctx.direction_key,
                    "top_k": ctx.easyrag_top_k,
                }
            )
        except Exception as exc:  # noqa: BLE001 — БД/сеть, ход не роняем
            logger.warning("search_wiki: подграф упал", exc_info=True)
            ctx.wiki_error = str(exc)
            return "Не удалось обратиться к базе знаний."

        snippets = [s for s in (result or {}).get("snippets") or [] if isinstance(s, dict)]
        if not snippets:
            return (
                f"По запросу «{query}» в базе знаний ничего не нашлось. "
                "Опирайся на сами данные."
            )
        known = {s.get("section_id") for s in ctx.wiki_snippets}
        for s in snippets:
            if s.get("section_id") not in known:
                ctx.wiki_snippets.append(s)
        return _format(snippets)

    return StructuredTool.from_function(
        coroutine=search_wiki,
        name="search_wiki",
        description=(
            "Найти в корпоративной базе знаний методику, норматив или расшифровку "
            "аббревиатуры — справку для интерпретации показателя, а не источник "
            "чисел: числа берутся только из данных. Аргумент: query — короткий "
            "вопрос по-русски."
        ),
    )


__all__ = ["make_search_wiki"]
