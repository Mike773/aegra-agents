"""Поиск в базе знаний через подграф easyrag.

Это справка для интерпретации показателей (методика, нормативы, расшифровка
аббревиатур), а НЕ источник чисел: числа всегда из данных.

Найденное копится в контексте запуска и уезжает в поля easyrag_* стейта, чтобы
опись источников в describe_answer работала так же, как в v4.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

# Модели нужен сам текст справки, а не опись найденного: секции отдаются целиком,
# режется только то, что не влезает в общий бюджет ответа инструмента.
_MAX_SECTION_CHARS = 2500
_MAX_TOTAL_CHARS = 8000

# [[target]] / [[target|display]] → видимое имя: разметка ссылок модели не нужна.
_WIKI_LINK_RE = re.compile(r"\[\[([^\[\]|\n]+?)(?:\|([^\[\]\n]+?))?]]")


def _clean_body(body: Any) -> str:
    text = _WIKI_LINK_RE.sub(lambda m: (m.group(2) or m.group(1)).strip(), str(body or ""))
    # Пустые строки схлопываются, переносы внутри секции сохраняются: списки
    # и таблицы markdown без них не читаются.
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    return "\n".join(ln for ln in lines if ln.strip())


def _group_by_page(snippets: list[dict[str, Any]]) -> list[tuple[str, bool, list[dict]]]:
    """Секции по страницам в порядке первого появления страницы в выдаче.

    Страница помечается связанной, только если все её секции пришли из
    расширения по ссылкам (``source='graph'``), а не по самому запросу.
    """
    pages: dict[str, tuple[str, list[dict]]] = {}
    for s in snippets:
        key = str(s.get("page_id") or s.get("slug") or s.get("page_title") or "-")
        title = s.get("page_title") or s.get("slug") or "-"
        pages.setdefault(key, (title, []))[1].append(s)
    return [
        (title, all(s.get("source") == "graph" for s in secs), secs)
        for title, secs in pages.values()
    ]


def _format(snippets: list[dict[str, Any]]) -> str:
    lines = ["Найдено в базе знаний:"]
    budget = _MAX_TOTAL_CHARS
    skipped: list[str] = []
    for page, related, secs in _group_by_page(snippets):
        header = f"\n### {page}" + (" (связанная страница)" if related else "")
        page_lines: list[str] = []
        for s in secs:
            title = s.get("section_title") or s.get("anchor") or "-"
            body = _clean_body(s.get("body_md"))
            if not body:
                continue
            if budget <= 0:
                skipped.append(f"{page} / {title}")
                continue
            limit = min(_MAX_SECTION_CHARS, budget)
            if len(body) > limit:
                body = body[:limit].rstrip() + "…"
            budget -= len(body)
            page_lines.append(f"#### {title}\n{body}")
        if page_lines:
            lines.append(header)
            lines.extend(page_lines)
    if len(lines) == 1 and not skipped:
        return (
            "В базе знаний нашлись только пустые страницы без текста. "
            "Опирайся на сами данные."
        )
    if skipped:
        lines.append(
            "\nНе поместились в ответ (уточни запрос, если нужны): "
            + "; ".join(skipped)
        )
    lines.append(
        "\nЭто справка для интерпретации: числа и вердикты бери только из данных."
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
