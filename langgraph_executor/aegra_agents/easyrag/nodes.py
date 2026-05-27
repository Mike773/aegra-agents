from __future__ import annotations

import asyncio

from langchain_gigachat import GigaChatEmbeddings

from .db import session_scope
from .gap import record_gap
from .retrieval import retrieve_sections
from .state import EasyRagState

_DEFAULT_TOP_K = 8


def make_embed_query_node(embedder: GigaChatEmbeddings):
    async def embed_query(state: EasyRagState) -> dict:
        query = (state.get("query") or "").strip()
        if not query:
            return {"query_vec": []}
        aembed = getattr(embedder, "aembed_query", None)
        if aembed is not None:
            vec = await aembed(query)
        else:
            vec = await asyncio.to_thread(embedder.embed_query, query)
        return {"query_vec": list(vec)}

    return embed_query


def make_retrieve_node():
    async def retrieve(state: EasyRagState) -> dict:
        query_vec = state.get("query_vec") or []
        direction_key = (state.get("direction_key") or "").strip()
        top_k = int(state.get("top_k") or _DEFAULT_TOP_K)

        if not query_vec or not direction_key:
            return {"snippets": []}

        async with session_scope() as session:
            sections = await retrieve_sections(
                session,
                direction_key=direction_key,
                query_vec=query_vec,
                top_k=top_k,
            )
        return {"snippets": [s.to_dict() for s in sections]}

    return retrieve


def make_maybe_record_gap_node():
    async def maybe_record_gap(state: EasyRagState) -> dict:
        async with session_scope() as session:
            await record_gap(
                session,
                direction_key=(state.get("direction_key") or "").strip(),
                question=(state.get("query") or "").strip(),
                embedding=state.get("query_vec") or None,
                resolved_section_ids=(),
            )
        return {"gap_recorded": True}

    return maybe_record_gap


def after_retrieve(state: EasyRagState) -> str:
    # Если выборка пуста — фиксируем gap, иначе — заканчиваем.
    return "maybe_record_gap" if not (state.get("snippets") or []) else "__end__"


__all__ = [
    "make_embed_query_node",
    "make_retrieve_node",
    "make_maybe_record_gap_node",
    "after_retrieve",
]
