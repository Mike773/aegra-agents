"""Эмбеддинги поверх GigaChat.

Тонкий адаптер над ``GigaChatEmbeddings`` из общего ``shared.clients``.
Предпочитаем ``aembed_documents``; иначе уходим в поток (``asyncio.to_thread``),
как уже сделано в ``easyrag.nodes``. Размерность — 1024 (фиксируется моделью).
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..shared.clients import create_gigachat_embeddings


class EmbeddingClient:
    def __init__(self, embedder: Any = None) -> None:
        self._embedder = embedder if embedder is not None else create_gigachat_embeddings()

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        aembed = getattr(self._embedder, "aembed_documents", None)
        if aembed is not None:
            return [list(v) for v in await aembed(texts)]
        vectors = await asyncio.to_thread(self._embedder.embed_documents, texts)
        return [list(v) for v in vectors]

    async def embed_one(self, text: str) -> list[float]:
        out = await self.embed_many([text])
        return out[0]


_default_embeddings: EmbeddingClient | None = None


def get_embeddings() -> EmbeddingClient:
    global _default_embeddings
    if _default_embeddings is None:
        _default_embeddings = EmbeddingClient()
    return _default_embeddings


__all__ = ["EmbeddingClient", "get_embeddings"]
