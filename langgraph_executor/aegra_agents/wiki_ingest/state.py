from __future__ import annotations

from typing import TypedDict


class WikiIngestState(TypedDict, total=False):
    # Вход: обязателен.
    direction_key: str

    # Накопление / результат.
    pending_doc_ids: list[str]
    processed_doc_ids: list[str]
    created_pages: list[str]
    merged_pages: list[str]
    relinked_pages: list[str]
    created_stub_pages: list[str]
    ambiguous_count: int
    errors: list[dict]
    report: str
