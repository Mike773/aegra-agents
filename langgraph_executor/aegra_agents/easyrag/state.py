from __future__ import annotations

from typing import TypedDict


class EasyRagState(TypedDict, total=False):
    query: str
    direction_key: str
    top_k: int

    query_vec: list[float]
    snippets: list[dict]
    gap_recorded: bool
