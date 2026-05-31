"""Разрешение пользовательской ссылки на документ для удаления.

Сопоставляет свободную ссылку (``reference``) с последней выдачей списка
(``last_listed``). Порядок попыток: порядковый номер → префикс id → подстрока в
uri → fuzzy по uri (через ``shared.text_similarity`` — не stdlib difflib, его может
не быть в прод-бандле).
"""
from __future__ import annotations

from ..shared.text_similarity import similarity_ratio
from .config import get_settings


def resolve_delete_target(
    reference: str, listed: list[dict]
) -> tuple[str | None, str]:
    """Вернуть (doc_id|None, status), status ∈ {ok, not_found, ambiguous, empty}."""
    cfg = get_settings()
    ref = (reference or "").strip()
    if not listed:
        return None, "empty"

    # 1) Порядковый номер (1-based) против последней выдачи.
    if ref.isdigit():
        i = int(ref)
        if 1 <= i <= len(listed):
            return listed[i - 1]["id"], "ok"
        return None, "not_found"

    low = ref.casefold()
    if not low:
        return None, "not_found"

    # 2) Префикс id (UUID, ≥4 симв.).
    if len(low) >= 4:
        pref = [d for d in listed if str(d["id"]).lower().startswith(low)]
        if len(pref) == 1:
            return pref[0]["id"], "ok"
        if len(pref) > 1:
            return None, "ambiguous"

    # 3) Подстрока в uri.
    subs = [d for d in listed if low in (d.get("uri") or "").casefold()]
    if len(subs) == 1:
        return subs[0]["id"], "ok"
    if len(subs) > 1:
        return None, "ambiguous"

    # 4) Fuzzy по uri.
    scored = sorted(
        (
            (similarity_ratio(low, (d.get("uri") or "").casefold()), d)
            for d in listed
        ),
        key=lambda t: t[0],
        reverse=True,
    )
    if scored and scored[0][0] >= cfg.delete_fuzzy_threshold:
        if (
            len(scored) > 1
            and scored[1][0] >= cfg.delete_fuzzy_threshold
            and abs(scored[0][0] - scored[1][0]) < cfg.delete_fuzzy_ambiguity_gap
        ):
            return None, "ambiguous"
        return scored[0][1]["id"], "ok"

    return None, "not_found"


__all__ = ["resolve_delete_target"]
