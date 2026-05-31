"""Стабильный slug для wiki — Unicode (включая кириллицу), lowercase, dash-separated.

Используется и для ``wiki_page.slug`` (первичный ключ страницы для линковки),
и для ``wiki_section.anchor`` (уникален в пределах страницы). Кириллица не
транслитерируется: «медведь» → «медведь», «лягушка-квакушка» → «лягушка-квакушка».
Это даёт читаемые идентификаторы и совпадает с тем, что LLM пишет в [[…]]:
``make_slug("[[Медведь]]")`` сразу матчится со страницей «медведь» без
дополнительных алиасов.

Slug всегда получается не пустым: если ``python-slugify`` вернул пустую строку
(например, на чисто-символьном вводе), берётся детерминированный хеш.
"""
from __future__ import annotations

import hashlib

from slugify import slugify

_MAX_LEN = 200


def make_slug(text: str, *, max_length: int = _MAX_LEN) -> str:
    s = slugify(
        text or "",
        lowercase=True,
        max_length=max_length,
        allow_unicode=True,
    )
    if s:
        return s
    digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]
    return f"page-{digest}"


__all__ = ["make_slug"]
