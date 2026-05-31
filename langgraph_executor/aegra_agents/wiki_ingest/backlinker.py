"""Back-link: дописать [[…]] ссылки в старые страницы после появления новых сущностей.

``wiki_link`` — производный индекс из ``body_md``. Если страница X была создана
до того, как в wiki появилась страница Y, и упоминает Y голым текстом, ссылки
``X → Y`` не существует, пока кто-нибудь не перепишет ``X.body_md``.

Порт easyRag/wiki/backlinker.py с фильтрацией каталога и кандидатов по
``direction_key`` — back-link идёт строго внутри направления.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..easyrag.models import WikiPage
from .config import Settings, get_settings
from .embeddings import EmbeddingClient, get_embeddings
from .llm import LLMClient, get_llm
from .markdown import parse_page, strip_self_links
from .merge_utils import (
    reembed_sections,
    restore_provenance_by_anchor,
    snapshot_provenance,
)
from .repository import upsert_page
from .resolve_prompts import (
    WIKI_RELINK_SCHEMA,
    WIKI_RELINK_SYSTEM,
    WIKI_RELINK_TOOL_DESCRIPTION,
    WIKI_RELINK_TOOL_NAME,
    build_relink_user_prompt,
)

# Алиасы/title короче этой длины игнорируются prefilter'ом.
_MIN_FRESH_TOKEN_LEN = 3


@dataclass(frozen=True)
class BackfillResult:
    relinked: tuple[str, ...] = field(default_factory=tuple)
    skipped: tuple[str, ...] = field(default_factory=tuple)

    @property
    def relinked_count(self) -> int:
        return len(self.relinked)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


async def backfill_links(
    session: AsyncSession,
    *,
    direction_key: str,
    exclude_slugs: Iterable[str] = (),
    only_slugs: Iterable[str] | None = None,
    force: bool = False,
    prefilter: bool | None = None,
    llm: LLMClient | None = None,
    embeddings: EmbeddingClient | None = None,
    settings: Settings | None = None,
) -> BackfillResult:
    """Прогнать relink-LLM по страницам направления.

    По умолчанию проходит по всем страницам, кроме ``exclude_slugs``. Если задан
    ``only_slugs`` — обрабатывает ТОЛЬКО их (например, свежие страницы документа,
    чтобы проставить ссылки на только что созданные заглушки). ``prefilter``
    переопределяет ``settings.backlink_prefilter`` (для свежих страниц его стоит
    выключить: substring-prefilter не ловит склонённые формы — «бабушка» vs
    «бабушки»). Транзакцию ведёт caller.
    """
    cfg = settings or get_settings()
    if not cfg.backlink_enabled:
        return BackfillResult()

    effective_prefilter = cfg.backlink_prefilter if prefilter is None else prefilter
    exclude_set = {s for s in exclude_slugs if s}
    only_set = {s for s in only_slugs if s} if only_slugs is not None else None
    if only_set is not None and not only_set:
        return BackfillResult()
    if only_set is None and not exclude_set and not force:
        # Без триггера (ничего не создано/не merged) — лишних проходов не делаем.
        return BackfillResult()

    catalog_rows = (
        await session.execute(
            select(WikiPage.slug, WikiPage.title, WikiPage.aliases)
            .where(WikiPage.direction_key == direction_key)
            .order_by(WikiPage.updated_at.desc())
            .limit(cfg.merge_catalog_limit)
        )
    ).all()
    if not catalog_rows:
        return BackfillResult()

    catalog_by_slug: dict[str, tuple[str, list[str]]] = {
        slug: (title, list(aliases or [])) for slug, title, aliases in catalog_rows
    }
    catalog_strings_by_slug: dict[str, list[str]] = {
        slug: _strings_for(title, aliases)
        for slug, (title, aliases) in catalog_by_slug.items()
    }

    candidate_where = [WikiPage.direction_key == direction_key]
    if only_set is not None:
        candidate_where.append(WikiPage.slug.in_(only_set))
    else:
        candidate_where.append(WikiPage.slug.notin_(exclude_set))
    candidate_rows = (
        await session.execute(select(WikiPage).where(*candidate_where))
    ).scalars().all()
    if not candidate_rows:
        return BackfillResult()

    llm_client = llm or get_llm()
    embedder = embeddings or get_embeddings()

    relinked: list[str] = []
    skipped: list[str] = []

    for page in candidate_rows:
        page_catalog_slugs = {
            s for s in catalog_by_slug
            if s != page.slug and not _slugs_overlap(s, page.slug)
        }
        if not page_catalog_slugs:
            skipped.append(page.slug)
            continue

        relevant_slugs = _relevant_fresh_for_page(
            page,
            fresh_strings_by_slug={
                s: catalog_strings_by_slug[s] for s in page_catalog_slugs
            },
            prefilter=effective_prefilter,
        )
        if effective_prefilter and not relevant_slugs:
            skipped.append(page.slug)
            continue

        target_slugs = relevant_slugs if relevant_slugs else page_catalog_slugs
        catalog = [catalog_by_slug[slug] for slug in target_slugs]
        if not catalog:
            skipped.append(page.slug)
            continue

        prov_snapshot = await snapshot_provenance(session, page.id)
        current_body = page.body_md or ""
        current_aliases = list(page.aliases or [])

        user_prompt = build_relink_user_prompt(
            title=page.title,
            current_body=current_body,
            current_aliases=current_aliases,
            catalog=catalog,
        )
        raw = await llm_client.call_json(
            system=WIKI_RELINK_SYSTEM,
            user=user_prompt,
            tool_name=WIKI_RELINK_TOOL_NAME,
            tool_description=WIKI_RELINK_TOOL_DESCRIPTION,
            input_schema=WIKI_RELINK_SCHEMA,
        )

        new_body = _coerce_body(raw, fallback=current_body)
        new_body = strip_self_links(new_body, page_slug=page.slug)
        new_aliases = _coerce_aliases(raw, fallback=current_aliases)

        if not _has_link_change(current_body, new_body) and new_aliases == current_aliases:
            skipped.append(page.slug)
            continue

        updated = await upsert_page(
            session,
            direction_key=direction_key,
            slug=page.slug,
            title=page.title,
            body_md=new_body,
            aliases=new_aliases,
        )
        await reembed_sections(session, updated, embedder)
        await restore_provenance_by_anchor(session, updated.id, prov_snapshot)
        relinked.append(page.slug)

    await session.flush()
    return BackfillResult(relinked=tuple(relinked), skipped=tuple(skipped))


def _slugs_overlap(a: str, b: str) -> bool:
    """True если один slug является substring другого (``медведь`` ↔ ``медведь-косолапый``)."""
    if not a or not b or a == b:
        return False
    return a in b or b in a


def _strings_for(title: str | None, aliases: list[str] | None) -> list[str]:
    """Сформировать набор «достаточно длинных» имён для substring-prefilter'а."""
    out: list[str] = []
    if title:
        t = title.strip()
        if len(t) >= _MIN_FRESH_TOKEN_LEN:
            out.append(t)
    for a in aliases or []:
        if not a:
            continue
        a = a.strip()
        if len(a) >= _MIN_FRESH_TOKEN_LEN:
            out.append(a)
    return out


def _relevant_fresh_for_page(
    page: WikiPage,
    *,
    fresh_strings_by_slug: dict[str, list[str]],
    prefilter: bool,
) -> set[str]:
    """Какие fresh-slug'и реально имеет смысл сватать этой странице."""
    if not prefilter:
        return set()

    body = page.body_md or ""
    if not body:
        return set()
    body_lower = body.lower()

    parsed = parse_page(body)
    already_linked = {
        link.to_slug for section in parsed.sections for link in section.links
    }

    relevant: set[str] = set()
    for slug, strings in fresh_strings_by_slug.items():
        if slug in already_linked:
            continue
        for s in strings:
            if s.lower() in body_lower:
                relevant.add(slug)
                break
    return relevant


def _has_link_change(old_body: str, new_body: str) -> bool:
    """Проверить, изменился ли граф ссылок (а не только пробелы/пунктуация)."""
    old = {
        (s.anchor, link.to_slug)
        for s in parse_page(old_body).sections
        for link in s.links
    }
    new = {
        (s.anchor, link.to_slug)
        for s in parse_page(new_body).sections
        for link in s.links
    }
    return old != new


def _coerce_body(raw: dict | None, *, fallback: str) -> str:
    """Достать ``body_md`` из ответа LLM; на мусор — fallback на текущее тело."""
    if isinstance(raw, dict):
        body = raw.get("body_md")
        if isinstance(body, str) and body.strip():
            return body.strip()
    return (fallback or "").strip()


def _coerce_aliases(raw: dict | None, *, fallback: list[str]) -> list[str]:
    """Достать ``aliases`` из ответа LLM; на мусор — отдать те, что были."""
    if isinstance(raw, dict):
        raw_aliases = raw.get("aliases")
        if isinstance(raw_aliases, list):
            cleaned = [
                a.strip()
                for a in raw_aliases
                if isinstance(a, str) and a and a.strip()
            ]
            if cleaned:
                return cleaned
    return list(fallback)


__all__ = ["BackfillResult", "backfill_links"]
