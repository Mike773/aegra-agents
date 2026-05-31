"""Запись wiki-страниц в БД и пересборка link-индекса.

Порт из easyRag с фильтрацией по ``direction_key``: страница ищется/создаётся
в пределах направления (uq по ``(direction_key, slug)``), а разрешение слагов в
``wiki_link`` идёт только среди страниц того же направления — иначе ``[[Лиса]]``
из направления A могла бы разрешиться в страницу направления B при совпадении
slug'а.

``wiki_link`` — производный индекс рёбер: всегда строится из ``wiki_page.body_md``
и никогда не редактируется руками. Любая операция, меняющая markdown страницы,
обязана пересобрать её рёбра здесь же.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..easyrag.models import WikiLink, WikiPage, WikiSection
from .markdown import ParsedPage, parse_page
from .sanitize import sanitize_body_md
from .slug import make_slug

logger = logging.getLogger(__name__)


async def upsert_page(
    session: AsyncSession,
    *,
    direction_key: str,
    slug: str,
    title: str,
    body_md: str,
    type_: str | None = None,
    aliases: Sequence[str] = (),
) -> WikiPage:
    """Создать или обновить страницу направления и пересинхронизировать секции + рёбра.

    Возвращает persistent ``WikiPage``. Транзакцию ведёт вызывающий код.

    Семантика опциональных полей на UPDATE-пути:
    * ``aliases`` всегда перезаписывается (``()`` очищает).
    * ``type_=None`` означает «не трогать существующее значение».
    """
    aliases_list = list(aliases)
    body_md, repairs = sanitize_body_md(body_md, page_slug=slug)
    if repairs:
        logger.warning(
            "upsert_page sanitized body_md for %s/%s: %s",
            direction_key,
            slug,
            "; ".join(repairs),
        )
    parsed = parse_page(body_md)

    page = (
        await session.execute(
            select(WikiPage).where(
                WikiPage.direction_key == direction_key,
                WikiPage.slug == slug,
            )
        )
    ).scalar_one_or_none()

    if page is None:
        page = WikiPage(
            slug=slug,
            title=title,
            body_md=body_md,
            type=type_,
            aliases=aliases_list,
            direction_key=direction_key,
        )
        session.add(page)
        await session.flush()  # получить page.id для секций
    else:
        page.title = title
        page.body_md = body_md
        page.aliases = aliases_list
        if type_ is not None:
            page.type = type_
        page.version = (page.version or 0) + 1
        # Сносим старые секции — CASCADE удалит привязанные wiki_link.
        await session.execute(
            delete(WikiSection).where(WikiSection.page_id == page.id)
        )
        await session.flush()

    for ps in parsed.sections:
        session.add(
            WikiSection(
                page_id=page.id,
                ord=ps.ord,
                anchor=ps.anchor,
                title=ps.title,
                body_md=ps.body_md,
                direction_key=direction_key,
            )
        )
    await session.flush()

    await _insert_links_for_page(session, page, parsed)
    await _resolve_dangling_links_for_slug(session, page.slug, page.id, direction_key)
    return page


async def ensure_stub_page(
    session: AsyncSession,
    *,
    direction_key: str,
    name: str,
) -> WikiPage | None:
    """Создать пустую страницу-заглушку (``type='stub'``), если её ещё нет.

    Wikipedia-like: сущность упомянута, но фактов о ней пока нет — заводим
    плейсхолдер с пустым ``body_md`` (0 секций, без embedding'а), который
    наполнится позже обычным merge при появлении фактов из нового источника.
    Возвращает созданную страницу или ``None``, если страница с таким slug в
    этом ``direction_key`` уже существует (любого типа).
    """
    name = (name or "").strip()
    if not name:
        return None
    slug = make_slug(name)
    existing = (
        await session.execute(
            select(WikiPage).where(
                WikiPage.direction_key == direction_key,
                WikiPage.slug == slug,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        return None
    return await upsert_page(
        session,
        direction_key=direction_key,
        slug=slug,
        title=name,
        body_md="",
        type_="stub",
    )


async def rebuild_page_links(session: AsyncSession, page: WikiPage) -> None:
    """Пересобрать ``wiki_link`` для одной страницы по её текущему ``body_md``."""
    parsed = parse_page(page.body_md)
    await session.execute(
        delete(WikiLink).where(WikiLink.from_page_id == page.id)
    )
    await session.flush()
    await _insert_links_for_page(session, page, parsed)
    await _resolve_dangling_links_for_slug(
        session, page.slug, page.id, page.direction_key
    )


async def _resolve_dangling_links_for_slug(
    session: AsyncSession, slug: str, page_id: object, direction_key: str
) -> None:
    """Заполнить ``to_page_id`` у строк ``wiki_link`` направления, ссылающихся на ``slug``.

    Резолвим только рёбра, исходящие из страниц этого же направления — чтобы не
    «склеить» одинаковые slug'и разных направлений. ``to_page_id IS NULL``
    гарантирует, что мы не затрём уже разрешённые ссылки.
    """
    from_in_direction = select(WikiPage.id).where(
        WikiPage.direction_key == direction_key
    )
    await session.execute(
        update(WikiLink)
        .where(
            WikiLink.to_slug == slug,
            WikiLink.to_page_id.is_(None),
            WikiLink.from_page_id.in_(from_in_direction),
        )
        .values(to_page_id=page_id)
    )


async def _insert_links_for_page(
    session: AsyncSession, page: WikiPage, parsed: ParsedPage
) -> None:
    """Вставить ``wiki_link`` строки для распарсенных секций страницы.

    ``to_page_id`` подставляется по существующим страницам ТОГО ЖЕ направления;
    неизвестные slug'и остаются NULL (висячие ссылки).
    """
    if not parsed.sections:
        return

    sections = (
        await session.execute(
            select(WikiSection).where(WikiSection.page_id == page.id)
        )
    ).scalars().all()
    section_by_anchor = {s.anchor: s for s in sections}

    referenced = {link.to_slug for ps in parsed.sections for link in ps.links}
    slug_to_id: dict[str, object] = {}
    if referenced:
        rows = (
            await session.execute(
                select(WikiPage.slug, WikiPage.id).where(
                    WikiPage.slug.in_(referenced),
                    WikiPage.direction_key == page.direction_key,
                )
            )
        ).all()
        slug_to_id = {s: i for s, i in rows}

    for ps in parsed.sections:
        sec = section_by_anchor.get(ps.anchor)
        if sec is None:  # invariant: section just inserted
            continue
        # PK = (from_page_id, from_section_id, to_slug) — дедуплицируем в секции.
        seen: set[str] = set()
        for link in ps.links:
            if link.to_slug in seen:
                continue
            seen.add(link.to_slug)
            session.add(
                WikiLink(
                    from_page_id=page.id,
                    from_section_id=sec.id,
                    to_slug=link.to_slug,
                    to_page_id=cast("object", slug_to_id.get(link.to_slug)),
                )
            )
    await session.flush()


__all__ = [
    "ensure_stub_page",
    "rebuild_page_links",
    "upsert_page",
]
