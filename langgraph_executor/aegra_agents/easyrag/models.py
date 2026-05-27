"""ORM-модели подагента easyrag.

Схема ``wiki_rag``. Размер вектора фиксирован — 1024 (GigaChat).
Каждая таблица, по которой идёт выборка/запись, несёт ``direction_key``.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    TIMESTAMP,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBED_DIM = 1024
SCHEMA = "wiki_rag"


class Base(DeclarativeBase):
    pass


class WikiPage(Base):
    __tablename__ = "wiki_page"
    __table_args__ = (
        UniqueConstraint("direction_key", "slug", name="uq_wiki_page_direction_slug"),
        Index("ix_wiki_page_direction", "direction_key"),
        Index("ix_wiki_page_type", "type"),
        {"schema": SCHEMA},
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str | None] = mapped_column(String(64))
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    body_md: Mapped[str] = mapped_column(Text, nullable=False, default="")
    direction_key: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    sections: Mapped[list["WikiSection"]] = relationship(
        back_populates="page",
        cascade="all, delete-orphan",
        order_by="WikiSection.ord",
    )


class WikiSection(Base):
    __tablename__ = "wiki_section"
    __table_args__ = (
        UniqueConstraint("page_id", "ord", name="uq_wiki_section_page_ord"),
        UniqueConstraint("page_id", "anchor", name="uq_wiki_section_page_anchor"),
        Index("ix_wiki_section_direction", "direction_key"),
        {"schema": SCHEMA},
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    page_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.wiki_page.id", ondelete="CASCADE"),
        nullable=False,
    )
    ord: Mapped[int] = mapped_column(Integer, nullable=False)
    anchor: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body_md: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))
    direction_key: Mapped[str] = mapped_column(Text, nullable=False)

    page: Mapped[WikiPage] = relationship(back_populates="sections")


class WikiLink(Base):
    """Производный индекс рёбер. Пересобирается из wiki_page.body_md."""

    __tablename__ = "wiki_link"
    __table_args__ = (
        Index("ix_wiki_link_to_slug", "to_slug"),
        Index("ix_wiki_link_to_page", "to_page_id"),
        {"schema": SCHEMA},
    )

    from_page_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.wiki_page.id", ondelete="CASCADE"),
        primary_key=True,
    )
    from_section_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.wiki_section.id", ondelete="CASCADE"),
        primary_key=True,
    )
    to_slug: Mapped[str] = mapped_column(String(255), primary_key=True)
    to_page_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.wiki_page.id", ondelete="SET NULL"),
    )


class QueryGap(Base):
    __tablename__ = "query_gap"
    __table_args__ = (
        Index("ix_query_gap_direction", "direction_key"),
        Index("ix_query_gap_resolved_at", "resolved_at"),
        {"schema": SCHEMA},
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))
    direction_key: Mapped[str] = mapped_column(Text, nullable=False)
    asked_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    resolved_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    resolved_section_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)), nullable=False, default=list
    )
    unresolved_abbr: Mapped[list[str]] = mapped_column(
        ARRAY(String(32)), nullable=False, default=list
    )


__all__ = ["Base", "WikiPage", "WikiSection", "WikiLink", "QueryGap", "EMBED_DIM", "SCHEMA"]
