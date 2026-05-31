"""Разбиение исходного текста на чанки.

Чанки нужны для:
* провенанса — каждая wiki-секция указывает на чанк(и), из которых её собрали;
* enrichment-loop — на чанках строится поиск «эта тема в исходниках вообще была?».

Стратегия:
* Граница чанка — пустая строка (``\\n\\n+``). Это сохраняет смысловые абзацы
  целиком и не режет посреди предложения.
* Если получившийся «параграф-блок» сам по себе длиннее ``max_size``, режем
  его на куски по ``target_size`` с перекрытием ``overlap`` символов.
* Возвращаем char_start/char_end в исходном тексте — это нужно для провенанса.

Парсер чистый, без сети и БД, детерминированный.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_TARGET_SIZE = 1200
DEFAULT_MAX_SIZE = 1800
DEFAULT_OVERLAP = 150

_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    ord: int
    text: str
    char_start: int
    char_end: int


def chunk_text(
    text: str,
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Разбить ``text`` на упорядоченные чанки."""
    if not text or not text.strip():
        return []
    if overlap >= target_size:
        raise ValueError("overlap must be smaller than target_size")
    if max_size < target_size:
        raise ValueError("max_size must be >= target_size")

    paragraphs = _split_paragraphs(text)
    chunks: list[Chunk] = []
    buf: list[tuple[int, int, str]] = []
    buf_len = 0

    def _flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        start = buf[0][0]
        end = buf[-1][1]
        body = text[start:end]
        chunks.append(Chunk(ord=len(chunks), text=body, char_start=start, char_end=end))
        buf = []
        buf_len = 0

    for para_start, para_end in paragraphs:
        para_len = para_end - para_start
        if para_len > max_size:
            _flush()
            for piece_start, piece_end in _slice_with_overlap(
                para_start, para_end, target_size, overlap
            ):
                chunks.append(
                    Chunk(
                        ord=len(chunks),
                        text=text[piece_start:piece_end],
                        char_start=piece_start,
                        char_end=piece_end,
                    )
                )
            continue

        if buf and buf_len + para_len > target_size:
            _flush()
        buf.append((para_start, para_end, text[para_start:para_end]))
        buf_len += para_len

    _flush()
    return chunks


def _split_paragraphs(text: str) -> list[tuple[int, int]]:
    """Найти параграфы как непустые блоки между ``\\n\\n``."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    n = len(text)
    while cursor < n:
        m = _PARA_SPLIT_RE.search(text, cursor)
        end = m.start() if m else n
        block = text[cursor:end]
        lstripped = len(block) - len(block.lstrip())
        rstripped = len(block) - len(block.rstrip())
        start = cursor + lstripped
        stop = end - rstripped
        if stop > start:
            spans.append((start, stop))
        cursor = m.end() if m else n
    return spans


def _slice_with_overlap(
    start: int, end: int, size: int, overlap: int
) -> list[tuple[int, int]]:
    """Нарезать [start, end) на куски длины ``size`` с перекрытием ``overlap``."""
    out: list[tuple[int, int]] = []
    step = size - overlap
    pos = start
    while pos < end:
        piece_end = min(pos + size, end)
        out.append((pos, piece_end))
        if piece_end >= end:
            break
        pos += step
    return out


__all__ = ["Chunk", "chunk_text"]
