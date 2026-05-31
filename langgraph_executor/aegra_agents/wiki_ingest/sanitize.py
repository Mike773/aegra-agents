"""Санитизация ``body_md`` перед записью в БД.

LLM (особенно средней мощности — например, GigaChat-2-Max) иногда возвращает
тело страницы в нечистом виде:

* как одна строка с literal ``\\n`` вместо реальных переводов строк
  — это ломает ``_H2_RE`` в :mod:`wiki_ingest.markdown`, и страница
  превращается в одну гигантскую секцию;
* с хвостом из артефактов tool-call'а — ``"+\\n}``, ``"}``, висящие ``]`` —
  которые остаются в body, но не несут смысла;
* со смешением кириллицы и латиницы в ``[[target|display]]`` (типичный
  пример — ``[[Старик|Dед]]`` с латинской ``D``).

Эта подсистема — узкая прослойка перед :func:`wiki_ingest.markdown.parse_page`
в :func:`wiki_ingest.repository.upsert_page`. Никакого «угадывания» структуры
здесь нет — только консервативные правки очевидно битого ввода с пометками,
что именно было поправлено (для логирования вызывающим кодом).
"""
from __future__ import annotations

import re

# [[target]] или [[target|display]] — копия из markdown.py.
_LINK_RE = re.compile(r"\[\[([^\[\]|\n]+?)(?:\|([^\[\]\n]+?))?]]")

# Строка считается «мусорной», если в ней нет ни одного word-character.
_TRAILING_JUNK_RE = re.compile(r'^[\s"\\}+\],\[]*$')

# Гомоглифы: Latin -> Cyrillic. Только однозначные кейсы.
_LATIN_TO_CYRILLIC: dict[str, str] = {
    "A": "А", "B": "В", "C": "С", "D": "Д", "E": "Е",
    "H": "Н", "K": "К", "M": "М", "O": "О", "P": "Р",
    "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р",
    "x": "х", "y": "у",
}


def _has_cyrillic(s: str) -> bool:
    return any("Ѐ" <= ch <= "ӿ" for ch in s)


def _has_latin(s: str) -> bool:
    return any("a" <= ch.lower() <= "z" for ch in s)


def sanitize_body_md(
    text: str, *, page_slug: str | None = None
) -> tuple[str, list[str]]:
    """Очистить тело страницы перед ``parse_page``.

    Возвращает кортеж ``(cleaned, repairs)``. ``repairs`` — список коротких
    меток применённых правок, удобный для лог-варнинга. Если правок не было,
    ``repairs`` пустой.

    ``page_slug`` сейчас не используется, но передаётся ради будущих правил,
    привязанных к slug'у самой страницы.
    """
    del page_slug  # reserved
    if not text:
        return text or "", []

    repairs: list[str] = []
    out = text

    # 1. Раскрытие escaped-строки: модель вернула тело как JSON-escaped строку
    #    (literal ``\n`` вместо реальных LF).
    literal_nl = out.count("\\n")
    real_nl = out.count("\n")
    if literal_nl and (real_nl == 0 or (literal_nl >= 3 and literal_nl > 2 * real_nl)):
        out = out.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
        repairs.append("unescaped-literal-newlines")

    # 2. Обрезка trailing-JSON-мусора.
    lines = out.split("\n")
    while lines and _TRAILING_JUNK_RE.fullmatch(lines[-1]):
        dropped = lines.pop()
        if dropped.strip():
            repairs.append(f"trimmed-tail:{dropped!r}")
    out = "\n".join(lines).rstrip()

    # 3. Починка mixed-script в ``[[target|display]]``.
    def _repair_link(m: re.Match[str]) -> str:
        target = m.group(1)
        display = m.group(2)
        if display is None:
            return m.group(0)
        target_cyr = _has_cyrillic(target)
        target_lat = _has_latin(target)
        if not target_cyr or target_lat:
            return m.group(0)
        disp_cyr = _has_cyrillic(display)
        disp_lat = _has_latin(display)
        if not (disp_cyr and disp_lat):
            return m.group(0)
        fixed = "".join(_LATIN_TO_CYRILLIC.get(ch, ch) for ch in display)
        if fixed != display and not _has_latin(fixed):
            repairs.append(f"link-homograph:{display!r}->{fixed!r}")
            return f"[[{target}|{fixed}]]"
        repairs.append(f"link-stripped-display:{display!r}")
        return f"[[{target}]]"

    out = _LINK_RE.sub(_repair_link, out)

    return out, repairs


__all__ = ["sanitize_body_md"]
