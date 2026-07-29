"""Справочник уровней peer-групп — конфиг развёрнутого сервиса.

Уровни приходят в данных кодами (``ORG``/``TERR``/``OFFICE`` — лишь пример) и
инстанс-специфичны: набор и иерархия свои в каждой компании. Поэтому справочник
задаётся окружением, а не запросом: один инстанс = одна компания.

Формат переменной ``PEER_LEVELS`` — компактная строка или JSON-массив::

    PEER_LEVELS=ORG:вся организация|TERR:территориальный банк|OFFICE:офис
    PEER_LEVELS=[{"code": "ORG", "title": "вся организация"}, ...]

Порядок — от УЗКОЙ группы к широкой: первый уровень становится референсным для
расчётов (см. ``analytics._level_order``). Больше ``MAX_PEER_LEVELS`` записей
конфиг не принимает — лишние отбрасываются.

По умолчанию справочник ПУСТ: захардкоженного набора уровней здесь нет и быть не
должно. Пустой справочник = прежнее поведение, коды показываются как есть.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PEER_LEVELS_ENV = "PEER_LEVELS"
# Потолок из бизнес-требования: уровней сравнения не больше трёх.
MAX_PEER_LEVELS = 3


@dataclass(frozen=True)
class PeerLevel:
    """Уровень peer-группы: код как в данных + человеческое название."""

    code: str
    title: str


def _from_json(raw: str) -> list[PeerLevel]:
    """JSON-массив [{"code", "title"}] — форма для конфиг-менеджеров."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[PeerLevel] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or item.get("level") or "").strip()
        title = str(item.get("title") or item.get("name") or "").strip()
        if code and title:
            out.append(PeerLevel(code=code, title=title))
    return out


def _from_compact(raw: str) -> list[PeerLevel]:
    """Компактная строка ``КОД:название|КОД:название`` — форма для .env."""
    out: list[PeerLevel] = []
    for chunk in raw.split("|"):
        code, sep, title = chunk.partition(":")
        code, title = code.strip(), title.strip()
        if sep and code and title:
            out.append(PeerLevel(code=code, title=title))
    return out


def parse_peer_levels(raw: str | None) -> tuple[PeerLevel, ...]:
    """Разбирает значение конфига. Битый/пустой вход → пустой кортеж.

    Справочник — не критичный для работы конфиг: без него аналитик отдаёт коды
    уровней как есть. Поэтому любая ошибка формата деградирует до «справочника
    нет», а не роняет загрузку графа.
    """
    text = (raw or "").strip()
    if not text:
        return ()
    levels = _from_json(text) if text.startswith("[") else _from_compact(text)
    if not levels:
        logger.warning(
            "%s задан, но не разобран — уровни будут показаны кодами: %.100s",
            PEER_LEVELS_ENV, text,
        )
        return ()
    if len(levels) > MAX_PEER_LEVELS:
        logger.warning(
            "%s: уровней %d, берём первые %d — %s",
            PEER_LEVELS_ENV, len(levels), MAX_PEER_LEVELS,
            ", ".join(lv.code for lv in levels[MAX_PEER_LEVELS:]),
        )
        levels = levels[:MAX_PEER_LEVELS]
    return tuple(levels)


_levels: tuple[PeerLevel, ...] | None = None


def get_peer_levels() -> tuple[PeerLevel, ...]:
    """Справочник уровней из окружения (кэшируется на процесс)."""
    global _levels
    if _levels is None:
        _levels = parse_peer_levels(os.environ.get(PEER_LEVELS_ENV))
    return _levels


def reset_cache() -> None:
    """Сбрасывает кэш — нужен тестам, меняющим окружение."""
    global _levels
    _levels = None


def level_title(code: object) -> str:
    """Человеческое название уровня; уровень вне справочника отдаём как есть.

    Данные могут содержать уровень, которого нет в конфиге, — терять его нельзя,
    поэтому фолбэк на сам код (это честнее, чем выкинуть строку сравнения).
    """
    text = str(code or "").strip()
    if not text:
        return ""
    for lv in get_peer_levels():
        if lv.code.casefold() == text.casefold():
            return lv.title
    return text


def level_order() -> list[str]:
    """Коды уровней в порядке конфига — от узкой группы к широкой."""
    return [lv.code for lv in get_peer_levels()]


__all__ = [
    "MAX_PEER_LEVELS",
    "PEER_LEVELS_ENV",
    "PeerLevel",
    "get_peer_levels",
    "level_order",
    "level_title",
    "parse_peer_levels",
    "reset_cache",
]
