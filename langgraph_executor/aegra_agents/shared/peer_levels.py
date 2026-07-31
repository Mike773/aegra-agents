"""Приоритет уровней peer-групп — конфиг развёрнутого сервиса.

Наборы уровней инстанс-специфичны: коды приходят в данных полем ``level``, их
человеческие названия — полем ``level_name`` там же. Конфиг задаёт ТОЛЬКО
порядок уровней, от узкой группы к широкой: первый становится референсным для
расчётов (см. ``analytics._level_order``). Названия здесь не хранятся — они
приходят вместе с данными.

Формат переменной ``PEER_LEVELS`` — коды через ``|`` или JSON-массив::

    PEER_LEVELS=OFFICE|TERR|ORG
    PEER_LEVELS=["OFFICE", "TERR", "ORG"]

Старая форма ``КОД:название`` тоже разбирается — берётся часть до двоеточия,
название игнорируется. Больше ``MAX_PEER_LEVELS`` записей конфиг не принимает —
лишние отбрасываются.

По умолчанию порядок ПУСТ: захардкоженного набора уровней здесь нет и быть не
должно. Пустой конфиг = уровни упорядочиваются эвристикой по размеру группы.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

PEER_LEVELS_ENV = "PEER_LEVELS"
# Потолок из бизнес-требования: уровней сравнения не больше трёх.
MAX_PEER_LEVELS = 3


def _code(item: object) -> str:
    """Код уровня из элемента конфига: строка, ``КОД:название`` или объект."""
    if isinstance(item, dict):
        return str(item.get("code") or item.get("level") or "").strip()
    return str(item or "").split(":", 1)[0].strip()


def parse_peer_levels(raw: str | None) -> tuple[str, ...]:
    """Разбирает значение конфига в упорядоченные коды уровней.

    Битый или пустой вход → пустой кортеж: приоритет — не критичный конфиг, без
    него порядок определяет эвристика, поэтому ошибка формата деградирует, а не
    роняет загрузку графа.
    """
    text = (raw or "").strip()
    if not text:
        return ()
    if text.startswith("["):
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            items = None
        codes = [_code(i) for i in items] if isinstance(items, list) else []
    else:
        codes = [_code(chunk) for chunk in text.split("|")]

    codes = [c for c in codes if c]
    if not codes:
        logger.warning(
            "%s задан, но не разобран — порядок уровней определит эвристика: %.100s",
            PEER_LEVELS_ENV, text,
        )
        return ()
    if len(codes) > MAX_PEER_LEVELS:
        logger.warning(
            "%s: уровней %d, берём первые %d — отброшены %s",
            PEER_LEVELS_ENV, len(codes), MAX_PEER_LEVELS,
            ", ".join(codes[MAX_PEER_LEVELS:]),
        )
        codes = codes[:MAX_PEER_LEVELS]
    return tuple(codes)


_levels: tuple[str, ...] | None = None


def get_peer_levels() -> tuple[str, ...]:
    """Коды уровней в порядке конфига (кэшируется на процесс)."""
    global _levels
    if _levels is None:
        _levels = parse_peer_levels(os.environ.get(PEER_LEVELS_ENV))
    return _levels


def reset_cache() -> None:
    """Сбрасывает кэш — нужен тестам, меняющим окружение."""
    global _levels
    _levels = None


def level_order() -> list[str]:
    """Коды уровней от узкой группы к широкой."""
    return list(get_peer_levels())


__all__ = [
    "MAX_PEER_LEVELS",
    "PEER_LEVELS_ENV",
    "get_peer_levels",
    "level_order",
    "parse_peer_levels",
    "reset_cache",
]
