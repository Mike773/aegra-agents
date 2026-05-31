"""Сходство строк без прямой зависимости от stdlib ``difflib``.

В прод-окружении (PyInstaller-бандл) ``difflib`` может быть недоступен, зато есть
``cydifflib`` — drop-in C-замена с тем же API. Берём ``SequenceMatcher`` из
``cydifflib``, а локально (где может стоять только stdlib) откатываемся на
``difflib``. Семантика ``.ratio()`` идентична.
"""
from __future__ import annotations

try:
    from cydifflib import SequenceMatcher
except ImportError:  # локально без cydifflib — обычный stdlib difflib
    from difflib import SequenceMatcher


def similarity_ratio(a: str, b: str) -> float:
    """Сходство строк в диапазоне [0, 1] (как ``SequenceMatcher.ratio()``)."""
    return SequenceMatcher(None, a, b).ratio()


__all__ = ["SequenceMatcher", "similarity_ratio"]
