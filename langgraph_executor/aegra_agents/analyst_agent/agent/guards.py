"""Защита цикла инструментов от зацикливания и переполнения контекста.

Все счётчики — в объекте на запуск (не в модульных переменных): граф вызывают
конкурентно, и общие счётчики портили бы друг друга.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Кап выдачи одного инструмента. Страховка от аномально «жирного» результата:
# цикл тащит всю историю вызовов на каждом шаге.
DEFAULT_OUTPUT_CAP = 20000
# Мягкий потолок числа вызовов за один ответ. Ограничитель — латентность и
# rate-limit модели, а не размер контекста.
DEFAULT_BUDGET = 18
# Сколько раз подряд модель может игнорировать заглушку, прежде чем закончим.
MAX_REPEAT_BLOCKS = 3

REPEAT_NOTICE = (
    "Этот вызов с теми же аргументами уже был, повторный результат не даётся. "
    "Возьми данные из предыдущего ответа, поменяй параметры или переходи к ответу."
)
BUDGET_NOTICE = (
    "Бюджет вызовов инструментов исчерпан. Заверши анализ и напиши итоговый "
    "ответ по уже собранным данным."
)
TRUNCATION_NOTICE = (
    "\n\n[выдача обрезана по длине — это НЕ отсутствие данных; уточни запрос, "
    "если нужны остальные строки]"
)


@dataclass
class RunGuards:
    """Счётчики одного запуска цикла."""

    budget: int = DEFAULT_BUDGET
    output_cap: int = DEFAULT_OUTPUT_CAP
    max_rounds: int | None = None
    calls_made: int = 0
    repeat_blocks: int = 0
    _seen: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.max_rounds is None:
            self.max_rounds = self.budget + 4

    @staticmethod
    def key(name: str, args: dict[str, Any]) -> str:
        """Ключ вызова. Пустые аргументы отбрасываются: «параметра нет» и
        «параметр пустой» — один и тот же вызов, иначе модель, зовущая
        инструмент без обязательного параметра, крутит его по кругу."""
        meaningful = {k: v for k, v in (args or {}).items() if v is not None}
        return f"{name}|{sorted(meaningful.items(), key=lambda kv: kv[0])!r}"

    def is_repeat(self, name: str, args: dict[str, Any]) -> bool:
        return self.key(name, args) in self._seen

    def over_budget(self) -> bool:
        return self.calls_made >= self.budget

    def register(self, name: str, args: dict[str, Any]) -> None:
        self._seen.add(self.key(name, args))
        self.calls_made += 1

    def note_block(self) -> bool:
        """Заглушка выдана. True — модель уперлась и пора заканчивать."""
        self.repeat_blocks += 1
        return self.repeat_blocks >= MAX_REPEAT_BLOCKS

    def clip(self, text: str) -> str:
        if len(text) <= self.output_cap:
            return text
        return text[: self.output_cap] + TRUNCATION_NOTICE


def blank_to_none(value: Any) -> Any:
    """Пустое значение любого вида → None.

    Модели присылают вместо опускания аргумента пустую строку, а иногда пустой
    объект или список (замечено у GigaChat на живом прогоне). Без нормализации
    такой аргумент уходит в резолв как есть и молча ничего не находит.
    Ноль и False — валидные значения, их не трогаем.
    """
    if isinstance(value, str) and value.strip() == "":
        return None
    if isinstance(value, (dict, list, tuple, set)) and not value:
        return None
    return value


def clean_args(args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    return {k: blank_to_none(v) for k, v in args.items()}


__all__ = [
    "BUDGET_NOTICE",
    "DEFAULT_BUDGET",
    "DEFAULT_OUTPUT_CAP",
    "MAX_REPEAT_BLOCKS",
    "REPEAT_NOTICE",
    "TRUNCATION_NOTICE",
    "RunGuards",
    "blank_to_none",
    "clean_args",
]
