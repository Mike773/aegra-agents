"""Классическая стратегия стадии 1: tool-loop с защитой от зацикливания.

Использует ``langchain.agents.create_agent``. У GigaChat запрос С ФУНКЦИЯМИ
ограничен ~4096 токенами, поэтому выдача каждого инструмента усекается, число
шагов цикла лимитировано, а повторные вызовы с теми же аргументами заменяются
подсказкой «не повторяй». Стадия 2 (синтез) — в agent_base.synthesize_answer.

Все мутирующие счётчики (повторы вызовов, бюджет) живут в локальном dict-state,
а не в модульных переменных: иначе concurrent-вызовы графа портили бы друг
другу счётчики.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError

from .agent_base import _text, format_facts
from .prompts import SYSTEM_PROMPT_RULES

# Цикл сбора (стадия 1) тащит на каждом шаге всю историю вызовов инструментов.
# Чтобы контекст не перерос окно модели, выдача каждого инструмента усекается,
# а число шагов цикла ограничено.
_TOOL_OUTPUT_CAP = 6000
_RECURSION_LIMIT = 50

# Мягкий потолок числа фактических вызовов инструментов за один сбор. Достигнув
# его, агент на любой следующий вызов получает подсказку завершиться — это
# переводит «упор в жёсткий лимит рекурсии» в управляемую раннюю остановку.
_TOOL_CALL_BUDGET = 18

# Сколько раз подряд агент может игнорировать заглушку (повтор/бюджет), прежде
# чем стадия 1 будет принудительно завершена.
_MAX_REPEAT_BLOCKS = 3

_REPEAT_NOTICE = (
    "Этот инструмент уже вызывался с такими же аргументами — его результат "
    "есть выше. НЕ повторяй вызов: используй уже собранные данные или заверши "
    "сбор."
)
_BUDGET_NOTICE = (
    "Достигнут предел числа обращений к инструментам — собранных данных "
    "достаточно. НЕ вызывай больше инструменты: заверши сбор и дай ответ по уже "
    "полученным данным."
)


@dataclass
class _RunState:
    """Per-run счётчики антизацикливания. Хранятся в замыкании guard'ов."""

    seen: set[str] = field(default_factory=set)
    count: int = 0


def _guard_tools(tools: list[Any], state: _RunState) -> None:
    """Оборачивает инструменты: усечение выдачи + защита от повторных вызовов.

    Каждый вызов сверяется со state — повтор с теми же аргументами или превышение
    бюджета возвращают заглушку вместо реального запуска. Усечение защищает от
    того, что одна «жирная» выдача переполнит контекстное окно.
    """
    for tool in tools:
        if getattr(tool, "_guarded", False):
            continue
        original = tool.func
        if original is None:
            continue

        def _wrap(orig: Any, name: str) -> Any:
            def guarded(*args: Any, **kwargs: Any) -> Any:
                key = f"{name}|{args!r}|{sorted(kwargs.items())!r}"
                if key in state.seen:
                    return _REPEAT_NOTICE
                if state.count >= _TOOL_CALL_BUDGET:
                    return _BUDGET_NOTICE
                state.seen.add(key)
                state.count += 1
                out = orig(*args, **kwargs)
                if isinstance(out, str) and len(out) > _TOOL_OUTPUT_CAP:
                    return (
                        out[:_TOOL_OUTPUT_CAP]
                        + f"\n…[длинная выдача обрезана: показаны первые "
                        f"{_TOOL_OUTPUT_CAP} символов из {len(out)}. Это НЕ "
                        "отсутствие данных — показанных строк (они идут в "
                        "порядке значимости) достаточно для ответа; при "
                        "необходимости уточни запрос фильтрами.]"
                    )
                return out

            return guarded

        tool.func = _wrap(original, tool.name)
        tool._guarded = True


def compose_system_prompt(overview: dict[str, Any]) -> str:
    """Системный промпт стадии 1 + динамический «Состав датасета»."""
    return SYSTEM_PROMPT_RULES + "\n\n" + format_facts(overview)


@dataclass
class ClassicStrategy:
    """Один прогон стадии 1: build() готовит агента, run() выполняет цикл сбора."""

    state: _RunState = field(default_factory=_RunState)

    def build(
        self, model: Any, tools: list[Any], overview: dict[str, Any]
    ) -> Any:
        _guard_tools(tools, self.state)
        system_prompt = compose_system_prompt(overview)
        agent = create_agent(model=model, tools=tools, system_prompt=system_prompt)
        return agent.with_config({"recursion_limit": _RECURSION_LIMIT})

    def run(self, agent: Any, messages: list[Any]) -> tuple[list[Any], bool]:
        """Возвращает (накопленные сообщения, завершилось_штатно).

        Цикл сбора прогоняется через stream, чтобы при упоре в лимит рекурсии
        сохранить уже накопленные сообщения: ответ синтезируется даже из частичного
        транскрипта, а не теряется вместе с GraphRecursionError.
        """
        self.state.seen.clear()
        self.state.count = 0
        last_messages: list[Any] = list(messages)
        completed = True
        try:
            for snapshot in agent.stream({"messages": messages}, stream_mode="values"):
                last_messages = snapshot.get("messages", last_messages)
                stop_notices = sum(
                    1
                    for m in last_messages
                    if isinstance(m, ToolMessage)
                    and (_REPEAT_NOTICE in _text(m) or _BUDGET_NOTICE in _text(m))
                )
                if stop_notices >= _MAX_REPEAT_BLOCKS:
                    # Модель зациклилась на повторах либо исчерпала бюджет
                    # вызовов и не реагирует на заглушку. Нужные данные уже
                    # собраны более ранними вызовами — обрываем пустой цикл и
                    # идём к синтезу (это не потеря данных, поэтому completed=True).
                    break
        except GraphRecursionError:
            completed = False
        return last_messages, completed
