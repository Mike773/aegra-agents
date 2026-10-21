"""Схемы инструментов для GigaChat API v2.

На API v2 (`use_api_v2=True`) `langchain_gigachat` строит схему функции через
`convert_to_openai_function`, и каждое необязательное поле `str | None` уходит
модели как `anyOf: [{type: string}, {type: null}]`. API v2 такого не принимает:
у параметра должен быть ровно один тип. Поэтому на v2 инструменты переводятся в
спецификации заранее, а `anyOf` с `null` сворачивается в единственный тип.
Необязательность от этого не теряется: поле просто не попадает в `required`.

Готовый dict API v2 библиотека передаёт как есть, поэтому `few_shot_examples`
и прочие расширения берутся из её же конвертера, а не собираются здесь заново.
На прежнем API инструменты привязываются без изменений.
"""
from __future__ import annotations

from typing import Any

from langchain_gigachat.utils.function_calling import normalize_tool_for_binding

_NULL = {"type": "null"}


def _flatten(node: Any, path: str) -> Any:
    if isinstance(node, list):
        return [_flatten(item, f"{path}[{i}]") for i, item in enumerate(node)]
    if not isinstance(node, dict):
        return node

    node = dict(node)
    variants = node.get("anyOf")
    if isinstance(variants, list):
        rest = [v for v in variants if v != _NULL]
        if len(rest) != 1:
            # Настоящее объединение типов на v2 не выразить — лучше упасть на
            # сборке, чем молча отдать модели не тот тип.
            raise ValueError(
                f"{path}: anyOf из нескольких типов не поддерживается GigaChat API v2"
            )
        del node["anyOf"]
        node = {**rest[0], **node}
        if len(rest) < len(variants) and node.get("default", ...) is None:
            del node["default"]
    if isinstance(node.get("type"), list):
        types = [t for t in node["type"] if t != "null"]
        if len(types) != 1:
            raise ValueError(
                f"{path}: несколько типов у параметра не поддерживаются GigaChat API v2"
            )
        node["type"] = types[0]
        if node.get("default", ...) is None:
            del node["default"]
    return {k: _flatten(v, f"{path}.{k}") for k, v in node.items()}


def uses_api_v2(llm: Any) -> bool:
    return getattr(llm, "use_api_v2", False) is True


def v2_tool_spec(tool: Any) -> dict[str, Any]:
    """Инструмент → готовая спецификация API v2 с одним типом на параметр."""
    spec = normalize_tool_for_binding(tool, use_api_v2=True)
    function = spec["function"]
    name = function.get("name", "?")
    for key in ("parameters", "return_parameters"):
        if isinstance(function.get(key), dict):
            function[key] = _flatten(function[key], f"{name}.{key}")
    return spec


def bind_tools(llm: Any, tools: list[Any]) -> Any:
    """`llm.bind_tools` с поправкой схем под API v2, если клиент на нём."""
    if not uses_api_v2(llm):
        return llm.bind_tools(tools)
    return llm.bind_tools([v2_tool_spec(t) for t in tools])


__all__ = ["bind_tools", "uses_api_v2", "v2_tool_spec"]
