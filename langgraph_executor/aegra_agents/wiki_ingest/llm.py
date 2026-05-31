"""LLM-обёртка для структурированного вывода поверх GigaChat.

Перенос ``LLMClient.call_json`` из easyRag (tool-binding через
``bind_tools(tool_choice=...)``), но клиент берётся из общего
``shared.clients.create_gigachat_client()`` — провайдер только GigaChat, без
OpenAI/mock. Модель «вызывает» tool с заданной JSON-схемой, мы возвращаем его
аргументы — это надёжнее, чем парсить свободный JSON из текста.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..shared.clients import create_gigachat_client
from .config import get_settings


class LLMClient:
    def __init__(
        self,
        chat: Any = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self._chat = chat if chat is not None else create_gigachat_client().get_llm()
        # Выравниваем параметры с easyRag прямо на инстансе клиента (он свой у
        # wiki_ingest — create_gigachat_client() отдаёт отдельный GigaChat, так
        # что json_analyzer/orchestrator не затрагиваются). Без max_tokens
        # GigaChat усекает длинные tool-ответы (save_entities/merge) → меньше
        # извлечённых сущностей.
        cfg = get_settings()
        mt = max_tokens if max_tokens is not None else cfg.llm_max_tokens
        tmp = temperature if temperature is not None else cfg.llm_temperature
        if mt is not None:
            self._chat.max_tokens = mt
        if tmp is not None:
            self._chat.temperature = tmp

    async def call_json(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Запросить у модели структурированный ответ по JSON-схеме."""
        tool = _schema_to_tool(tool_name, tool_description, input_schema)
        bound = self._chat.bind_tools([tool], tool_choice=tool_name)
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        response = await bound.ainvoke(messages)

        for call in getattr(response, "tool_calls", None) or []:
            if call.get("name") == tool_name:
                return dict(call.get("args") or {})

        # Фолбэк: GigaChat иногда кладёт аргументы в
        # additional_kwargs.function_call.arguments.
        extra = getattr(response, "additional_kwargs", {}) or {}
        fn = extra.get("function_call")
        if fn and fn.get("name") == tool_name:
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    return json.loads(args)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"LLM вернул не-JSON аргументы tool {tool_name}: {args!r}"
                    ) from exc
            if isinstance(args, dict):
                return dict(args)

        raise RuntimeError(f"LLM не вызвал tool {tool_name}: {response!r}")


def _schema_to_tool(name: str, description: str, input_schema: dict[str, Any]) -> dict[str, Any]:
    """Собрать tool в OpenAI-формате, отдавая JSON-схему без потерь.

    ``bind_tools`` принимает dict в OpenAI-формате напрямую — надёжнее, чем
    строить pydantic-носитель схемы (``convert_to_openai_tool`` иначе читает
    поля из ``model_fields`` и отдаёт пустой ``parameters``).
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": input_schema,
        },
    }


_default_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


__all__ = ["LLMClient", "get_llm"]
