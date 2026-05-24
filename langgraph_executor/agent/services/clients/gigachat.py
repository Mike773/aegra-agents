"""Заглушка реального клиента GigaChat.

В рабочем репозитории `langgraph_executor` уже есть полноценная реализация
этого модуля. Здесь приводится минимальная версия с тем же публичным API,
чтобы локальные наброски графов импортировались и компилировались.
При интеграции этот файл будет вытеснён реальным.
"""
from __future__ import annotations

import os

from langchain_gigachat import GigaChat


class GigaChatClient:
    def __init__(self) -> None:
        self._llm: GigaChat | None = None

    def get_llm(self) -> GigaChat:
        if self._llm is None:
            self._llm = GigaChat(
                credentials=os.environ.get("GIGACHAT_CREDENTIALS", ""),
                scope=os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
                base_url=os.environ.get("GIGACHAT_BASE_URL") or None,
                verify_ssl_certs=os.environ.get("GIGACHAT_VERIFY_SSL", "False").lower() == "true",
            )
        return self._llm

    def create_embedding(self, text: str) -> list[float]:
        # Реальная реализация — в репозитории проекта (вызов /embeddings GigaChat).
        # Здесь — заглушка фиксированной размерности для локальных проверок.
        raise NotImplementedError("create_embedding реализован в реальном GigaChatClient проекта")


def create_gigachat_client() -> GigaChatClient:
    return GigaChatClient()
