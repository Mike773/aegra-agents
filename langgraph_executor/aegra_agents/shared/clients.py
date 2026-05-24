"""Тонкая обёртка над клиентом GigaChat проекта.

Единственная точка импорта для всех графов — если путь к настоящему
клиенту в проекте поменяется, правка делается здесь, а не в каждом графе.
"""
from langgraph_executor.agent.services.clients.gigachat import (
    GigaChatClient,
    create_gigachat_client,
)

__all__ = ["GigaChatClient", "create_gigachat_client"]
