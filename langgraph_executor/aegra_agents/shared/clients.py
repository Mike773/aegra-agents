"""GigaChat-клиент для агентов.

Читает параметры подключения из переменных окружения (`.env` подгружается
автоматически), отдаёт готовый `langchain_gigachat.GigaChat`.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_gigachat import GigaChat, GigaChatEmbeddings

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break
    _ENV_LOADED = True


def _clean(value: str | None) -> str:
    """Срезает inline-комментарий и пробелы — в `.env` встречаются `KEY=val   # note`."""
    if value is None:
        return ""
    return value.split("#", 1)[0].strip()


def _truthy(value: str | None) -> bool:
    return _clean(value).lower() in {"1", "true", "yes", "on"}


class GigaChatClient:
    def __init__(self) -> None:
        _ensure_env_loaded()
        self._llm: GigaChat | None = None

    def get_llm(self) -> GigaChat:
        if self._llm is None:
            credentials = _clean(os.environ.get("GIGACHAT_CREDENTIALS"))
            if not credentials:
                raise RuntimeError(
                    "GIGACHAT_CREDENTIALS не задан — проверь .env рядом с проектом."
                )
            kwargs: dict[str, object] = {
                "credentials": credentials,
                "scope": _clean(os.environ.get("GIGACHAT_SCOPE")) or "GIGACHAT_API_PERS",
                "verify_ssl_certs": _truthy(os.environ.get("GIGACHAT_VERIFY_SSL")),
            }
            base_url = _clean(os.environ.get("GIGACHAT_BASE_URL"))
            if base_url:
                kwargs["base_url"] = base_url
            model = _clean(os.environ.get("LLM_MODEL"))
            if model:
                kwargs["model"] = model
            self._llm = GigaChat(**kwargs)
        return self._llm

    def create_embedding(self, text: str) -> list[float]:
        raise NotImplementedError(
            "create_embedding пока не реализован для локального клиента"
        )


def create_gigachat_client() -> GigaChatClient:
    return GigaChatClient()


def create_gigachat_embeddings() -> GigaChatEmbeddings:
    """Фабрика эмбеддингов GigaChat.

    Использует те же credentials/scope, что и чат-клиент. Модель берётся из
    `EMBEDDINGS_MODEL` (по умолчанию `Embeddings`).
    """
    _ensure_env_loaded()
    credentials = _clean(os.environ.get("GIGACHAT_CREDENTIALS"))
    if not credentials:
        raise RuntimeError(
            "GIGACHAT_CREDENTIALS не задан — проверь .env рядом с проектом."
        )
    kwargs: dict[str, object] = {
        "model": _clean(os.environ.get("EMBEDDINGS_MODEL")) or "Embeddings",
        "credentials": credentials,
        "scope": _clean(os.environ.get("GIGACHAT_SCOPE")) or "GIGACHAT_API_PERS",
        "verify_ssl_certs": _truthy(os.environ.get("GIGACHAT_VERIFY_SSL")),
    }
    base_url = _clean(os.environ.get("GIGACHAT_BASE_URL"))
    if base_url:
        kwargs["base_url"] = base_url
    auth_url = _clean(os.environ.get("GIGACHAT_AUTH_URL"))
    if auth_url:
        kwargs["auth_url"] = auth_url
    return GigaChatEmbeddings(**kwargs)


__all__ = [
    "GigaChatClient",
    "create_gigachat_client",
    "create_gigachat_embeddings",
]
