"""Мок GigaChat для нагрузочного воспроизведения в обход квоты (429).

Зачем: реальный аккаунт GigaChat душит 429 уже на ~6 параллельных запросах
(см. docs/diagnose_concurrency.md), поэтому честный прод-сценарий 30×30 на живом
LLM локально не снять. Этот мок убирает сеть и квоту, но сохраняет ТОТ ЖЕ
ресурсный профиль: блокирующая «генерация» с задержкой (`MOCK_LLM_LATENCY`,
дефолт 2с) живёт в потоке (как llm.invoke в to_thread / sync-узле) — значит
видно реальную нагрузку на thread-pool и БД-пул.

Реализация — подкласс GigaChat с переопределённым ``_generate``/``_agenerate``:
весь langchain-обвес (``bind_tools``, ``create_agent`` в tool-loop json_analyzer)
работает как с настоящей моделью, перехватывается только сам вызов модели.

Включение (перед импортом графов!):
    import langgraph_executor.mock_llm as mock; mock.install()
или через переменную окружения в раннере (см. scripts/loadtest_inprocess.py).
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_gigachat import GigaChat, GigaChatEmbeddings

# Канонические тексты мока — на них ассертятся e2e-тесты.
MOCK_DESCRIBE_NARRATIVE = (
    "Mock-описание хода анализа: запросил обзор датасета (schema_overview), "
    "по его результату оценил метрики и сформулировал вывод."
)
MOCK_GATHER_REASONING = "Начну с обзора состава датасета, чтобы понять, какие метрики есть."


def _latency() -> float:
    return float(os.environ.get("MOCK_LLM_LATENCY") or "2.0")


def _emb_dim() -> int:
    return int(os.environ.get("MOCK_EMB_DIM") or "1024")


def _decide_content(messages: List[BaseMessage]) -> str:
    """Возвращает контент, который провёдёт граф по тяжёлому пути.

    Сопоставляем системный промпт с известными константами оркестратора, чтобы
    роутер ушёл в analytics, генератор wiki-запросов отдал валидный JSON-массив,
    а tool-loop json_analyzer завершился с первого шага (ответ без tool_calls).
    """
    # ленивый импорт — модуль грузится до сборки графов
    try:
        from langgraph_executor.aegra_agents.analytic_orchestrator_v2.prompts import (
            CONFIRM_PROMPT,
            DESCRIBE_ANSWER_PROMPT,
            ROUTER_PROMPT,
            WIKI_QUERIES_PROMPT,
        )
    except Exception:  # noqa: BLE001
        ROUTER_PROMPT = CONFIRM_PROMPT = WIKI_QUERIES_PROMPT = "\0"
        DESCRIBE_ANSWER_PROMPT = "\0"
    try:
        from langgraph_executor.aegra_agents.json_analyzer_v3.prompts import (
            METRIC_KINDS_SYSTEM_PROMPT,
            RELATIONS_SYSTEM_PROMPT,
        )
    except Exception:  # noqa: BLE001
        METRIC_KINDS_SYSTEM_PROMPT = RELATIONS_SYSTEM_PROMPT = "\0"

    blob = "\n".join(
        m.content for m in messages
        if isinstance(getattr(m, "content", None), str)
    )

    def _is(prompt: str) -> bool:
        head = (prompt or "")[:48].strip()
        return bool(head) and head in blob

    if _is(ROUTER_PROMPT):
        return "analytics"          # роутер → тяжёлый путь
    if _is(CONFIRM_PROMPT):
        return "confirm"
    if _is(WIKI_QUERIES_PROMPT):
        return '["что такое показатель", "методика расчёта метрики"]'
    if _is(DESCRIBE_ANSWER_PROMPT):
        return MOCK_DESCRIBE_NARRATIVE
    # LLM-кэши json_analyzer ждут строгий JSON-массив.
    if _is(RELATIONS_SYSTEM_PROMPT) or _is(METRIC_KINDS_SYSTEM_PROMPT):
        return "[]"
    # initial_analysis / form_insights / tool-loop synth — свободный текст без tool_calls
    return "Содержательный разбор показателей сотрудника (mock)."


def _decide_message(messages: List[BaseMessage]) -> AIMessage:
    """AIMessage мока: на стадии сбора json_analyzer один раз зовёт инструмент.

    Первый вызов tool-loop (в сообщениях ещё нет ToolMessage) возвращает tool_call
    schema_overview с текстом-мотивацией — так e2e проходит реальный tool-loop,
    получает tool_steps с reasoning и шаг tool_call в трассе. Второй вызов (после
    ToolMessage) отдаёт обычный текст — loop завершается.
    """
    try:
        from langgraph_executor.aegra_agents.json_analyzer_v3.prompts import (
            SYSTEM_PROMPT_RULES,
        )
    except Exception:  # noqa: BLE001
        SYSTEM_PROMPT_RULES = "\0"
    head = SYSTEM_PROMPT_RULES[:48].strip()
    is_gather = bool(head) and any(
        isinstance(getattr(m, "content", None), str) and head in m.content
        for m in messages
    )
    has_tool_result = any(isinstance(m, ToolMessage) for m in messages)
    if is_gather and not has_tool_result:
        return AIMessage(
            content=MOCK_GATHER_REASONING,
            tool_calls=[{
                "name": "schema_overview", "args": {},
                "id": "mock-call-1", "type": "tool_call",
            }],
        )
    return AIMessage(content=_decide_content(messages))


def _result(messages: List[BaseMessage]) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=_decide_message(messages))])


class MockGigaChat(GigaChat):
    """GigaChat без сети: спит MOCK_LLM_LATENCY и отдаёт канонический ответ."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        time.sleep(_latency())            # блокирует ИМЕННО поток — как реальный invoke
        return _result(messages)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        await asyncio.sleep(_latency())
        return _result(messages)


class MockEmbeddings(GigaChatEmbeddings):
    """Эмбеддинги без сети: детерминированный ненулевой вектор фиксированной длины."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * _emb_dim() for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * _emb_dim()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


def _build_llm() -> MockGigaChat:
    llm = MockGigaChat(credentials="mock", verify_ssl_certs=False)
    try:
        from langgraph_executor.diag import wrap_llm_timing
        return wrap_llm_timing(llm)
    except Exception:  # noqa: BLE001
        return llm


# Минимальный НЕпустой датасет — чтобы json_analyzer собрал непустой store и
# реально запустил вложенный sync agent.stream() (tool-loop). Форма — под
# json_analyzer_v2/loader.py:load_dataset_obj (me + дерево metrics).
_MOCK_DATASET = {
    "me": {
        "tabnum": "100500", "fio": "Иванов И.И.", "post": "Аналитик", "depart": "Отдел",
        "metrics": [
            {
                "id": 1, "metric_name": "Выручка", "metric_description": "Выручка за период",
                "metric_type": "money", "measure_type": "руб", "date": "2026-05",
                "calc_period": "month", "fact": 100, "plan": 120, "benchmark": 110,
                "influent_percent": 50,
                "child_metrics": [
                    {"id": 2, "metric_name": "Сделки", "metric_description": "Кол-во сделок",
                     "metric_type": "count", "fact": 10, "plan": 12},
                ],
            },
        ],
    },
    "employees": [],
}


def _install_full_dataset() -> None:
    """Подменяет stub-датасет на непустой, чтобы раскрылся тяжёлый путь."""
    from langgraph_executor.aegra_agents.shared import agent_dataset

    def build_json_output(self):  # noqa: ANN001
        return dict(_MOCK_DATASET)

    agent_dataset.GetBatchAgentDatasetByFiltersComponent.build_json_output = (  # type: ignore[assignment]
        build_json_output
    )


def install(full_path: bool | None = None) -> None:
    """Подменяет фабрики в shared.clients на мок. Вызывать ДО импорта графов.

    full_path: если True (или env ``MOCK_FULL_PATH``), также подменяет stub-датасет
    на непустой — тогда json_analyzer доходит до вложенного tool-loop.
    """
    from langgraph_executor.aegra_agents.shared import clients

    class _MockClient:
        def get_llm(self):
            return _build_llm()

    clients.create_gigachat_client = lambda: _MockClient()  # type: ignore[assignment]
    clients.create_gigachat_embeddings = lambda: MockEmbeddings(  # type: ignore[assignment]
        credentials="mock", verify_ssl_certs=False
    )

    if full_path is None:
        full_path = (os.environ.get("MOCK_FULL_PATH") or "").strip().lower() in {
            "1", "true", "yes", "on",
        }
    if full_path:
        _install_full_dataset()


__all__ = [
    "install",
    "MockGigaChat",
    "MockEmbeddings",
    "MOCK_DESCRIBE_NARRATIVE",
    "MOCK_GATHER_REASONING",
]
