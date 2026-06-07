from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Intent = Literal[
    "analytics",
    "wiki",
    "chat",
    "done",
    "assignments",
    "assignments_select",
]


class TraceStep(TypedDict, total=False):
    """Один шаг рассуждения для сквозного лога (Блок A ТЗ).

    stage — узел конвейера: 'route' | 'easyrag' | 'json_analyzer' |
            'assignments' | 'respond' | 'initial';
    kind  — тип шага: 'intent' | 'kb_hit' | 'tool_call' | 'derived_metric' |
            'hypothesis' | 'decision' | 'error';
    summary — человекочитаемая строка одной фразой;
    detail  — машиночитаемые детали (имя инструмента, аргументы, similarity,
              имена метрик, числа и т.п.).
    """

    stage: str
    kind: str
    summary: str
    detail: dict


class OrchestratorOutput(TypedDict, total=False):
    """Публичная (возвращаемая) схема графа — output_schema в build_graph.

    Содержит ВСЕ каналы стейта, КРОМЕ внутренних, которые не нужно отдавать
    наружу в run/stream (см. OrchestratorState ниже). Новое возвращаемое поле
    добавляй сюда; внутреннее-невозвращаемое — в OrchestratorState.
    """

    messages: Annotated[list[AnyMessage], add_messages]

    boss_tabnum: str
    employee_tabnum: str
    position: str | None
    direction_key: str | None

    # Первое сообщение — заранее подготовленный вопрос-брифинг, возможно
    # содержащий желаемый формат ответа. Сохраняется на первом ходе и затем
    # подмешивается в системный контекст респондера на всех последующих ходах,
    # чтобы формат держался по всему диалогу. Пользовательские реплики приходят
    # отдельными ходами и брифинг не перезаписывают.
    briefing: str | None

    # Сквозной лог шагов рассуждения текущего хода (Блок A ТЗ). Жизненный цикл —
    # строго per-turn: первый узел хода (load_data/route) перезаписывает список,
    # downstream-узлы дополняют его явной конкатенацией (не reducer — иначе
    # трасса копилась бы между ходами).
    reasoning_trace: list[TraceStep]

    metrics_error: str | None
    loaded: bool

    intent: Intent | None

    # Результат вызова easyrag-подграфа (свежий, под последний вопрос пользователя).
    easyrag_query: str | None
    easyrag_snippets: list[dict]
    easyrag_error: str | None
    # Релевантные запросу страницы-заглушки (type='stub', пустые) — заполняется,
    # только когда обычная выборка пуста: сущность заведена, но не наполнена.
    easyrag_stub_pages: list[dict]

    # Опорный (широкий) первичный разбор метрик — результат json_analyzer по
    # брифингу на ПЕРВОМ ходе. Ставится один раз и НЕ перезаписывается узкими
    # analytics-ходами (в отличие от analytics_answer ниже). Респондер всегда
    # держит его как стабильную опору, чтобы свежий узкий ответ не подменял собой
    # «всю картину» метрик на последующих ходах.
    metrics_summary: str | None

    # Результат вызова json_analyzer-подграфа под последний аналитический вопрос.
    # Sticky: не сбрасывается между циклами, пока следующий analytics-цикл не
    # перезапишет — респондер использует самый свежий ответ.
    analytics_question: str | None
    analytics_answer: str | None
    analytics_error: str | None

    # Поручения сотруднику в формате инсайтов сервиса: каждый элемент —
    # {type, metric_id, metric_name, text}, type ∈
    # {main_problem, problem, norm, achievement} (см. _parse_insights_json).
    # candidate_assignments — последняя извлечённая «корзина» кандидатов.
    # pending_assignments — кандидаты, по которым прямо сейчас ждём выбор пользователя.
    # Пока pending не пуст, роутер форсит intent="assignments_select".
    # selected_assignments — выбранное к фиксации в текущей итерации.
    # last_committed_assignments — что реально ушло в сервис поручений в прошлый раз.
    candidate_assignments: list[dict]
    pending_assignments: list[dict]
    selected_assignments: list[dict]
    last_committed_assignments: list[dict]


class OrchestratorState(OrchestratorOutput, total=False):
    """Полный стейт графа (output-схема + внутренние каналы).

    Узлы типизируются этим типом и работают со всеми полями. Поля, объявленные
    ЗДЕСЬ (а не в OrchestratorOutput), остаются каналами стейта — чекпойнтятся и
    живут между ходами, — но НЕ входят в output-схему, поэтому aegra не отдаёт их
    наружу в run/stream.
    """

    # Полный JSON-датасет сотрудника. Грузится load_data на ходе 1, переживает
    # ходы по loaded-гейту, нужен call_json_analyzer/extract_assignments/ground_wiki
    # как raw_json для json_analyzer. Наружу не отдаётся — большой и не нужен клиенту.
    metrics: Any
