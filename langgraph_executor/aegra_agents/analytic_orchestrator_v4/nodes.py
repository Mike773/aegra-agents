from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import markdown2

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_gigachat import GigaChat
from sqlalchemy import select

from ..easyrag.db import session_scope
from ..easyrag.gap import record_gap
from ..easyrag.models import WikiPage
from ..gap_resolver.judge import answer_in_sources
from ..json_analyzer_v5.loader import load_dataset_obj
from ..shared.text_similarity import similarity_ratio
from ..shared.agent_dataset import (
    GetBatchAgentAggregateDatasetByFiltersComponent,
    GetBatchAgentDatasetByFiltersComponent,
)
from ..shared.assignments_service import SendAssignmentsComponent
from ..shared.offload import run_blocking
from ..shared.orgstructure import IsuEmployeeOrgstructureInfo
from .prompts import (
    ANALYZER_INITIAL_TASK,
    BUSINESS_SYSTEM_PROMPT,
    CLASSIFY_INSIGHTS_PROMPT,
    DESCRIBE_ANSWER_PROMPT,
    FIRST_ANSWER_FORMAT_PROMPT,
    INITIAL_TASK_HINT,
    LOAD_ERROR_PROMPT,
    RESPONDER_TASK_HINT,
    ROUTER_PROMPT,
    SAVE_INSIGHT_AUTO_FIXED,
    STAR_PROSE_BLOCK,
    WIKI_QUERIES_PROMPT,
)
from .state import OrchestratorState, TraceStep

logger = logging.getLogger(__name__)

_DEFAULT_DATASET = "metrics_for_agent_analyst"
_DEFAULT_EASYRAG_TOP_K = 5
_EASYRAG_SNIPPET_PREVIEW = 400
# Wiki-grounding: сколько запросов к wiki максимум генерит LLM и сколько метрик
# (имя+описание) кладём в промпт генерации запросов.
_DEFAULT_WIKI_MAX_QUERIES = 3
_WIKI_METRIC_SPECS_CAP = 30
_WIKI_METRIC_DESC_PREVIEW = 200
_MAX_CANDIDATES = 5
# Классификация метрик в инсайты для сервиса поручений.
_MAX_INSIGHTS = 12
# Сколько символов описания метрики кладём в каталог для промпта классификации
# (описание помогает LLM сопоставить разговорную формулировку с именем метрики).
_CATALOG_DESC_PREVIEW = 200
# Порог нечёткого совпадения имени метрики из ответа LLM с именем из каталога.
_METRIC_NAME_FUZZY = 0.82
_INSIGHT_TYPES = ("main_problem", "problem", "norm", "achievement")
_INSIGHT_TYPE_LABELS = {
    "main_problem": "Главная проблема",
    "problem": "Проблема",
    "norm": "Норма",
    "achievement": "Достижение",
}
# Порог нечёткого совпадения слова запроса с заголовком заглушки.
# Ловит склонённые формы: «бабушку» ↔ «бабушка».
_STUB_MATCH_RATIO = 0.72
_STUB_LOOKUP_LIMIT = 5
# Сколько шагов tool_call максимум разворачивать в трассу из json_analyzer.
_TRACE_TOOL_STEPS_CAP = 12
# Дообогащение вопроса к json_analyzer контекстом диалога: сколько последних пар
# «вопрос-ответ» подмешивать и до скольки символов резать каждый прошлый ответ.
_DIALOG_CTX_MAX_PAIRS = 4
_DIALOG_CTX_ANSWER_CAP = 800

# Per-turn сброс wiki-контекста: easyrag_snippets наполняют только call_easyrag и
# ground_wiki ВНУТРИ хода, а respond читает их безусловно. Без сброса в route
# сниппеты прошлого хода протекли бы в последующий chat-ответ.
_EASYRAG_RESET = {
    "easyrag_query": None,
    "easyrag_snippets": [],
    "easyrag_stub_pages": [],
    "easyrag_error": None,
}


def make_load_data_node():
    async def load_data(state: OrchestratorState, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable") or {}
        boss = (cfg.get("boss_tabnum") or "").strip()
        employee = (cfg.get("employee_tabnum") or "").strip()
        position = cfg.get("position")
        dataset_name = cfg.get("dataset_name") or _DEFAULT_DATASET
        # Внешняя привязка инсайтов: без обоих значений в сервис ничего не пишем.
        source_type = str(cfg.get("source_type") or "").strip() or None
        source_id = str(cfg.get("source_id") or "").strip() or None
        # Предагрегаты peer-групп грузим по умолчанию; клиент может выключить.
        use_aggregates = _config_flag(config, "use_peer_aggregates", default=True)

        # Первое входящее сообщение — заранее подготовленный вопрос-брифинг
        # (роль, методология, желаемый формат ответа). Сохраняем его, чтобы
        # initial_analysis выполнил задание по нему, а респондер держал формат
        # на последующих ходах. Сброс трассы — это первый узел первого хода.
        briefing = _last_user_text(state).strip() or None

        if not boss or not employee:
            return {
                "loaded": True,
                "metrics": None,
                "briefing": briefing,
                "source_type": source_type,
                "source_id": source_id,
                "reasoning_trace": [],
                "metrics_error": (
                    "В configurable нет boss_tabnum/employee_tabnum: "
                    f"boss_tabnum={boss!r}, employee_tabnum={employee!r}."
                ),
            }

        # orgstructure + dataset — СИНХРОННЫЕ внешние HTTP-клиенты. Уводим весь
        # блокирующий блок в выделенный пул с таймаутом (offload.run_blocking),
        # чтобы не блокировать event loop и не зависнуть навсегда, если сервис
        # данных не отвечает. См. shared/offload.py.
        def _fetch() -> tuple[str, dict | None, str | None, Any, str | None]:
            orgstructure = IsuEmployeeOrgstructureInfo(
                manager_id=boss,
                position=position,
                employee_id=employee,
            )
            dkey = orgstructure.direction_key()
            filters = orgstructure.combined_json()
            try:
                component = GetBatchAgentDatasetByFiltersComponent(
                    dataset_name=dataset_name,
                    filters=filters,
                )
                metrics = component.build_json_output()
            except Exception as exc:  # noqa: BLE001 — внешний клиент, сужать нечем
                return (
                    dkey,
                    None,
                    f"Ошибка при загрузке датасета {dataset_name!r}: {exc}",
                    None,
                    None,
                )
            if not use_aggregates:
                return dkey, metrics, None, None, None
            # Предагрегаты peer-групп теперь индивидуальные: какие грузить,
            # говорит сам датасет (aggregates_ids у персоны). Собираем уникальный
            # список по всем персонам и делаем по одному вызову на id — без
            # дублей, когда у босса и подчинённых предагрегаты совпадают.
            # Опциональны: сбой по одному id не роняет ни загрузку, ни остальные.
            aggregates: list | None = None
            agg_errors: list[str] = []
            for agg_id in _collect_aggregate_ids(metrics):
                try:
                    batch = GetBatchAgentAggregateDatasetByFiltersComponent(
                        dataset_name=dataset_name,
                        filters=[
                            {"object_type": "aggregate_id", "object_id": agg_id}
                        ],
                    ).build_json_output()
                except Exception as exc:  # noqa: BLE001 — внешний клиент
                    agg_errors.append(f"{agg_id}: {exc}")
                    continue
                if isinstance(batch, list):
                    for rec in batch:
                        # Пометка происхождения: по ней аналитик привязывает
                        # записи к персонам через их aggregates_ids.
                        if isinstance(rec, dict):
                            rec["aggregate_id"] = agg_id
                    aggregates = (aggregates or []) + batch
            agg_error = (
                f"Ошибка при загрузке агрегатов {dataset_name!r}: "
                + "; ".join(agg_errors)[:300]
                if agg_errors
                else None
            )
            return dkey, metrics, None, aggregates, agg_error

        try:
            direction_key, metrics, error, aggregates, agg_error = (
                await run_blocking(_fetch)
            )
        except asyncio.TimeoutError:
            direction_key, metrics, error, aggregates, agg_error = "", None, (
                "Сервис данных (оргструктура/датасет) не ответил за отведённое время."
            ), None, None
        except Exception as exc:  # noqa: BLE001 — сбой оргструктуры до датасета
            direction_key, metrics, error, aggregates, agg_error = "", None, (
                f"Ошибка внешнего сервиса данных: {type(exc).__name__}: {exc}"[:300]
            ), None, None

        return {
            "boss_tabnum": boss,
            "employee_tabnum": employee,
            "position": position,
            "direction_key": direction_key,
            # Нормализуем форму на входе в стейт: компонент на проде может отдать
            # JSON строкой — тогда dict-потребители (_org_structure_block,
            # _describe_sources_block, _strip_rankings, _has_star_data) молча
            # теряют данные, хотя аналитик строку парсит. Непарсящееся значение
            # оставляем как есть (прежнее поведение, ошибку отдаст аналитик).
            "metrics": _metrics_obj(metrics) or metrics,
            "metrics_error": error,
            "aggregates": aggregates,
            "aggregates_error": agg_error,
            "source_type": source_type,
            "source_id": source_id,
            "briefing": briefing,
            "reasoning_trace": [],
            "loaded": True,
        }

    return load_data


def _metrics_obj(metrics: Any) -> dict | None:
    """Датасет метрик как dict — та же толерантность к форме, что у аналитика
    (_parse_raw_json в json_analyzer_v5): прод может отдать JSON строкой, и
    аналитик её парсит. Потребители в оркестраторе, требующие строго dict,
    без этой нормализации молча решают, что данных нет, хотя разбор метрик
    в ответе есть. Не-dict и непарсящаяся строка → None."""
    if isinstance(metrics, dict):
        return metrics
    if isinstance(metrics, str):
        text = metrics.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def _collect_aggregate_ids(metrics: Any) -> list[str]:
    """Уникальные aggregates_ids по всем персонам (me + employees), порядок
    первого появления. Поле лежит у персоны рядом с metrics (его туда кладёт
    прод-компонент датасета); у босса и подчинённых наборы могут совпадать —
    дедупим, чтобы не дублировать запросы предагрегатов. Датасет может прийти
    JSON-строкой — та же толерантность, что у _metrics_obj."""
    data = _metrics_obj(metrics)
    if data is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for person in [data.get("me"), *(data.get("employees") or [])]:
        if not isinstance(person, dict):
            continue
        for agg_id in person.get("aggregates_ids") or []:
            if agg_id is None:
                continue
            text = str(agg_id).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


def _org_structure_block(state: OrchestratorState) -> str | None:
    """Краткий блок «кого анализируем» из датасета (me/employees) + позиция.

    {me}/{employees} бизнес-спецификации → руководитель и сотрудник(и) в фокусе.
    Только ФИО и позиция (не метрики — они идут отдельным блоком). Имена нужны,
    чтобы модель писала «у Иванова…», а не безлично. Источник внутренних названий
    пользователю не раскрывается (об этом сказано в системном промпте).
    """
    metrics = _metrics_obj(state.get("metrics"))
    if metrics is None:
        return None
    lines: list[str] = []
    me = metrics.get("me") or {}
    boss_fio = str((me.get("fio") if isinstance(me, dict) else "") or "").strip()
    if boss_fio:
        lines.append(f"Руководитель подразделения: {boss_fio}.")
    employees = metrics.get("employees") or []
    names = [
        str(e.get("fio") or "").strip()
        for e in employees if isinstance(e, dict) and str(e.get("fio") or "").strip()
    ]
    if names:
        position = str(state.get("position") or "").strip()
        pos = f", позиция — {position}" if position else ""
        lines.append("В фокусе анализа: " + ", ".join(names) + pos + ".")
    if not lines:
        return None
    return (
        "Кого анализируем (служебный ориентир — НЕ упоминай этот блок дословно):\n"
        + "\n".join(lines)
    )


def make_initial_analysis_node(llm: GigaChat, json_analyzer_graph: Any):
    """Первичный многоуровневый разбор по бизнес-спецификации (первый ход).

    Роль/методологию/формат задаёт BUSINESS_SYSTEM_PROMPT (наш, не пользователя).
    Первое сообщение — лишь триггер («Что происходит?»). Сырому JSON не доверяем —
    факты по метрикам собирает json_analyzer_v5 (peer-агрегаты уровней, ex/rr).
    Ответ идёт по структуре «главная проблема → сводка аналитики → сравнение с
    коллегами → что ещё можно разобрать».
    """

    async def initial_analysis(state: OrchestratorState, config: RunnableConfig) -> dict:
        if state.get("metrics_error") or state.get("metrics") is None:
            step: TraceStep = {
                "stage": "initial",
                "kind": "error",
                "summary": state.get("metrics_error") or "Метрики не загружены.",
            }
            return {
                "messages": [_final_message(LOAD_ERROR_PROMPT, config)],
                "reasoning_trace": _append_trace(state, [step]),
            }

        metrics = state.get("metrics")
        briefing = (state.get("briefing") or "").strip()
        direction_key = (state.get("direction_key") or "").strip()
        # Вопрос для глубокого сбора фактов: триггер руководителя + фиксированное
        # задание под структуру ответа (главная зона, динамика, фон коллег, ещё
        # 1-2 зоны и достижение). Без него аналитик собирает только главную зону,
        # и модели ответа нечего назвать в предложении продолжить — начинает
        # выдумывать метрики.
        question = "\n\n".join(
            p for p in (briefing, ANALYZER_INITIAL_TASK) if p
        )

        analysis, tool_steps, err = await _run_analyzer(
            json_analyzer_graph, metrics, question, direction_key,
            aggregates=state.get("aggregates"),
            use_aggregates=_config_flag(config, "use_peer_aggregates", default=True),
            wiki_context=_wiki_snippets_text(state),
        )

        trace_steps: list[TraceStep] = [{
            "stage": "initial",
            "kind": "intent",
            "summary": "Первый ход: многоуровневый разбор по бизнес-методологии.",
            "detail": {"has_briefing": bool(briefing)},
        }]
        if err:
            trace_steps.append({"stage": "initial", "kind": "error", "summary": err})
        trace_steps.extend(_analyzer_trace_steps(tool_steps))

        # Системный контекст: роль/методология (бизнес-промпт) → кого анализируем
        # → справочный контекст (методика/нормативы) → факты по метрикам → задача.
        parts: list[str] = [BUSINESS_SYSTEM_PROMPT]
        org_block = _org_structure_block(state)
        if org_block:
            parts.append(org_block)
        wiki_block = _easyrag_system_block(state)
        if wiki_block:
            parts.append(wiki_block)
        memory_block = _memory_system_block(state)
        if memory_block:
            parts.append(memory_block)
        if analysis:
            parts.append("Данные по метрикам сотрудника (собраны из полного набора):\n" + analysis)
        else:
            # Разбора нет (аналитик упал/пусто). Сырой обрезанный JSON в контекст НЕ
            # отдаём — обрезка даёт неверные числа. Честно говорим «данных нет» и
            # просим не выдумывать значения.
            parts.append(
                "Данных по метрикам сейчас нет: аналитический разбор недоступен. "
                "Не приводи конкретных числовых значений по памяти; предложи "
                "уточнить вопрос или повторить запрос — он будет пересчитан аналитиком."
            )
        if _has_star_data(metrics):
            parts.append(STAR_PROSE_BLOCK)
        parts.append(INITIAL_TASK_HINT)

        human = briefing or "Что происходит с показателями сотрудника?"
        ai = await asyncio.to_thread(llm.invoke, [
            SystemMessage(content="\n\n".join(parts)),
            HumanMessage(content=human),
        ])
        text = ai.content if isinstance(ai.content, str) else str(ai.content)

        # Форма первого ответа: рассказ с главной проблемой и приглашением
        # продолжить. Промпт сам решает по запросу руководителя, задан ли там
        # формат, — тогда черновик возвращается дословно. Сбой не фатален:
        # остаёмся на черновике. Только первый ход — respond это не касается.
        if text.strip() and _config_flag(config, "format_first_answer", default=True):
            try:
                formatted = await asyncio.to_thread(llm.invoke, [
                    SystemMessage(content=FIRST_ANSWER_FORMAT_PROMPT),
                    HumanMessage(content=(
                        f"Запрос руководителя:\n{human}\n\n"
                        f"Черновой ответ аналитика:\n{text}"
                    )),
                ])
                new_text = formatted.content if isinstance(
                    formatted.content, str) else str(formatted.content)
                if new_text.strip():
                    text = new_text
                    trace_steps.append({
                        "stage": "initial",
                        "kind": "decision",
                        "summary": "Привёл первый ответ к форме рассказа "
                                   "(главная проблема + приглашение продолжить).",
                    })
            except Exception as exc:  # noqa: BLE001 — форма опциональна
                logger.warning("Переформатирование первого ответа упало: %s", exc)
                trace_steps.append({
                    "stage": "initial",
                    "kind": "error",
                    "summary": f"Не удалось привести ответ к форме рассказа: {exc}",
                })

        trace_steps.append({
            "stage": "initial",
            "kind": "decision",
            "summary": "Сформировал первичный разбор и предложил, что разобрать дальше.",
        })
        new_trace = _append_trace(state, trace_steps)

        out_state = {
            "messages": [_final_message(await _with_description(
                text, {**state, "reasoning_trace": new_trace}, config,
                llm=llm, question=human,
            ), config)],
            "reasoning_trace": new_trace,
        }
        # Опорный широкий разбор — единственное sticky-поле первого хода, ставится
        # ОДИН раз и не перезаписывается узкими analytics-ходами.
        if analysis and not err:
            out_state["metrics_summary"] = analysis
        return out_state

    return initial_analysis


_ROUTE_LABELS = {"analytics", "wiki", "save_insight", "chat", "done"}


def make_route_node(llm: GigaChat):
    def route(state: OrchestratorState) -> dict:
        # route — первый узел каждого последующего хода: здесь СБРАСЫВАЕМ трассу
        # (она строго per-turn) и кладём первый шаг — классификацию запроса.
        last_text = _last_user_text(state)
        if not last_text:
            return {
                **_EASYRAG_RESET,
                "intent": "chat",
                "reasoning_trace": [{
                    "stage": "route", "kind": "intent",
                    "summary": "Пустая реплика — обычный чат.",
                    "detail": {"intent": "chat"},
                }],
            }

        ai = llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=last_text),
        ])
        label = (ai.content or "").strip().lower()
        if label not in _ROUTE_LABELS:
            label = "chat"
        return {
            **_EASYRAG_RESET,
            "intent": label,
            "reasoning_trace": [{
                "stage": "route", "kind": "intent",
                "summary": f"Определил тип запроса как «{label}».",
                "detail": {"intent": label, "query": last_text[:200]},
            }],
        }

    return route


def make_respond_node(llm: GigaChat):
    async def respond(state: OrchestratorState, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable") or {}

        # Роль/методологию/формат задаёт бизнес-промпт (override имеет приоритет).
        system_prompt = cfg.get("system_prompt_override") or BUSINESS_SYSTEM_PROMPT

        parts: list[str] = [system_prompt]
        org_block = _org_structure_block(state)
        if org_block:
            parts.append(org_block)
        memory_block = _memory_system_block(state)
        if memory_block:
            parts.append(memory_block)
        metrics_block = _metrics_system_block(state)
        if metrics_block:
            parts.append(metrics_block)
        easyrag_block = _easyrag_system_block(state)
        if easyrag_block:
            parts.append(easyrag_block)
        # Правила про звезду нужны только вместе с фактами по метрикам: в чисто
        # разговорный/wiki-ход их подмешивать незачем.
        if metrics_block and _has_star_data(state.get("metrics")):
            parts.append(STAR_PROSE_BLOCK)
        parts.append(RESPONDER_TASK_HINT)
        system_text = "\n\n".join(parts)

        messages: list[Any] = [SystemMessage(content=system_text)]
        messages.extend(_history_for_llm(state.get("messages") or []))

        ai = await asyncio.to_thread(llm.invoke, messages)
        text = ai.content if isinstance(ai.content, str) else str(ai.content)

        used = []
        if metrics_block:
            used.append("метрики")
        if easyrag_block:
            used.append("wiki")
        decision: TraceStep = {
            "stage": "respond",
            "kind": "decision",
            "summary": (
                "Сформировал ответ на основе: " + ", ".join(used)
                if used else "Сформировал ответ без внешних источников."
            ),
            "detail": {"sources": used},
        }
        new_trace = _append_trace(state, [decision])
        return {
            "messages": [_final_message(await _with_description(
                text, {**state, "reasoning_trace": new_trace}, config,
                llm=llm, question=_last_user_text(state),
            ), config)],
            "reasoning_trace": new_trace,
        }

    return respond


def make_call_json_analyzer_node(json_analyzer_graph: Any):
    """Обёртка, дёргающая json_analyzer-подграф под последний вопрос.

    Подграф пересобирает sqlite-кэш по ``raw_json`` и pgvector-эмбеддинги по
    ``direction_key`` на каждом вызове. Ошибки изолируются: респондер
    отработает по сырому JSON метрик (фоллбэк в ``_metrics_system_block``).
    """

    async def call_json_analyzer(state: OrchestratorState, config: RunnableConfig) -> dict:
        last_text = _last_user_text(state)
        direction_key = (state.get("direction_key") or "").strip()
        metrics = state.get("metrics")

        if not last_text:
            err = "Нет реплики пользователя для аналитического запроса."
            return {
                "analytics_question": None,
                "analytics_answer": None,
                "analytics_error": err,
                "reasoning_trace": _append_trace(
                    state, [{"stage": "json_analyzer", "kind": "error", "summary": err}]
                ),
            }

        # В аналитик уходит вопрос, дообогащённый контекстом диалога (чтобы
        # разрешить «эти/западающие/те», а также «давай подробнее» о зоне,
        # названной в блоке продолжения прошлого хода); в state сохраняем сырой
        # last_text.
        enriched_q = _question_with_dialogue_context(state, last_text)
        answer, tool_steps, err = await _run_analyzer(
            json_analyzer_graph, metrics, enriched_q, direction_key,
            aggregates=state.get("aggregates"),
            use_aggregates=_config_flag(config, "use_peer_aggregates", default=True),
            wiki_context=_wiki_snippets_text(state),
        )
        if err and metrics is None:
            err = state.get("metrics_error") or err

        steps = _analyzer_trace_steps(tool_steps)
        if err:
            steps.append({"stage": "json_analyzer", "kind": "error", "summary": err})
            step_text = "📊 Аналитик метрик недоступен — отвечу по сырым данным."
        elif answer:
            steps.append({
                "stage": "json_analyzer", "kind": "decision",
                "summary": "Аналитик метрик собрал данные и сформировал ответ.",
            })
            step_text = "📊 Проанализировал метрики аналитиком."
        else:
            step_text = ""

        return {
            "analytics_question": last_text,
            "analytics_answer": answer,
            "analytics_error": err,
            "reasoning_trace": _append_trace(state, steps),
            **_step_update(config, step_text),
        }

    return call_json_analyzer


async def _record_unanswered_gap(
    direction_key: str, question: str, query_vec: list[float] | None
) -> None:
    """Зафиксировать gap: сниппеты были, но судья счёл, что ответа в них нет.

    Пустую выборку фиксирует сам easyrag-подграф (maybe_record_gap), поэтому сюда
    попадает только случай «нашли, но не отвечает» — иначе задвоили бы gap.
    """
    async with session_scope() as session:
        await record_gap(
            session,
            direction_key=direction_key,
            question=question,
            embedding=query_vec or None,
            resolved_section_ids=(),
        )


def make_call_easyrag_node(easyrag_graph: Any):
    """Обёртка, дёргающая easyrag-подграф под последний вопрос пользователя.

    Подграф возвращает релевантные секции wiki по ``direction_key`` сотрудника
    и пишет gap, если ничего не найдено. Ошибки сети/БД не валят оркестратор —
    респондер просто отработает без wiki-контекста.
    """

    async def call_easyrag(state: OrchestratorState, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable") or {}
        if cfg.get("easyrag_enabled") is False:
            return {"easyrag_snippets": [], "easyrag_stub_pages": [], "easyrag_error": None}

        direction_key = (state.get("direction_key") or "").strip()
        last_text = _last_user_text(state)
        if not direction_key or not last_text:
            return {
                "easyrag_snippets": [],
                "easyrag_stub_pages": [],
                "easyrag_query": last_text or None,
                "easyrag_error": None,
            }

        top_k = int(cfg.get("easyrag_top_k") or _DEFAULT_EASYRAG_TOP_K)
        try:
            result = await easyrag_graph.ainvoke({
                "query": last_text,
                "direction_key": direction_key,
                "top_k": top_k,
            })
            snippets = result.get("snippets") or []
            # Сниппеты есть, но отвечают ли они на вопрос — решает LLM-судья. Если
            # нет — фиксируем gap (пустую выборку easyrag уже записал сам, поэтому
            # тут только случай «нашли, но не отвечает»). Side-эффект: запись gap
            # никогда не должна влиять на ответ пользователю.
            if snippets and cfg.get("gap_on_unanswered") is not False:
                try:
                    verdict = await answer_in_sources(
                        last_text, [s.get("body_md") or "" for s in snippets]
                    )
                    if verdict.found is False:
                        await _record_unanswered_gap(
                            direction_key, last_text, result.get("query_vec")
                        )
                except Exception:  # noqa: BLE001 — запись gap не влияет на ответ
                    pass
            # Контента нет — но, возможно, сущность уже заведена как пустая
            # заглушка. Найдём релевантные заглушки, чтобы респондер сказал об этом.
            stub_pages = (
                [] if snippets
                else await _find_relevant_stub_pages(direction_key, last_text)
            )
            return {
                "easyrag_query": last_text,
                "easyrag_snippets": snippets,
                "easyrag_stub_pages": stub_pages,
                "easyrag_error": None,
                "reasoning_trace": _append_trace(
                    state, _easyrag_trace_steps(snippets, stub_pages, None)
                ),
                **_step_update(config, _wiki_step_text(snippets, stub_pages, None)),
            }
        except Exception as exc:  # noqa: BLE001 — внешний подграф (сеть/БД), сужать нечем
            err = f"{type(exc).__name__}: {exc}"[:300]
            return {
                "easyrag_query": last_text,
                "easyrag_snippets": [],
                "easyrag_stub_pages": [],
                "easyrag_error": err,
                "reasoning_trace": _append_trace(
                    state, _easyrag_trace_steps([], [], err)
                ),
                **_step_update(config, _wiki_step_text([], [], err)),
            }

    return call_easyrag


def _format_metric_specs(specs: list[dict]) -> str:
    return "\n".join(
        f"- {s['name']}" + (f" — {s['description']}" if s.get("description") else "")
        for s in specs
    )


def _generate_wiki_queries(
    llm: GigaChat, state: OrchestratorState, specs: list[dict], *, max_n: int
) -> list[str]:
    """LLM-генерация поисковых запросов к wiki по вопросу/брифингу + метрикам.

    Затравка: вопрос/брифинг текущего хода + список метрик. На analytics-ходу
    добавляем выводы аналитика (он уже отработал) — запросы становятся точнее.
    На первом ходу answer'а ещё нет (асимметрия by design). Сбой LLM → [].
    """
    seed = _last_user_text(state).strip()
    parts = [
        f"Вопрос руководителя: {seed}" if seed else "Первичный обзор метрик сотрудника.",
        "Метрики сотрудника:\n" + _format_metric_specs(specs),
    ]
    analytics_answer = state.get("analytics_answer")
    if analytics_answer and not state.get("analytics_error"):
        parts.append("Выводы аналитика по метрикам:\n" + str(analytics_answer))
    try:
        ai = llm.invoke([
            SystemMessage(content=WIKI_QUERIES_PROMPT),
            HumanMessage(content="\n\n".join(parts)),
        ])
        return _parse_query_list_json(ai.content, max_n=max_n)
    except Exception:  # noqa: BLE001 — внешний LLM, сужать нечем
        return []


async def _gather_wiki_snippets(
    easyrag_graph: Any, queries: list[str], *, direction_key: str, top_k: int
) -> tuple[list[dict], str | None]:
    """Параллельные запросы к easyrag + мёрж сниппетов.

    Дедуп по section_id (оставляем больший similarity), сорт по similarity desc,
    cap top_k. Ошибку возвращаем только если упали ВСЕ запросы и сниппетов нет —
    частичный сбой не должен прятать находки.
    """
    results = await asyncio.gather(
        *[
            easyrag_graph.ainvoke({
                "query": q,
                "direction_key": direction_key,
                "top_k": top_k,
            })
            for q in queries
        ],
        return_exceptions=True,
    )
    best: dict[str, dict] = {}
    errors: list[str] = []
    for res in results:
        if isinstance(res, Exception):
            errors.append(f"{type(res).__name__}: {res}"[:300])
            continue
        for s in (res.get("snippets") if isinstance(res, dict) else None) or []:
            sid = s.get("section_id")
            key = sid if sid is not None else id(s)
            prev = best.get(key)
            if prev is None or (s.get("similarity") or 0) > (prev.get("similarity") or 0):
                best[key] = s
    snippets = sorted(
        best.values(), key=lambda s: s.get("similarity") or 0, reverse=True
    )[:top_k]
    err = errors[0] if (errors and not snippets) else None
    return snippets, err


def make_ground_wiki_node(llm: GigaChat, easyrag_graph: Any):
    """Wiki-grounding: по вопросу/брифингу + метрикам генерит запросы к wiki и
    подмешивает найденные сниппеты в state (для initial_analysis и respond).

    Запросы генерируются ВСЕГДА при наличии метрик (без гейта «понимает ли модель»):
    wiki хранит корпоративную специфику — методику, нормативы, определения, —
    которой у модели по определению нет. Ошибки изолируются: при сбое узел просто
    не добавляет wiki-контекст, конвейер продолжает работу.
    """

    async def ground_wiki(state: OrchestratorState, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable") or {}
        if cfg.get("easyrag_enabled") is False or cfg.get("wiki_grounding_enabled") is False:
            return {}
        if state.get("metrics") is None or state.get("metrics_error"):
            return {}
        direction_key = (state.get("direction_key") or "").strip()
        if not direction_key:
            return {}

        specs = _distinct_metric_specs(state.get("metrics"))
        if not specs:
            return {}

        queries = await asyncio.to_thread(
            _generate_wiki_queries,
            llm, state, specs,
            max_n=int(cfg.get("wiki_max_queries") or _DEFAULT_WIKI_MAX_QUERIES),
        )
        if not queries:
            return {}

        top_k = int(cfg.get("easyrag_top_k") or _DEFAULT_EASYRAG_TOP_K)
        snippets, err = await _gather_wiki_snippets(
            easyrag_graph, queries, direction_key=direction_key, top_k=top_k
        )
        joined_query = " | ".join(queries)

        if not snippets and not err:
            stub_pages = await _find_relevant_stub_pages(direction_key, joined_query)
            return {
                "easyrag_query": joined_query,
                "easyrag_snippets": [],
                "easyrag_stub_pages": stub_pages,
                "easyrag_error": None,
                "reasoning_trace": _append_trace(
                    state, _easyrag_trace_steps([], stub_pages, None)
                ),
                **_step_update(config, _wiki_step_text([], stub_pages, None)),
            }

        return {
            "easyrag_query": joined_query,
            "easyrag_snippets": snippets,
            "easyrag_stub_pages": [],
            "easyrag_error": err,
            "reasoning_trace": _append_trace(
                state, _easyrag_trace_steps(snippets, [], err)
            ),
            **_step_update(config, _wiki_step_text(snippets, [], err)),
        }

    return ground_wiki


def _wiki_step_text(
    snippets: list[dict], stub_pages: list[dict], err: str | None
) -> str:
    """Однострочный текст wiki-шага для промежуточного сообщения хода."""
    if err:
        return "📖 Контекст из wiki недоступен."
    if snippets:
        pages = []
        for s in snippets[:3]:
            name = s.get("page_title") or s.get("section_title") or s.get("slug")
            if name and name not in pages:
                pages.append(name)
        tail = f": {', '.join(pages)}" if pages else ""
        return f"📖 Нашёл в wiki {len(snippets)} фрагмент(ов){tail}."
    if stub_pages:
        names = ", ".join(s.get("title") or s.get("slug") or "-" for s in stub_pages)
        return f"📖 В wiki есть заглушки по теме (без содержания): {names}."
    return "📖 В wiki релевантного не нашёл."


def _easyrag_trace_steps(
    snippets: list[dict], stub_pages: list[dict], err: str | None
) -> list[TraceStep]:
    """Шаги kb_hit по найденным секциям wiki (или пусто/заглушка/ошибка)."""
    if err:
        return [{"stage": "easyrag", "kind": "error",
                 "summary": f"Контекст из wiki недоступен: {err}"}]
    if not snippets:
        if stub_pages:
            names = ", ".join(s.get("title") or s.get("slug") or "-" for s in stub_pages)
            return [{"stage": "easyrag", "kind": "kb_hit",
                     "summary": f"В wiki есть пустые заглушки по теме: {names} (без содержания)."}]
        return [{"stage": "easyrag", "kind": "kb_hit",
                 "summary": "В базе знаний релевантного контента не нашлось."}]
    steps: list[TraceStep] = []
    for s in snippets[:5]:
        page = s.get("page_title") or s.get("slug") or "-"
        title = s.get("section_title") or s.get("anchor") or "-"
        sim = s.get("similarity")
        sim_str = f", релевантность {sim:.2f}" if isinstance(sim, (int, float)) else ""
        body = (s.get("body_md") or "").strip().replace("\n", " ")
        if len(body) > 160:
            body = body[:160] + "…"
        steps.append({
            "stage": "easyrag", "kind": "kb_hit",
            "summary": f"Нашёл в wiki «{page} / {title}»{sim_str}: {body}",
            "detail": {"page": page, "section": title, "similarity": sim},
        })
    return steps


async def _find_relevant_stub_pages(direction_key: str, query: str) -> list[dict]:
    """Найти пустые страницы-заглушки направления, релевантные запросу.

    Заглушки (``type='stub'``) не имеют секций/эмбеддингов, поэтому vector-поиск
    easyrag их не находит. Сопоставляем имя заглушки со словами запроса лексически
    (substring + нечёткое сходство для склонённых форм). Возвращает ``[{slug, title}]``.
    """
    if not direction_key or not query.strip():
        return []
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(WikiPage.slug, WikiPage.title, WikiPage.aliases).where(
                    WikiPage.direction_key == direction_key,
                    WikiPage.type == "stub",
                )
            )
        ).all()
    if not rows:
        return []
    q_words = {w for w in re.findall(r"\w+", query.casefold()) if len(w) >= 3}
    if not q_words:
        return []

    out: list[dict] = []
    seen: set[str] = set()
    for slug, title, aliases in rows:
        names = [title or ""] + [a for a in (aliases or []) if a]
        if slug not in seen and _name_matches_query(names, q_words):
            seen.add(slug)
            out.append({"slug": slug, "title": (title or slug)})
        if len(out) >= _STUB_LOOKUP_LIMIT:
            break
    return out


def _name_matches_query(names: list[str], q_words: set[str]) -> bool:
    for name in names:
        nlow = (name or "").strip().casefold()
        if len(nlow) < 3:
            continue
        for w in q_words:
            if w == nlow or w in nlow or nlow in w:
                return True
            if similarity_ratio(w, nlow) >= _STUB_MATCH_RATIO:
                return True
    return False


async def _classify_insights(
    llm: GigaChat, metrics: Any, source_text: str
) -> list[dict]:
    """Разбор аналитика → структурированные инсайты сервиса.

    Экрана подтверждения нет, поэтому классификация вызывается напрямую перед
    отправкой. Любой сбой LLM/парсинга → [] (инсайт просто не пишется).
    """
    if metrics is None or not (source_text or "").strip():
        return []
    catalog = _collect_metric_catalog(metrics)
    ctx = [
        f"Каталог метрик (id — название):\n{_format_metric_catalog(catalog)}",
        f"Разбор аналитика:\n{source_text}",
    ]
    try:
        ai = await asyncio.to_thread(llm.invoke, [
            SystemMessage(content=CLASSIFY_INSIGHTS_PROMPT),
            HumanMessage(content="\n\n".join(ctx)),
        ])
        return _parse_insights_json(ai.content, catalog)
    except Exception:  # noqa: BLE001 — LLM-вызов, сужать нечем
        logger.warning("Классификация инсайтов упала", exc_info=True)
        return []


async def _submit_insight(
    state: OrchestratorState, config: RunnableConfig | None, ins: dict
) -> str | None:
    """Отправка ОДНОГО главного инсайта (content.insight — объект, контракт v4).

    Возвращает текст ошибки либо None при успехе.
    """
    cfg = (config or {}).get("configurable") or {}
    # thread_id — id треда aegra из /threads, сервер инжектит его в configurable.
    thread_id = str(cfg.get("thread_id") or "").strip()

    def _submit() -> None:
        SendAssignmentsComponent(
            boss_tabnum=(state.get("boss_tabnum") or "").strip(),
            employee_tabnum=(state.get("employee_tabnum") or "").strip(),
            direction_key=(state.get("direction_key") or "").strip(),
            thread_id=thread_id,
            insight=ins,
            source_type=state.get("source_type"),
            source_id=state.get("source_id"),
        ).submit()

    try:
        # Синхронный HTTP-клиент сервиса — в выделенный пул с таймаутом.
        await run_blocking(_submit)
        return None
    except asyncio.TimeoutError:
        return "Сервис инсайтов не ответил за отведённое время (timeout)."
    except Exception as exc:  # noqa: BLE001 — внешний клиент, сужать нечем
        return f"{type(exc).__name__}: {exc}"[:300]


def _has_source(state: OrchestratorState) -> bool:
    """Инсайты пишем, только когда пришли ОБА параметра привязки."""
    return bool((state.get("source_type") or "").strip()
                and (state.get("source_id") or "").strip())


def _first_insight(insights: list[dict]) -> dict | None:
    """Один стартовый инсайт: главная проблема, иначе «норма», иначе первый."""
    for kind in ("main_problem", "norm"):
        for ins in insights:
            if ins.get("type") == kind:
                return ins
    return insights[0] if insights else None


def make_auto_insight_node(llm: GigaChat):
    """Стартовый инсайт: сразу после первичного разбора пишем в сервис главную
    проблему сотрудника (если её нет — вывод «ситуация в норме»).

    Подтверждения руководителя нет — привязка идёт по source_type/source_id из
    configurable. Их нет — узел ничего не делает. Пользователю узел не пишет:
    итоговое сообщение хода уже отдал initial_analysis. Любой сбой изолирован —
    первичный разбор руководитель получает в любом случае.
    """

    async def auto_insight(state: OrchestratorState, config: RunnableConfig) -> dict:
        if not _has_source(state):
            return {"reasoning_trace": _append_trace(state, [{
                "stage": "assignments", "kind": "decision",
                "summary": "Привязка инсайта не задана — в сервис ничего не пишу.",
            }])}

        # Источник стартового инсайта — ТОЛЬКО первичный разбор аналитика. Если он
        # не удался (сбой сети/LLM), metrics_summary пуст, а в диалоге лежит ответ
        # «данных нет» — классифицировать его нельзя: инсайт получился бы выдуманным.
        summary = (state.get("metrics_summary") or "").strip()
        if not summary:
            return {"reasoning_trace": _append_trace(state, [{
                "stage": "assignments", "kind": "decision",
                "summary": "Первичный разбор не собран — стартовый инсайт не пишу.",
            }])}

        insights = await _classify_insights(llm, state.get("metrics"), summary)
        chosen = _first_insight(_enforce_single_main_problem(insights))
        if chosen is None:
            return {"reasoning_trace": _append_trace(state, [{
                "stage": "assignments", "kind": "decision",
                "summary": "Из первичного разбора не выделился инсайт для фиксации.",
            }])}

        err = await _submit_insight(state, config, chosen)
        if err:
            return {"reasoning_trace": _append_trace(state, [{
                "stage": "assignments", "kind": "error",
                "summary": f"Не удалось создать стартовый инсайт: {err}",
            }])}
        return {
            "reasoning_trace": _append_trace(state, [{
                "stage": "assignments", "kind": "decision",
                "summary": (
                    f"Создал инсайт «{_INSIGHT_TYPE_LABELS.get(chosen.get('type'), '')}» "
                    f"по метрике «{chosen.get('metric_name') or '—'}»."
                ),
                "detail": {
                    "type": chosen.get("type"),
                    "metric_name": chosen.get("metric_name"),
                    "source_type": state.get("source_type"),
                    "source_id": state.get("source_id"),
                },
            }]),
        }

    return auto_insight


def make_save_insight_node(llm: GigaChat):
    """Явная просьба «запиши этот вывод»: ручной фиксации больше нет.

    Главный инсайт уходит в сервис один раз — автоматически на первом ходе
    треда (auto_insight). Узел лишь объясняет это руководителю: без LLM и без
    обращения к сервису. llm в сигнатуре — для симметрии фабрик графа. Маршрут
    интента save_insight сохранён.
    """
    del llm  # намеренно не используется

    async def save_insight(state: OrchestratorState, config: RunnableConfig) -> dict:
        new_trace = _append_trace(state, [{
            "stage": "assignments", "kind": "decision",
            "summary": "Просьба зафиксировать вывод: главный инсайт уже "
                       "фиксируется автоматически, повторно не отправляю.",
        }])
        return {
            "messages": [_final_message(SAVE_INSIGHT_AUTO_FIXED, config)],
            "reasoning_trace": new_trace,
        }

    return save_insight


def need_load(state: OrchestratorState) -> str:
    return "route" if state.get("loaded") else "load_data"


def after_route(state: OrchestratorState) -> str:
    intent = state.get("intent")
    # Вопрос по метрикам, в т.ч. «давай подробнее» о названной зоне/достижении.
    if intent == "analytics":
        return "call_json_analyzer"
    if intent == "wiki":
        return "call_easyrag"
    # Явная просьба зафиксировать вывод — пишем инсайт без подтверждения.
    if intent == "save_insight":
        return "save_insight"
    # chat, done и любой неожиданный intent — к респондеру (done: прощается).
    return "respond"


def _last_user_text(state: OrchestratorState) -> str:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def _last_ai_text(state: OrchestratorState) -> str:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, AIMessage) and not _is_step(m):
            content = m.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
    return ""


def _plain_text(m: Any) -> str:
    """Текст сообщения как строка (контент бывает строкой или списком блоков)."""
    content = getattr(m, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content or "")


def _question_with_dialogue_context(state: OrchestratorState, current_q: str) -> str:
    """Дообогащает вопрос к json_analyzer контекстом предыдущих взаимодействий.

    json_analyzer вызывается без истории, поэтому ссылки вроде «детализируй
    западающие», «разбери эти», «а по тем что с трендом» он разрешить не может.
    Подмешиваем опорный разбор и последние пары «вопрос руководителя → ответ
    аналитика» (из очищенной истории), чтобы аналитик САМ понял referent. Имена
    метрик отдельным LLM-шагом НЕ вытаскиваем (по решению). Если контекста нет —
    возвращаем вопрос как есть.
    """
    history = _history_for_llm(state.get("messages") or [])
    # Собираем пары human→следующий ai. Последняя human-реплика (текущий вопрос)
    # пары не образует — у неё ещё нет ответа, поэтому естественно исключается.
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(history):
        m = history[i]
        if isinstance(m, HumanMessage) and i + 1 < len(history) \
                and isinstance(history[i + 1], AIMessage):
            q = _plain_text(m).strip()
            a = _plain_text(history[i + 1]).strip()
            if q and a:
                pairs.append((q, a))
            i += 2
            continue
        i += 1
    pairs = pairs[-_DIALOG_CTX_MAX_PAIRS:]

    ctx_parts: list[str] = []
    summary = (state.get("metrics_summary") or "").strip()
    if summary:
        ctx_parts.append("Первичный разбор метрик:\n" + summary)
    for q, a in pairs:
        if len(a) > _DIALOG_CTX_ANSWER_CAP:
            a = a[:_DIALOG_CTX_ANSWER_CAP] + "…"
        ctx_parts.append(f"Вопрос: {q}\nОтвет: {a}")

    if not ctx_parts:
        return current_q

    block = "\n\n".join(ctx_parts)
    return (
        "Предыдущие взаимодействия (вопросы руководителя и ответы аналитика) — "
        "опирайся на них, чтобы понять, о каких именно метриках идёт речь в "
        "текущем вопросе (ссылки вроде «эти», «западающие», «те»):\n"
        f"{block}\n\nТекущий вопрос: {current_q}"
    )


# --- Блок A/B: сквозная трасса рассуждения и режим describe_answer -----------

def _trace(state: OrchestratorState) -> list[TraceStep]:
    return list(state.get("reasoning_trace") or [])


def _append_trace(state: OrchestratorState, steps: list[TraceStep]) -> list[TraceStep]:
    """Per-turn накопление: читаем текущую трассу и возвращаем её + новые шаги.

    Плоское поле TypedDict перезаписывается каждым возвратом узла, поэтому
    накапливаем явной конкатенацией (reducer не используем — он копил бы трассу
    и между ходами, а она строго per-turn).
    """
    return _trace(state) + steps


def _config_flag(
    config: RunnableConfig | None, key: str, *, default: bool
) -> bool:
    """Булев флаг из configurable: bool как есть, строка 'true'/'1'/'yes'/'да' → True.

    Отсутствующее значение (None) даёт default — так describe_answer выключен по
    умолчанию, а emit_progress_messages включён.
    """
    cfg = (config or {}).get("configurable") or {}
    val = cfg.get(key)
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "1", "yes", "да"}
    return bool(val)


_KIND_LABELS = {
    "intent": "Классификация запроса",
    "kb_hit": "База знаний (wiki)",
    "tool_call": "Метрики",
    "derived_metric": "Производный показатель",
    "hypothesis": "Гипотеза",
    "decision": "Вывод",
    "error": "Сбой",
}


# Заголовок раздела трассы — общий для рендера и для вырезания из истории,
# чтобы они не разъезжались.
_TRACE_SECTION_TITLE = "### Как я пришёл к выводу"


def render_trace(trace: list[TraceStep], preamble: str = "") -> str:
    """Детерминированный markdown-раздел «Как я пришёл к выводу» из трассы.

    Без LLM — строго по накопленным шагам, поэтому без галлюцинаций (Блок B.3.a).
    ``preamble`` (блок «Исходные данные») вставляется после заголовка раздела;
    раздел рендерится и при пустой трассе, если preamble непустой.
    """
    steps = [s for s in (trace or []) if (s.get("summary") or "").strip()]
    if not steps and not preamble:
        return ""
    lines = ["---", _TRACE_SECTION_TITLE, ""]
    if preamble:
        lines.append(preamble)
        lines.append("")
    for i, step in enumerate(steps, 1):
        label = _KIND_LABELS.get(step.get("kind", ""), step.get("stage") or "Шаг")
        lines.append(f"{i}. **{label}.** {step['summary'].strip()}")
    return "\n".join(lines).rstrip()


def _strip_trace_section(content: Any) -> Any:
    """Убирает дописанный раздел трассы из текста AIMessage.

    `_with_description` клеит раздел прямо в content (его видит пользователь),
    и эта реплика оседает в истории. Если подавать её в LLM как есть, модель
    имитирует раздел и генерирует свою (галлюцинированную) копию. Поэтому при
    подаче истории в LLM раздел вырезаем — вместе с предшествующим «---».
    """
    if not isinstance(content, str):
        return content
    idx = content.find(_TRACE_SECTION_TITLE)
    if idx == -1:
        return content
    prefix = content[:idx].rstrip()
    if prefix.endswith("---"):
        prefix = prefix[:-3].rstrip()
    return prefix


def _history_for_llm(messages: list[Any]) -> list[Any]:
    """История для LLM-вызова респондера с вырезанной трассой из AI-реплик.

    Шаговые сообщения (_STEP_KEY) полностью отбрасываем: это служебный прогресс
    хода («📖 Нашёл в wiki…»), а не реплики ассистента — иначе модель начнёт их
    имитировать как свои прошлые ответы.
    """
    out: list[Any] = []
    for m in messages or []:
        if _is_step(m):
            continue
        if isinstance(m, AIMessage):
            # При answer_html content — HTML; в LLM подаём исходный markdown.
            src = (m.additional_kwargs or {}).get(_MARKDOWN_KEY)
            out.append(AIMessage(content=_strip_trace_section(
                src if isinstance(src, str) else m.content
            )))
        else:
            out.append(m)
    return out


def _trace_for_description(trace: list[TraceStep]) -> str:
    """Трасса одним текстом для LLM-описания: шаги по порядку, у tool_call — детали.

    summary шага аналитика уже содержит `tool(args) → результат`; отдельно
    добавляем только мотивацию вызова (reasoning), если модель её писала.
    """
    lines: list[str] = []
    for i, step in enumerate(trace or [], 1):
        summary = (step.get("summary") or "").strip()
        if not summary:
            continue
        label = _KIND_LABELS.get(step.get("kind", ""), step.get("stage") or "шаг")
        lines.append(f"{i}. [{label}] {summary}")
        if step.get("kind") == "tool_call":
            reasoning = ((step.get("detail") or {}).get("reasoning") or "").strip()
            if reasoning:
                lines.append(f"   Мотивация вызова: {reasoning}")
    return "\n".join(lines)


async def _describe_trace_llm(
    llm: Any, question: str, answer: str, trace: list[TraceStep]
) -> str:
    """Развёрнутое описание хода анализа по трассе (describe_answer=true).

    Любая ошибка LLM → "" — вызывающий откатится на детерминированный render_trace,
    так что режим не может сломать ответ.
    """
    steps_text = _trace_for_description(trace)
    if not steps_text:
        return ""
    parts = []
    if (question or "").strip():
        parts.append(f"Вопрос руководителя:\n{question.strip()}")
    parts.append(f"Итоговый ответ:\n{answer}")
    parts.append(f"Трасса (шаги анализа по порядку):\n{steps_text}")
    try:
        ai = await asyncio.to_thread(llm.invoke, [
            SystemMessage(content=DESCRIBE_ANSWER_PROMPT),
            HumanMessage(content="\n\n".join(parts)),
        ])
        content = ai.content if isinstance(ai.content, str) else str(ai.content)
        return content.strip()
    except Exception:  # noqa: BLE001 — LLM-вызов, сужать нечем
        logger.warning("describe_answer: LLM-описание трассы упало, фоллбэк на список",
                       exc_info=True)
        return ""


# Превью содержательной долгосрочной памяти и запроса к wiki в блоке
# «Исходные данные» (полные тексты там не нужны — это опись источников).
_SOURCES_MEMORY_PREVIEW = 200
_SOURCES_QUERY_PREVIEW = 120


def _metrics_source_line(metrics: Any) -> str:
    """Строка «- Метрики: …» для блока источников: уникальные, верхнеуровневые, периоды.

    Тот же канонический ``load_dataset_obj``, что и остальные сканы датасета:
    уникальность — по ``metric_id`` (метрика повторяется в разрезах по element и
    по периодам), верхнеуровневые — корневые строки без parent_uid.
    """
    try:
        rows = load_dataset_obj(metrics)
    except Exception:  # noqa: BLE001 — датасет от внешнего клиента, форма не гарантирована
        rows = []
    if not rows:
        return "- Метрики: датасет пуст."
    seen_ids: set[str] = set()
    top_names: list[str] = []
    periods: dict[str, set[str]] = {}
    for r in rows:
        mid = str(r.get("metric_id") or "").strip()
        if mid:
            seen_ids.add(mid)
        if not r.get("parent_uid"):
            name = str(r.get("metric_name") or "").strip()
            if name and name not in top_names:
                top_names.append(name)
        date = str(r.get("date") or "").strip()
        if date:
            periods.setdefault(str(r.get("calc_period") or "период"), set()).add(date)
    parts = [f"всего уникальных метрик — {len(seen_ids)}"]
    if top_names:
        parts.append("верхнеуровневые: " + ", ".join(top_names))
    if periods:
        per_strs = []
        for period, dates in sorted(periods.items()):
            ds = sorted(dates)
            span = ds[0] if len(ds) == 1 else f"{ds[0]} … {ds[-1]}"
            per_strs.append(f"{period} — {len(ds)} ({span})")
        parts.append("периоды: " + "; ".join(per_strs))
    return "- Метрики: " + "; ".join(parts) + "."


def _wiki_source_line(state: OrchestratorState) -> str:
    """Строка «- Wiki: …» для блока источников: что реально пришло из easyrag.

    Поля easyrag_* сбрасываются в начале каждого хода (route), поэтому строка
    отражает текущий ход, а не накопленную историю.
    """
    snippets = state.get("easyrag_snippets") or []
    if snippets:
        # Только названия «страница / раздел», без similarity и тел фрагментов:
        # блок для пользователя, технические оценки ему не нужны.
        refs = []
        for s in snippets[:5]:
            page = s.get("page_title") or s.get("slug") or "-"
            title = s.get("section_title") or s.get("anchor") or "-"
            refs.append(f"«{page} / {title}»")
        return f"- Wiki: найдено фрагментов — {len(snippets)}: " + ", ".join(refs) + "."
    if state.get("easyrag_error"):
        # Текст исключения пользователю не показываем — он остаётся в
        # easyrag_error и в трассе для диагностики.
        return "- Wiki: не удалось получить данные из базы знаний."
    query = (state.get("easyrag_query") or "").strip()
    if not query:
        return "- Wiki: не запрашивалась на этом ходе."
    if len(query) > _SOURCES_QUERY_PREVIEW:
        query = query[:_SOURCES_QUERY_PREVIEW] + "…"
    stubs = state.get("easyrag_stub_pages") or []
    if stubs:
        names = ", ".join(s.get("title") or s.get("slug") or "-" for s in stubs)
        return (
            f"- Wiki: по запросу «{query}» найдены только пустые "
            f"страницы-заглушки: {names}."
        )
    return f"- Wiki: по запросу «{query}» ничего не найдено."


def _memory_source_line(state: OrchestratorState) -> str:
    """Строка «- Долгосрочная память: …»: тот же фильтр «…отсутствует», что и
    в _memory_system_block, но пустая память здесь показывается явно."""
    if state.get("memory_error"):
        # Как и с wiki: текст ошибки — в memory_error, пользователю только факт.
        return "- Долгосрочная память: не удалось загрузить."
    ctx = (state.get("memory_context") or "").strip()
    if ctx and "отсутствует" not in ctx.lower():
        preview = ctx.replace("\n", " ")
        if len(preview) > _SOURCES_MEMORY_PREVIEW:
            preview = preview[:_SOURCES_MEMORY_PREVIEW] + "…"
        return f"- Долгосрочная память: {preview}"
    return (
        "- Долгосрочная память: отсутствует "
        "(сохранённого контекста прошлых диалогов нет)."
    )


def _describe_sources_block(state: OrchestratorState) -> str:
    """Детерминированный подблок «Исходные данные» раздела describe_answer.

    Фактура шагов пайплайна берётся из ПЕРСИСТЕНТНОГО стейта, а не из трассы:
    трасса per-turn, и шаги загрузки первого хода (оргструктура, метрики,
    память) на последующих ходах в неё уже не попадают. Точные значения без
    LLM — числа и имена не искажаются.
    """
    lines: list[str] = []
    # Толерантный парсинг (dict или JSON-строка): нормализация в load_data не
    # спасает треды, где строка уже легла в чекпоинт до неё.
    metrics = _metrics_obj(state.get("metrics"))
    if metrics is not None:
        org_parts: list[str] = []
        me = metrics.get("me") or {}
        boss_fio = str((me.get("fio") if isinstance(me, dict) else "") or "").strip()
        if boss_fio:
            org_parts.append(f"руководитель — {boss_fio}")
        names = [
            str(e.get("fio") or "").strip()
            for e in (metrics.get("employees") or [])
            if isinstance(e, dict) and str(e.get("fio") or "").strip()
        ]
        if names:
            position = str(state.get("position") or "").strip()
            pos = f" (позиция: {position})" if position else ""
            org_parts.append("в фокусе анализа — " + ", ".join(names) + pos)
        if org_parts:
            lines.append("- Оргструктура: " + "; ".join(org_parts) + ".")
        else:
            lines.append("- Оргструктура: в датасете нет ФИО.")
        lines.append(_metrics_source_line(metrics))
    else:
        err = state.get("metrics_error")
        suffix = f" ({err})" if err else ""
        lines.append(f"- Оргструктура: данные не загружены{suffix}.")
        lines.append("- Метрики: не загружены.")
    lines.append(_wiki_source_line(state))
    lines.append(_memory_source_line(state))
    return "**Исходные данные:**\n" + "\n".join(lines)


async def _with_description(
    text: str,
    state: OrchestratorState,
    config: RunnableConfig | None,
    llm: Any = None,
    question: str = "",
) -> str:
    """Дописывает раздел «Как я пришёл к выводу», если describe_answer=true.

    Раздел открывается детерминированным подблоком «Исходные данные»
    (_describe_sources_block: оргструктура, метрики, wiki, память из стейта),
    дальше — ход анализа. Основной путь — LLM-описание по трассе (какие
    инструменты, зачем, как интерпретированы данные). Фоллбэк — детерминированный
    render_trace: когда llm не передан, в трассе нет содержательных шагов
    (tool_call/kb_hit — служебные ходы вроде отмены сохранения) или LLM-вызов
    упал/вернул пусто.
    """
    if not _config_flag(config, "describe_answer", default=False):
        return text
    sources = _describe_sources_block(state)
    trace = state.get("reasoning_trace") or []
    substantive = any(s.get("kind") in ("tool_call", "kb_hit") for s in trace)
    if llm is not None and substantive:
        narrative = await _describe_trace_llm(llm, question, text, trace)
        if narrative:
            body = f"{sources}\n\n{narrative}" if sources else narrative
            return f"{text}\n\n---\n{_TRACE_SECTION_TITLE}\n\n{body}"
    section = render_trace(trace, preamble=sources)
    return f"{text}\n\n{section}" if section else text


# --- Контракт сообщений хода: промежуточные шаги + итог последним -------------
#
# За один ход граф проходит ОДИН линейный путь, а add_messages добавляет в конец,
# поэтому порядок сообщений = порядок отработки узлов. Рабочие (не-листовые) узлы
# кладут короткие «шаговые» сообщения (_STEP_KEY), а терминальные листья графа —
# итоговый ответ (_FINAL_KEY). Итог всегда оказывается последним элементом
# messages: его показывает пользователю вызывающая система (messages[-1] либо по
# флагу orchestrator_final). Шаговые сообщения вырезаются из истории, подаваемой
# в LLM (_history_for_llm), чтобы модель не имитировала их как свои прошлые реплики.
_STEP_KEY = "orchestrator_step"    # промежуточный шаг хода
_FINAL_KEY = "orchestrator_final"  # итоговый ответ хода (его показываем юзеру)


def _step_update(config: RunnableConfig | None, text: str) -> dict:
    """{'messages': [шаговое AIMessage]} либо {} если прогресс выключен/пусто."""
    text = (text or "").strip()
    if not text or not _config_flag(config, "emit_progress_messages", default=True):
        return {}
    return {"messages": [AIMessage(content=text, additional_kwargs={_STEP_KEY: True})]}


def _is_step(m: Any) -> bool:
    if isinstance(m, dict):
        return bool((m.get("additional_kwargs") or {}).get(_STEP_KEY))
    return bool(getattr(m, "additional_kwargs", None) and m.additional_kwargs.get(_STEP_KEY))


# Исходный markdown итогового ответа при answer_html=true: content сообщения —
# уже HTML, а историю для LLM собираем из этого ключа (_history_for_llm), чтобы
# модель не видела HTML-разметку и не начала её имитировать.
_MARKDOWN_KEY = "orchestrator_markdown"


def _render_answer_html(text: str) -> str:
    """Markdown итогового ответа → HTML. Детерминированно, через markdown2 —
    БЕЗ LLM (модель разметку не пишет и испортить её не может)."""
    return markdown2.markdown(
        text, extras=["tables", "fenced-code-blocks", "cuddled-lists"]
    ).strip()


def _final_message(text: str, config: RunnableConfig | None) -> AIMessage:
    """Итоговое сообщение хода — помечаем флагом, чтобы потребитель брал его явно.

    По умолчанию (answer_html=true) content — HTML из markdown2; исходный
    markdown сохраняется в additional_kwargs[_MARKDOWN_KEY]. answer_html=false
    возвращает прежнее поведение: content — сырой markdown.
    """
    kwargs: dict = {_FINAL_KEY: True}
    if _config_flag(config, "answer_html", default=True):
        kwargs[_MARKDOWN_KEY] = text
        text = _render_answer_html(text)
    return AIMessage(content=text, additional_kwargs=kwargs)


def _strip_rankings(metrics: Any) -> Any:
    """Копия датасета без индивидуальных rankings (место сотрудника в peer-группе).

    Нужна при use_peer_aggregates=false: peer-сравнение выключено целиком, а
    rankings — это те же peer-данные, только персональные. Режем НА ВХОДЕ в
    аналитик, чтобы не мутировать metrics в стейте (они переживают ходы).
    """
    def _walk(nodes: Any) -> list[Any]:
        out: list[Any] = []
        for n in nodes or []:
            if not isinstance(n, dict):
                continue
            copy = {k: v for k, v in n.items() if k != "rankings"}
            if n.get("child_metrics"):
                copy["child_metrics"] = _walk(n["child_metrics"])
            out.append(copy)
        return out

    def _person(p: Any) -> Any:
        if not isinstance(p, dict):
            return p
        return {**p, "metrics": _walk(p.get("metrics"))}

    if not isinstance(metrics, dict):
        return metrics
    result = dict(metrics)
    if result.get("me") is not None:
        result["me"] = _person(result["me"])
    result["employees"] = [_person(e) for e in (result.get("employees") or [])]
    return result


async def _run_analyzer(
    json_analyzer_graph: Any,
    metrics: Any,
    question: str,
    direction_key: str,
    aggregates: Any = None,
    use_aggregates: bool = True,
    wiki_context: str | None = None,
) -> tuple[str | None, list[dict], str | None]:
    """Единая обёртка над json_analyzer-подграфом.

    Возвращает (answer, tool_steps, error). Ошибки изолируются: при сбое/нехватке
    входов answer=None, error заполнен, конвейер продолжает работу по фоллбэку.
    aggregates опциональны (None безопасен — режим без peer-сравнения);
    use_aggregates=False дополнительно срезает персональные rankings.
    wiki_context — готовый блок wiki-сниппетов (справка об интерпретации метрик);
    None безопасен — промпты аналитика прежние.
    """
    if metrics is None:
        return None, [], "Метрики не загружены — нечем кормить json_analyzer."
    if not (question or "").strip():
        return None, [], "Пустой вопрос для аналитического запроса."
    if not direction_key:
        return None, [], "Пустой direction_key — json_analyzer не изолирует кэш."
    try:
        result = await json_analyzer_graph.ainvoke({
            "raw_json": metrics if use_aggregates else _strip_rankings(metrics),
            "raw_aggregates": aggregates if use_aggregates else None,
            "question": question,
            "direction_key": direction_key,
            "wiki_context": wiki_context,
        })
    except Exception as exc:  # noqa: BLE001 — внешний подграф (LLM/БД), сужать нечем
        return None, [], f"{type(exc).__name__}: {exc}"[:300]
    answer = result.get("answer") if isinstance(result, dict) else None
    tool_steps = result.get("tool_steps") if isinstance(result, dict) else None
    return (answer or None), (tool_steps or []), None


def _analyzer_trace_steps(tool_steps: list[dict]) -> list[TraceStep]:
    """Мапит tool_steps подграфа в TraceStep (stage='json_analyzer')."""
    steps: list[TraceStep] = []
    for ts in (tool_steps or [])[:_TRACE_TOOL_STEPS_CAP]:
        args = ", ".join(f"{k}={v}" for k, v in (ts.get("args") or {}).items())
        summary = ts.get("result_summary") or ""
        steps.append({
            "stage": "json_analyzer",
            "kind": "tool_call",
            "summary": f"{ts.get('tool')}({args}) → {summary}".strip(),
            "detail": ts,
        })
    return steps


def _metrics_system_block(state: OrchestratorState) -> str | None:
    # Контекст метрик для респондера — композиция:
    #   (1) опорный широкий разбор (metrics_summary, стабильный, ход 1);
    #   (2) свежий узкий ответ аналитика — ЯВНО подписанный своим вопросом, чтобы
    #       модель не приняла его за «все метрики».
    # Поля не пересекаются по содержимому (первичный разбор пишется только в
    # metrics_summary, см. initial_analysis), поэтому дедуп здесь не нужен.
    parts: list[str] = []
    summary = (state.get("metrics_summary") or "").strip()
    if summary:
        parts.append("Опорные факты по метрикам сотрудника (первичные):\n" + summary)

    answer = (state.get("analytics_answer") or "").strip()
    question = (state.get("analytics_question") or "").strip()
    if answer and question and not state.get("analytics_error"):
        parts.append(f"Факты по метрикам под вопрос «{question}»:\n{answer}")

    if parts:
        return "\n\n".join(parts)

    if state.get("metrics_error"):
        return f"Контекст: {state['metrics_error']}"
    if state.get("metrics") is None:
        return None
    # Разбора нет, но метрики загружены. Сырой обрезанный JSON сюда не отдаём:
    # обрезка на большом датасете даёт неверные числа. Честно говорим, что разбор
    # недоступен, и просим не выдумывать значения (пересчитается через analytics-ход).
    return (
        "Метрики сотрудника загружены, но аналитический разбор сейчас недоступен. "
        "Не приводи конкретных числовых значений по памяти; предложи уточнить "
        "вопрос или повторить запрос — он будет пересчитан аналитиком."
    )


def _wiki_snippets_text(state: OrchestratorState) -> str | None:
    """Только строки wiki-сниппетов — общий блок для респондера и аналитика.

    Респондерные инструкции (заглушки, ошибки) сюда не входят: аналитику они не
    нужны, ему уходит именно этот текст (wiki_context в json_analyzer_v5).
    """
    snippets = state.get("easyrag_snippets") or []
    if not snippets:
        return None
    lines = ["Релевантные фрагменты wiki (по направлению сотрудника):"]
    for s in snippets[:5]:
        page = s.get("page_title") or s.get("slug") or "-"
        title = s.get("section_title") or s.get("anchor") or "-"
        sim = s.get("similarity")
        sim_str = f" sim={sim:.2f}" if isinstance(sim, (int, float)) else ""
        body = (s.get("body_md") or "").strip().replace("\n", " ")
        if len(body) > _EASYRAG_SNIPPET_PREVIEW:
            body = body[:_EASYRAG_SNIPPET_PREVIEW] + "…"
        lines.append(f"- [{page} / {title}{sim_str}]: {body}")
    return "\n".join(lines)


def _easyrag_system_block(state: OrchestratorState) -> str | None:
    snippets_text = _wiki_snippets_text(state)
    if snippets_text:
        return snippets_text
    if state.get("easyrag_error"):
        return f"Контекст из wiki недоступен: {state.get('easyrag_error')}"
    stubs = state.get("easyrag_stub_pages") or []
    if stubs:
        names = ", ".join(s.get("title") or s.get("slug") or "-" for s in stubs)
        return (
            "В базе знаний по этой теме есть ПУСТАЯ страница-заглушка "
            f"(сущность заведена, но ещё не описана): {names}. Содержания по "
            "ней пока нет. Сообщи пользователю, что тема в wiki уже известна, "
            "но информация по ней пока не внесена и появится после загрузки "
            "новых источников. НЕ придумывай содержание сам."
        )
    return None


def _memory_system_block(state: OrchestratorState) -> str | None:
    # Долгосрочная память: контекст предыдущих диалогов, если он загружен и
    # непустой («…отсутствует» — маркер пустой памяти, в промпт не подмешиваем).
    memory_context = state.get("memory_context")
    if memory_context and "отсутствует" not in memory_context.lower():
        return (
            "=== Контекст предыдущих диалогов (долгосрочная память) ===\n"
            f"{memory_context}"
        )
    return None


def _strip_code_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    # отбрасываем первую строку с ``` (возможно ```json)
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _load_json(text: Any) -> Any:
    """Текст (возможно в ```-ограждении) → распарсенный JSON или None при сбое/пустоте."""
    cleaned = _strip_code_fence(text if isinstance(text, str) else str(text or ""))
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None


def _has_star_data(metrics: Any) -> bool:
    """Есть ли в датасете сотрудника звёздные поля (star_received/is_star_metric).

    Считаем по СЫРОМУ датасету тем же каноническим парсером, что и остальные
    сканы здесь, а НЕ по тексту выжимки: подпись раздела пишет модель, и она
    может её перефразировать или опустить — тогда правила про звезду молча не
    доехали бы до ответа. Любой сбой разбора → False (прежнее поведение).
    """
    try:
        rows = load_dataset_obj(metrics)
    except Exception:  # noqa: BLE001 — датасет от внешнего клиента, форма не гарантирована
        return False
    return any(
        r.get("star_received") is not None or r.get("is_star_metric") for r in rows
    )


def _collect_metric_catalog(metrics: Any) -> list[dict]:
    """Плоский каталог метрик ``[{id, metric_name, description}]`` из датасета.

    Строится через канонический ``load_dataset_obj`` (тот же парсер, что у
    json_analyzer и _distinct_metric_specs) — поэтому совпадает с реальной формой
    данных, а не с догадкой о ключах. Дедуп по ``metric_id`` (одна метрика
    повторяется в разрезах по ``element`` — id и имя у них общие). Описание нужно,
    чтобы LLM мог сопоставить разговорную формулировку с именем метрики.
    """
    try:
        rows = load_dataset_obj(metrics)
    except Exception:  # noqa: BLE001 — датасет от внешнего клиента, форма не гарантирована
        return []
    catalog: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        mid = str(r.get("metric_id") or "").strip()
        name = str(r.get("metric_name") or "").strip()
        if not mid or mid in seen:
            continue
        seen.add(mid)
        catalog.append({
            "id": mid,
            "metric_name": name,
            "description": str(r.get("metric_description") or "").strip(),
        })
    return catalog


def _format_metric_catalog(catalog: list[dict]) -> str:
    """Каталог метрик в строки ``id | название — краткое описание`` для промпта."""
    lines: list[str] = []
    for c in catalog:
        if not c.get("id"):
            continue
        desc = c.get("description") or ""
        if len(desc) > _CATALOG_DESC_PREVIEW:
            desc = desc[:_CATALOG_DESC_PREVIEW] + "…"
        suffix = f" — {desc}" if desc else ""
        lines.append(f"{c['id']} | {c['metric_name']}{suffix}")
    return "\n".join(lines) or "(каталог пуст)"


def _resolve_insight_metric(
    text: str, metric_id: str, metric_name: str, catalog: list[dict]
) -> tuple[str, str]:
    """Детерминированно проставляет ``(metric_id, metric_name)`` по каталогу.

    LLM ненадёжно мапит разговорные формулировки на терсые имена метрик, поэтому
    доводим сопоставление в коде, по убыванию надёжности:
      1) точный id из каталога → каноничное имя;
      2) точное совпадение имени (casefold);
      3) нечёткое совпадение имени (similarity_ratio ≥ порога);
      4) скан текста инсайта на вхождение имён метрик (берём самое длинное).
    Если ничего не нашлось — возвращаем как было (может остаться пустым).
    """
    if not catalog:
        return metric_id, metric_name

    name_by_id = {c["id"]: c["metric_name"] for c in catalog}
    if metric_id and metric_id in name_by_id:
        return metric_id, name_by_id[metric_id] or metric_name

    nlow = metric_name.casefold().strip()
    if nlow:
        for c in catalog:
            if (c["metric_name"] or "").casefold().strip() == nlow:
                return c["id"], c["metric_name"]
        best = max(
            catalog,
            key=lambda c: similarity_ratio(nlow, (c["metric_name"] or "").casefold()),
        )
        if similarity_ratio(nlow, (best["metric_name"] or "").casefold()) >= _METRIC_NAME_FUZZY:
            return best["id"], best["metric_name"]

    low = (text or "").casefold()
    matched: dict | None = None
    for c in catalog:
        nm = (c["metric_name"] or "").strip()
        if nm and nm.casefold() in low:
            if matched is None or len(nm) > len(matched["metric_name"]):
                matched = c
    if matched:
        return matched["id"], matched["metric_name"]

    return metric_id, metric_name


def _enforce_single_main_problem(insights: list[dict]) -> list[dict]:
    """Гарантирует СТРОГО ОДНУ ``main_problem``: лишние понижаем до ``problem``."""
    seen_main = False
    for ins in insights:
        if ins.get("type") == "main_problem":
            if seen_main:
                ins["type"] = "problem"
            else:
                seen_main = True
    return insights


def _parse_insights_json(text: Any, catalog: list[dict]) -> list[dict]:
    """Парсит ответ классификатора в список инсайтов ``{type, metric_id, metric_name, text}``.

    Принимает как ``{"insights": [...]}``, так и голый массив. Сверяет
    ``metric_id``/``metric_name`` с каталогом (восстанавливает недостающее по
    парному полю), нормализует ``type`` и число ``main_problem``.
    """
    data = _load_json(text)
    if isinstance(data, dict):
        items = data.get("insights")
    elif isinstance(data, list):
        items = data
    else:
        items = None
    if not isinstance(items, list):
        return []

    result: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text_val = str(item.get("text") or "").strip()
        if not text_val:
            continue
        itype = str(item.get("type") or "").strip()
        if itype not in _INSIGHT_TYPES:
            itype = "problem"
        mid = str(item.get("metric_id") or "").strip()
        name = str(item.get("metric_name") or "").strip()
        # Детерминированно доводим сопоставление метрики по каталогу (LLM мапит
        # разговорные формулировки на терсые имена ненадёжно).
        mid, name = _resolve_insight_metric(text_val, mid, name, catalog)
        result.append({
            "type": itype,
            "metric_id": mid,
            "metric_name": name,
            "text": text_val,
        })
        if len(result) >= _MAX_INSIGHTS:
            break
    return _enforce_single_main_problem(result)


def _parse_indices_json(text: Any, max_n: int) -> list[int]:
    data = _load_json(text)
    if not isinstance(data, dict):
        return []
    indices = data.get("indices") or []
    if not isinstance(indices, list):
        return []
    result: list[int] = []
    for x in indices:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= max_n and i not in result:
            result.append(i)
    return result


def _parse_query_list_json(text: Any, max_n: int) -> list[str]:
    """Парсит JSON-массив строк-запросов к wiki. Дедуп (без учёта регистра),
    cap max_n. Любой сбой/не-массив → []."""
    data = _load_json(text)
    if not isinstance(data, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in data:
        q = str(item or "").strip()
        if not q:
            continue
        key = q.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(q)
        if len(result) >= max_n:
            break
    return result


def _distinct_metric_specs(metrics: Any, cap: int = _WIKI_METRIC_SPECS_CAP) -> list[dict]:
    """Уникальные метрики (имя + первое непустое описание) из сырого датасета.

    Имена метрик не хардкодим — берём то, что есть в JSON (см. loader). Значения
    fact/plan/benchmark не нужны: запросы к wiki — про определения, а не числа.
    Любой сбой парсинга → []."""
    try:
        rows = load_dataset_obj(metrics)
    except Exception:  # noqa: BLE001 — датасет от внешнего клиента, форма не гарантирована
        return []
    specs: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        name = (r.get("metric_name") or "").strip()
        if not name or name in seen:
            continue
        desc = (r.get("metric_description") or "").strip()
        if len(desc) > _WIKI_METRIC_DESC_PREVIEW:
            desc = desc[:_WIKI_METRIC_DESC_PREVIEW] + "…"
        seen.add(name)
        specs.append({"name": name, "description": desc})
        if len(specs) >= cap:
            break
    return specs
