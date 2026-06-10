from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_gigachat import GigaChat
from sqlalchemy import select

from ..easyrag.db import session_scope
from ..easyrag.gap import record_gap
from ..easyrag.models import WikiPage
from ..gap_resolver.judge import answer_in_sources
from ..json_analyzer_v2.loader import load_dataset_obj
from ..shared.text_similarity import similarity_ratio
from ..shared.agent_dataset import GetBatchAgentDatasetByFiltersComponent
from ..shared.assignments_service import SendAssignmentsComponent
from ..shared.orgstructure import IsuEmployeeOrgstructureInfo
from .prompts import (
    ASK_QUESTION_PROMPT,
    BUSINESS_SYSTEM_PROMPT,
    CLASSIFY_INSIGHTS_PROMPT,
    CONFIRM_PROMPT,
    FORM_INSIGHTS_EMPTY,
    FORM_INSIGHTS_INTRO,
    FORM_INSIGHTS_OUTRO,
    INITIAL_TASK_HINT,
    LOAD_ERROR_PROMPT,
    RESPONDER_TASK_HINT,
    ROUTER_PROMPT,
    SAVE_CANCEL_PROMPT,
    SAVE_DONE_PROMPT,
    SAVE_ERROR_PROMPT,
    WIKI_QUERIES_PROMPT,
)
from .state import OrchestratorState, TraceStep

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
    def load_data(state: OrchestratorState, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable") or {}
        boss = (cfg.get("boss_tabnum") or "").strip()
        employee = (cfg.get("employee_tabnum") or "").strip()
        position = cfg.get("position")
        dataset_name = cfg.get("dataset_name") or _DEFAULT_DATASET

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
                "reasoning_trace": [],
                "metrics_error": (
                    "В configurable нет boss_tabnum/employee_tabnum: "
                    f"boss_tabnum={boss!r}, employee_tabnum={employee!r}."
                ),
            }

        orgstructure = IsuEmployeeOrgstructureInfo(
            manager_id=boss,
            position=position,
            employee_id=employee,
        )
        direction_key = orgstructure.direction_key()
        try:
            component = GetBatchAgentDatasetByFiltersComponent(
                dataset_name=dataset_name,
                filters=orgstructure.combined_json(),
            )
            metrics = component.build_json_output()
            error: str | None = None
        except Exception as exc:  # noqa: BLE001 — внешний клиент, сужать нечем
            metrics = None
            error = f"Ошибка при загрузке датасета {dataset_name!r}: {exc}"

        return {
            "boss_tabnum": boss,
            "employee_tabnum": employee,
            "position": position,
            "direction_key": direction_key,
            "metrics": metrics,
            "metrics_error": error,
            "briefing": briefing,
            "reasoning_trace": [],
            "loaded": True,
        }

    return load_data


def _org_structure_block(state: OrchestratorState) -> str | None:
    """Краткий блок «кого анализируем» из датасета (me/employees) + позиция.

    {me}/{employees} бизнес-спецификации → руководитель и сотрудник(и) в фокусе.
    Только ФИО и позиция (не метрики — они идут отдельным блоком). Имена нужны,
    чтобы модель писала «у Иванова…», а не безлично. Источник внутренних названий
    пользователю не раскрывается (об этом сказано в системном промпте).
    """
    metrics = state.get("metrics")
    if not isinstance(metrics, dict):
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


# Строка варианта из блока «Что делаем дальше?»: «1.»/«2)»/«3 ...».
_OPTION_LINE_RE = re.compile(r"^\s*([123])[.)]\s*(.+?)\s*$")
# Реплики-выбор «второго варианта» без своего вопроса.
_PICK_SECOND = {"2", "2.", "2)", "второй", "второе", "вариант 2", "вариант №2"}


def _parse_continuation_options(text: Any) -> list[str]:
    """Достаёт три варианта из блока «Что делаем дальше?» ответа модели.

    Возвращает [opt1, opt2, opt3] (недостающие — пустые строки) либо [], если
    блок не распознан. Нужно, чтобы разрешить выбор «2» в конкретное направление
    доп. анализа (его текст модель сформулировала сама).
    """
    if not isinstance(text, str) or not text.strip():
        return []
    opts: dict[int, str] = {}
    for line in text.splitlines():
        m = _OPTION_LINE_RE.match(line)
        if m:
            idx = int(m.group(1))
            if idx not in opts:
                opts[idx] = m.group(2).strip()
    if not opts:
        return []
    return [opts.get(1, ""), opts.get(2, ""), opts.get(3, "")]


def _resolve_more_analysis_question(state: OrchestratorState, last_text: str) -> str:
    """Если реплика — голый выбор «2», подставляем текст 2-го варианта.

    Модель сама сформулировала направление доп. анализа во 2-м пункте блока
    продолжения; пользователь ответил «2» — разворачиваем его в это направление,
    иначе аналитику нечего анализировать. Свой вопрос пользователя не трогаем.
    """
    options = state.get("pending_options") or []
    t = (last_text or "").strip().casefold()
    if len(options) >= 2 and options[1] and t in _PICK_SECOND:
        return options[1]
    return last_text


def make_initial_analysis_node(llm: GigaChat, json_analyzer_graph: Any):
    """Первичный многоуровневый разбор по бизнес-спецификации (первый ход).

    Роль/методологию/формат задаёт BUSINESS_SYSTEM_PROMPT (наш, не пользователя).
    Первое сообщение — лишь триггер («Что происходит?»). Сырому JSON не доверяем —
    факты по метрикам собирает json_analyzer_v2 (без аналитики по бенчмарку,
    pop только у метрик с планом). Ответ обязан завершаться блоком «Что делаем
    дальше?» (1/2/3) — его варианты парсим в pending_options.
    """

    async def initial_analysis(state: OrchestratorState, config: RunnableConfig) -> dict:
        if state.get("metrics_error") or state.get("metrics") is None:
            step: TraceStep = {
                "stage": "initial",
                "kind": "error",
                "summary": state.get("metrics_error") or "Метрики не загружены.",
            }
            return {
                "messages": [_final_message(LOAD_ERROR_PROMPT)],
                "reasoning_trace": _append_trace(state, [step]),
            }

        metrics = state.get("metrics")
        briefing = (state.get("briefing") or "").strip()
        direction_key = (state.get("direction_key") or "").strip()
        # Вопрос для глубокого сбора фактов: сам триггер, иначе общий обзор.
        question = briefing or "Сделай первичный обзор ключевых метрик сотрудника."

        analysis, tool_steps, err = await _run_analyzer(
            json_analyzer_graph, metrics, question, direction_key
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
        parts.append(INITIAL_TASK_HINT)

        human = briefing or "Что происходит с показателями сотрудника?"
        ai = llm.invoke([
            SystemMessage(content="\n\n".join(parts)),
            HumanMessage(content=human),
        ])
        text = ai.content if isinstance(ai.content, str) else str(ai.content)

        trace_steps.append({
            "stage": "initial",
            "kind": "decision",
            "summary": "Сформировал первичный разбор и предложил варианты продолжения.",
        })
        new_trace = _append_trace(state, trace_steps)

        out_state = {
            "messages": [_final_message(_with_description(
                text, {"reasoning_trace": new_trace}, config
            ))],
            "reasoning_trace": new_trace,
            "pending_options": _parse_continuation_options(text),
        }
        # Опорный широкий разбор — единственное sticky-поле первого хода, ставится
        # ОДИН раз и не перезаписывается узкими analytics-ходами.
        if analysis and not err:
            out_state["metrics_summary"] = analysis
        return out_state

    return initial_analysis


_ROUTE_LABELS = {
    "analytics", "more_analysis", "wiki", "ask_question", "chat", "done", "finish",
}
# Ответ на «Все верно?» → следующий intent завершения.
_CONFIRM_INTENT = {
    "confirm": "finish_save",
    "edit": "finish_reform",
    "cancel": "finish_cancel",
}


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

        # Завершение анализа: показали сформированные инсайты и ждём «Все верно?».
        # Пока ждём — ЛЮБАЯ реплика трактуется как ответ на подтверждение
        # (confirm/edit/cancel), а не как обычный запрос. Это «отдельная ветвь».
        if state.get("pending_confirmation"):
            verdict = "confirm"
            try:
                ai = llm.invoke([
                    SystemMessage(content=CONFIRM_PROMPT),
                    HumanMessage(content=last_text),
                ])
                v = (ai.content or "").strip().lower()
                if v in _CONFIRM_INTENT:
                    verdict = v
            except Exception:  # noqa: BLE001 — LLM-вызов, сужать нечем
                verdict = "confirm"
            intent = _CONFIRM_INTENT[verdict]
            return {
                **_EASYRAG_RESET,
                "intent": intent,
                "reasoning_trace": [{
                    "stage": "route", "kind": "intent",
                    "summary": f"Подтверждение сохранения: «{verdict}».",
                    "detail": {"intent": intent, "verdict": verdict},
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
    def respond(state: OrchestratorState, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable") or {}

        # Вариант 3 выбран, но свой вопрос ещё не задан — приглашаем его задать,
        # без обращения к LLM и без блока продолжения.
        if state.get("intent") == "ask_question":
            new_trace = _append_trace(state, [{
                "stage": "respond", "kind": "decision",
                "summary": "Пользователь хочет задать свой вопрос — приглашаю сформулировать.",
            }])
            return {
                "messages": [_final_message(ASK_QUESTION_PROMPT)],
                "reasoning_trace": new_trace,
            }

        # Роль/методологию/формат задаёт бизнес-промпт (override имеет приоритет).
        system_prompt = cfg.get("system_prompt_override") or BUSINESS_SYSTEM_PROMPT

        parts: list[str] = [system_prompt]
        org_block = _org_structure_block(state)
        if org_block:
            parts.append(org_block)
        metrics_block = _metrics_system_block(state)
        if metrics_block:
            parts.append(metrics_block)
        easyrag_block = _easyrag_system_block(state)
        if easyrag_block:
            parts.append(easyrag_block)
        parts.append(RESPONDER_TASK_HINT)
        system_text = "\n\n".join(parts)

        messages: list[Any] = [SystemMessage(content=system_text)]
        messages.extend(_history_for_llm(state.get("messages") or []))

        ai = llm.invoke(messages)
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
        # Парсим предложенные варианты продолжения (для разрешения выбора «2»).
        # Если их нет (короткий chat/прощание) — pending_options обнуляем.
        return {
            "messages": [_final_message(_with_description(
                text, {"reasoning_trace": new_trace}, config
            ))],
            "reasoning_trace": new_trace,
            "pending_options": _parse_continuation_options(text),
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

        # Выбор «2» (доп. анализ) разворачиваем в направление из 2-го варианта,
        # которое модель сформулировала сама; свой вопрос пользователя не трогаем.
        base_q = (
            _resolve_more_analysis_question(state, last_text)
            if state.get("intent") == "more_analysis" else last_text
        )
        # В аналитик уходит вопрос, дообогащённый контекстом диалога (чтобы
        # разрешить «эти/западающие/те»); в state сохраняем сырой last_text.
        enriched_q = _question_with_dialogue_context(state, base_q)
        answer, tool_steps, err = await _run_analyzer(
            json_analyzer_graph, metrics, enriched_q, direction_key
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

        queries = _generate_wiki_queries(
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


def make_form_insights_node(llm: GigaChat):
    """post_insights(action="form"): классифицирует разбор диалога в инсайты и
    показывает их руководителю на подтверждение «Все верно?» — БЕЗ сохранения.

    Источник фактов — накопленные итоговые ответы аналитика в диалоге (то, что
    реально обсуждалось) плюс опорный разбор. При повторной форме (после правки)
    подмешиваем пожелание руководителя из последней реплики. Пустой результат —
    нормальный исход: говорим, что фиксировать нечего, и снимаем ожидание.
    """

    async def form_insights(state: OrchestratorState, config: RunnableConfig) -> dict:
        metrics = state.get("metrics")
        if metrics is None or state.get("metrics_error"):
            return {
                "messages": [_final_message(FORM_INSIGHTS_EMPTY)],
                "candidate_assignments": [],
                "pending_confirmation": False,
                "reasoning_trace": _append_trace(state, [{
                    "stage": "assignments", "kind": "error",
                    "summary": "Метрики не загружены — выводы не формирую.",
                }]),
            }

        catalog = _collect_metric_catalog(metrics)
        answers = _gather_agent_answers(state) or (state.get("metrics_summary") or "").strip()
        # Повторная форма (после правки): учитываем пожелание руководителя.
        correction = (
            _last_user_text(state).strip() if state.get("pending_confirmation") else ""
        )
        if not answers:
            return {
                "messages": [_final_message(FORM_INSIGHTS_EMPTY)],
                "candidate_assignments": [],
                "pending_confirmation": False,
                "reasoning_trace": _append_trace(state, [{
                    "stage": "assignments", "kind": "decision",
                    "summary": "В диалоге ещё нет разбора — нечего фиксировать.",
                }]),
            }

        ctx_parts = [
            f"Каталог метрик (id — название):\n{_format_metric_catalog(catalog)}",
            f"Разбор аналитика в диалоге:\n{answers}",
        ]
        if correction:
            ctx_parts.append(
                f"Пожелание руководителя по корректировке выводов:\n{correction}"
            )
        try:
            ai = llm.invoke([
                SystemMessage(content=CLASSIFY_INSIGHTS_PROMPT),
                HumanMessage(content="\n\n".join(ctx_parts)),
            ])
            insights = _parse_insights_json(ai.content, catalog)
        except Exception:  # noqa: BLE001 — LLM-вызов, сужать нечем
            insights = []

        if not insights:
            return {
                "messages": [_final_message(FORM_INSIGHTS_EMPTY)],
                "candidate_assignments": [],
                "pending_confirmation": False,
                "reasoning_trace": _append_trace(state, [{
                    "stage": "assignments", "kind": "decision",
                    "summary": "Классификация не дала выводов для фиксации.",
                }]),
            }

        # Показ инсайтов прозой (не сырой JSON), маркированным списком + «Все верно?».
        lines = [FORM_INSIGHTS_INTRO, ""]
        for ins in insights:
            label = _INSIGHT_TYPE_LABELS.get(ins.get("type"), "")
            name = (ins.get("metric_name") or "").strip()
            head = " — ".join(p for p in (label, name) if p)
            body = (ins.get("text") or "").strip()
            lines.append(f"• {head}: {body}" if head else f"• {body}")
        lines.append("")
        lines.append(FORM_INSIGHTS_OUTRO)
        text = "\n".join(lines)

        new_trace = _append_trace(state, [{
            "stage": "assignments", "kind": "decision",
            "summary": f"Сформировал выводы для сохранения: {len(insights)}.",
            "detail": {"insights": [
                {"type": i.get("type"), "metric_name": i.get("metric_name")}
                for i in insights
            ]},
        }])
        return {
            "messages": [_final_message(_with_description(
                text, {"reasoning_trace": new_trace}, config))],
            "candidate_assignments": insights,
            "pending_confirmation": True,
            "reasoning_trace": new_trace,
        }

    return form_insights


def make_save_insights_node():
    """post_insights(action="save"): по подтверждению сохраняет сформированные
    инсайты в сервис; по отмене — ничего не пишет. Всегда снимает ожидание
    подтверждения и чистит корзину.
    """

    def save_insights(state: OrchestratorState, config: RunnableConfig) -> dict:
        # Отмена сохранения (cancel) — ничего не пишем.
        if state.get("intent") == "finish_cancel":
            new_trace = _append_trace(state, [{
                "stage": "assignments", "kind": "decision",
                "summary": "Руководитель отменил сохранение — ничего не фиксирую.",
            }])
            return {
                "messages": [_final_message(_with_description(
                    SAVE_CANCEL_PROMPT, {"reasoning_trace": new_trace}, config))],
                "candidate_assignments": [],
                "pending_confirmation": False,
                "reasoning_trace": new_trace,
            }

        selected = state.get("candidate_assignments") or []
        employee = (state.get("employee_tabnum") or "").strip()
        direction_key = (state.get("direction_key") or "").strip()
        if not selected:
            new_trace = _append_trace(state, [{
                "stage": "assignments", "kind": "decision",
                "summary": "Нет сформированных выводов — сохранять нечего.",
            }])
            return {
                "messages": [_final_message(FORM_INSIGHTS_EMPTY)],
                "candidate_assignments": [],
                "pending_confirmation": False,
                "reasoning_trace": new_trace,
            }

        try:
            component = SendAssignmentsComponent(
                employee_tabnum=employee,
                direction_key=direction_key,
                insights=selected,
            )
            component.submit()
            committed = selected
            err: str | None = None
        except Exception as exc:  # noqa: BLE001 — внешний клиент, сужать нечем
            committed = []
            err = f"{type(exc).__name__}: {exc}"[:300]

        if err:
            text = SAVE_ERROR_PROMPT.format(err=err)
            step: TraceStep = {"stage": "assignments", "kind": "error",
                               "summary": f"Ошибка сохранения выводов: {err}"}
        else:
            text = SAVE_DONE_PROMPT
            step = {"stage": "assignments", "kind": "decision",
                    "summary": f"Сохранил {len(committed)} вывод(ов) в сервис.",
                    "detail": {"committed": [s.get("metric_name") for s in committed]}}

        new_trace = _append_trace(state, [step])
        return {
            "messages": [_final_message(_with_description(
                text, {"reasoning_trace": new_trace}, config))],
            "candidate_assignments": [],
            "pending_confirmation": False,
            "last_committed_assignments": committed,
            "reasoning_trace": new_trace,
        }

    return save_insights


def need_load(state: OrchestratorState) -> str:
    return "route" if state.get("loaded") else "load_data"


def after_route(state: OrchestratorState) -> str:
    intent = state.get("intent")
    # «2»/доп. анализ и конкретный вопрос по метрикам — через аналитика.
    if intent in ("analytics", "more_analysis"):
        return "call_json_analyzer"
    if intent == "wiki":
        return "call_easyrag"
    # Завершение анализа (post_insights): форма/переформа → показ на подтверждение.
    if intent in ("finish", "finish_reform"):
        return "form_insights"
    # Ответ на «Все верно?»: сохранить или отменить.
    if intent in ("finish_save", "finish_cancel"):
        return "save_insights"
    # ask_question, chat, done и любой неожиданный intent — к респондеру.
    # ask_question: респондер приглашает задать вопрос; done: коротко прощается.
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


def render_trace(trace: list[TraceStep]) -> str:
    """Детерминированный markdown-раздел «Как я пришёл к выводу» из трассы.

    Без LLM — строго по накопленным шагам, поэтому без галлюцинаций (Блок B.3.a).
    """
    steps = [s for s in (trace or []) if (s.get("summary") or "").strip()]
    if not steps:
        return ""
    lines = ["---", _TRACE_SECTION_TITLE, ""]
    for i, step in enumerate(steps, 1):
        label = _KIND_LABELS.get(step.get("kind", ""), step.get("stage") or "Шаг")
        lines.append(f"{i}. **{label}.** {step['summary'].strip()}")
    return "\n".join(lines)


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
            out.append(AIMessage(content=_strip_trace_section(m.content)))
        else:
            out.append(m)
    return out


def _with_description(
    text: str, state: OrchestratorState, config: RunnableConfig | None
) -> str:
    """Дописывает раздел «Как я пришёл к выводу», если describe_answer=true."""
    if not _config_flag(config, "describe_answer", default=False):
        return text
    section = render_trace(state.get("reasoning_trace") or [])
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


def _final_message(text: str) -> AIMessage:
    """Итоговое сообщение хода — помечаем флагом, чтобы потребитель брал его явно."""
    return AIMessage(content=text, additional_kwargs={_FINAL_KEY: True})


async def _run_analyzer(
    json_analyzer_graph: Any, metrics: Any, question: str, direction_key: str
) -> tuple[str | None, list[dict], str | None]:
    """Единая обёртка над json_analyzer-подграфом.

    Возвращает (answer, tool_steps, error). Ошибки изолируются: при сбое/нехватке
    входов answer=None, error заполнен, конвейер продолжает работу по фоллбэку.
    """
    if metrics is None:
        return None, [], "Метрики не загружены — нечем кормить json_analyzer."
    if not (question or "").strip():
        return None, [], "Пустой вопрос для аналитического запроса."
    if not direction_key:
        return None, [], "Пустой direction_key — json_analyzer не изолирует кэш."
    try:
        result = await json_analyzer_graph.ainvoke({
            "raw_json": metrics,
            "question": question,
            "direction_key": direction_key,
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
        parts.append("Опорный разбор метрик сотрудника (первичный):\n" + summary)

    answer = (state.get("analytics_answer") or "").strip()
    question = (state.get("analytics_question") or "").strip()
    if answer and question and not state.get("analytics_error"):
        parts.append(f"Ответ аналитика на вопрос «{question}»:\n{answer}")

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


def _easyrag_system_block(state: OrchestratorState) -> str | None:
    snippets = state.get("easyrag_snippets") or []
    err = state.get("easyrag_error")
    if not snippets:
        if err:
            return f"Контекст из wiki недоступен: {err}"
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


def _gather_agent_answers(state: OrchestratorState) -> str:
    """Накопленные ИТОГОВЫЕ ответы агента в диалоге (без шагов и трассы).

    Источник фактов для классификации инсайтов: классифицируем то, что реально
    обсуждалось. Шаговые сообщения (_STEP_KEY) и раздел трассы отбрасываем.
    """
    answers: list[str] = []
    for m in state.get("messages") or []:
        if isinstance(m, AIMessage) and not _is_step(m):
            txt = _strip_trace_section(_plain_text(m)).strip()
            if txt:
                answers.append(txt)
    return "\n\n".join(answers)


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
