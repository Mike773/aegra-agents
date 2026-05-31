from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_gigachat import GigaChat
from sqlalchemy import select

from ..easyrag.db import session_scope
from ..easyrag.models import WikiPage
from ..shared.text_similarity import similarity_ratio
from ..shared.agent_dataset import GetBatchAgentDatasetByFiltersComponent
from ..shared.assignments_service import SendAssignmentsComponent
from ..shared.orgstructure import IsuEmployeeOrgstructureInfo
from .prompts import (
    ASSIGNMENTS_ANALYSIS_QUESTION,
    EXTRACT_ASSIGNMENTS_PROMPT,
    INITIAL_ANALYSIS_PROMPT,
    LOAD_ERROR_PROMPT,
    PROPOSE_NO_CANDIDATES_PROMPT,
    RESPONDER_PROMPT,
    ROUTER_PROMPT,
    SELECT_ASSIGNMENTS_PROMPT,
)
from .state import OrchestratorState, TraceStep

_DEFAULT_DATASET = "metrics_for_agent_analyst"
_METRICS_PREVIEW_LIMIT = 8000
_DEFAULT_EASYRAG_TOP_K = 5
_EASYRAG_SNIPPET_PREVIEW = 400
_MAX_CANDIDATES = 5
# Порог нечёткого совпадения слова запроса с заголовком заглушки.
# Ловит склонённые формы: «бабушку» ↔ «бабушка».
_STUB_MATCH_RATIO = 0.72
_STUB_LOOKUP_LIMIT = 5
# Сколько шагов tool_call максимум разворачивать в трассу из json_analyzer.
_TRACE_TOOL_STEPS_CAP = 12


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


def make_initial_analysis_node(llm: GigaChat, json_analyzer_graph: Any):
    """Первичный анализ по ПОДГОТОВЛЕННОМУ вопросу-брифингу (первое сообщение).

    Первое сообщение пользователя — заранее заданная инструкция: роль,
    методология, желаемый формат ответа. Анализ грубому JSON не доверяем —
    прогоняем брифинг через json_analyzer (видит полный датасет инструментами),
    а затем формируем ответ строго по инструкции/формату из брифинга. Если
    брифинга нет — деградируем на дефолтный INITIAL_ANALYSIS_PROMPT.
    """

    async def initial_analysis(state: OrchestratorState, config: RunnableConfig) -> dict:
        if state.get("metrics_error") or state.get("metrics") is None:
            step: TraceStep = {
                "stage": "initial",
                "kind": "error",
                "summary": state.get("metrics_error") or "Метрики не загружены.",
            }
            return {
                "messages": [AIMessage(content=LOAD_ERROR_PROMPT)],
                "reasoning_trace": _append_trace(state, [step]),
            }

        metrics = state.get("metrics")
        briefing = (state.get("briefing") or "").strip()
        direction_key = (state.get("direction_key") or "").strip()
        # Вопрос для глубокого сбора: сам брифинг, иначе фиксированный обзор.
        question = briefing or "Сделай первичный обзор ключевых метрик сотрудника."

        analysis, tool_steps, err = await _run_analyzer(
            json_analyzer_graph, metrics, question, direction_key
        )

        trace_steps: list[TraceStep] = [{
            "stage": "initial",
            "kind": "intent",
            "summary": (
                "Первый ход: выполняю подготовленное задание (брифинг)."
                if briefing else "Первый ход: первичный обзор метрик."
            ),
            "detail": {"has_briefing": bool(briefing)},
        }]
        if err:
            trace_steps.append({"stage": "initial", "kind": "error", "summary": err})
        trace_steps.extend(_analyzer_trace_steps(tool_steps))

        # Системный контекст: инструкция/формат из брифинга (или дефолт) + данные.
        parts: list[str] = [briefing or INITIAL_ANALYSIS_PROMPT]
        if analysis:
            parts.append("Данные, собранные аналитиком метрик из полного датасета:\n" + analysis)
        else:
            parts.append(_metrics_payload(metrics))
            if err:
                parts.append(f"(Глубокий анализ недоступен: {err} — опирайся на JSON выше.)")

        ai = llm.invoke([
            SystemMessage(content="\n\n".join(parts)),
            HumanMessage(content="Выполни задание по данным выше и верни ответ в требуемом формате."),
        ])
        text = ai.content if isinstance(ai.content, str) else str(ai.content)

        trace_steps.append({
            "stage": "initial",
            "kind": "decision",
            "summary": "Сформировал первичный анализ по брифингу и собранным данным.",
        })
        new_trace = _append_trace(state, trace_steps)

        out_state = {
            "messages": [AIMessage(content=_with_description(
                text, {"reasoning_trace": new_trace}, config
            ))],
            "reasoning_trace": new_trace,
            # Sticky: результат глубокого анализа доступен респондеру дальше.
            "analytics_question": question,
            "analytics_answer": analysis,
            "analytics_error": err,
        }
        return out_state

    return initial_analysis


_ROUTE_LABELS = {"analytics", "wiki", "chat", "done", "assignments"}


def make_route_node(llm: GigaChat):
    def route(state: OrchestratorState) -> dict:
        # route — первый узел каждого последующего хода: здесь СБРАСЫВАЕМ трассу
        # (она строго per-turn) и кладём первый шаг — классификацию запроса.
        last_text = _last_user_text(state)
        if not last_text:
            return {
                "intent": "chat",
                "reasoning_trace": [{
                    "stage": "route", "kind": "intent",
                    "summary": "Пустая реплика — обычный чат.",
                    "detail": {"intent": "chat"},
                }],
            }

        # Пока висит pending-список, ЛЮБАЯ реплика интерпретируется как выбор
        # по нему — это и есть «отдельная ветвь диалога». Выход из неё —
        # «никакие/отмена», что в commit_assignments отрисуется как cancel.
        if state.get("pending_assignments"):
            return {
                "intent": "assignments_select",
                "reasoning_trace": [{
                    "stage": "route", "kind": "intent",
                    "summary": "Идёт выбор поручений — реплику трактую как выбор.",
                    "detail": {"intent": "assignments_select"},
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
        system_prompt = cfg.get("system_prompt_override") or RESPONDER_PROMPT

        parts: list[str] = [system_prompt]
        # Брифинг первого хода задаёт формат/методологию — держим его на всех
        # последующих ходах, чтобы ответы пользователю шли в едином формате.
        briefing = (state.get("briefing") or "").strip()
        if briefing:
            parts.append(
                "Исходное задание (соблюдай его роль, методологию и ФОРМАТ ответа):\n"
                + briefing
            )
        metrics_block = _metrics_system_block(state)
        if metrics_block:
            parts.append(metrics_block)
        easyrag_block = _easyrag_system_block(state)
        if easyrag_block:
            parts.append(easyrag_block)
        system_text = "\n\n".join(parts)

        messages: list[Any] = [SystemMessage(content=system_text)]
        messages.extend(state.get("messages") or [])

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
            "detail": {"sources": used, "briefing": bool(briefing)},
        }
        new_trace = _append_trace(state, [decision])
        return {
            "messages": [AIMessage(content=_with_description(
                text, {"reasoning_trace": new_trace}, config
            ))],
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

        answer, tool_steps, err = await _run_analyzer(
            json_analyzer_graph, metrics, last_text, direction_key
        )
        if err and metrics is None:
            err = state.get("metrics_error") or err

        steps = _analyzer_trace_steps(tool_steps)
        if err:
            steps.append({"stage": "json_analyzer", "kind": "error", "summary": err})
        elif answer:
            steps.append({
                "stage": "json_analyzer", "kind": "decision",
                "summary": "Аналитик метрик собрал данные и сформировал ответ.",
            })

        return {
            "analytics_question": last_text,
            "analytics_answer": answer,
            "analytics_error": err,
            "reasoning_trace": _append_trace(state, steps),
        }

    return call_json_analyzer


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
            }

    return call_easyrag


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


def make_extract_assignments_node(llm: GigaChat, json_analyzer_graph: Any):
    """Извлекает кандидатов-поручения из анализа проблемных зон сотрудника.

    Запускается автоматически после ``initial_analysis`` и повторно — когда
    роутер ловит явный intent ``assignments``. Источник анализа — подграф
    ``json_analyzer``, который видит ПОЛНЫЙ датасет через инструменты (SQLite +
    pgvector), а не обрезанный под ``_METRICS_PREVIEW_LIMIT`` JSON. Анализ затем
    структурируется вторым LLM-вызовом в JSON-поручения.

    Фоллбэк: если json_analyzer недоступен (нет ``direction_key``, упал или
    вернул пусто) — деградируем на прежний путь по обрезанному JSON, чтобы не
    терять функциональность. Пустой список кандидатов — нормальный исход.
    """

    async def extract(state: OrchestratorState) -> dict:
        metrics = state.get("metrics")
        if metrics is None or state.get("metrics_error"):
            return {
                "candidate_assignments": [],
                "pending_assignments": [],
                "reasoning_trace": _append_trace(state, [{
                    "stage": "assignments", "kind": "error",
                    "summary": "Метрики не загружены — поручения не извлекаю.",
                }]),
            }

        direction_key = (state.get("direction_key") or "").strip()
        analysis, tool_steps, _err = await _run_analyzer(
            json_analyzer_graph, metrics, ASSIGNMENTS_ANALYSIS_QUESTION, direction_key
        )
        if analysis:
            context = f"Анализ проблемных зон сотрудника:\n{analysis}"
        else:
            # Фоллбэк на сырой (обрезанный) JSON + предыдущий анализ.
            context = _metrics_payload(metrics)
            last_ai = _last_ai_text(state)
            if last_ai:
                context = f"{context}\n\nПредыдущий анализ:\n{last_ai}"

        try:
            ai = llm.invoke([
                SystemMessage(content=EXTRACT_ASSIGNMENTS_PROMPT),
                HumanMessage(content=context),
            ])
            candidates = _parse_assignments_json(ai.content)
        except Exception:  # noqa: BLE001 — LLM-вызов, сужать нечем
            candidates = []

        steps = _analyzer_trace_steps(tool_steps)
        steps.append({
            "stage": "assignments", "kind": "decision",
            "summary": f"Проанализировал проблемные зоны, выделил кандидатов: {len(candidates)}.",
            "detail": {"candidates": [c.get("title") for c in candidates]},
        })
        return {
            "candidate_assignments": candidates,
            "pending_assignments": candidates,
            "reasoning_trace": _append_trace(state, steps),
        }

    return extract


def make_propose_assignments_node():
    """Показывает пользователю нумерованный список кандидатов и просит выбор.

    Если кандидатов нет — пишет короткое сообщение и сбрасывает pending,
    чтобы роутер вернулся к обычному классифицированию следующей реплики.
    """

    def propose(state: OrchestratorState) -> dict:
        pending = state.get("pending_assignments") or []
        if not pending:
            return {
                "messages": [AIMessage(content=PROPOSE_NO_CANDIDATES_PROMPT)],
                "pending_assignments": [],
            }

        lines = ["Выделил кандидатов на поручения сотруднику:"]
        for i, p in enumerate(pending, 1):
            lines.append(f"{i}. {p.get('title', '').strip() or '(без названия)'}")
            problem = (p.get("problem") or "").strip()
            if problem:
                lines.append(f"   Проблема: {problem}")
            action = (p.get("action") or "").strip()
            if action:
                lines.append(f"   Действие: {action}")
        lines.append("")
        lines.append(
            "Какие зафиксировать? Напишите номера через запятую "
            "(например, «1, 3»), «все» или «никакие»."
        )
        return {"messages": [AIMessage(content="\n".join(lines))]}

    return propose


def make_select_assignments_node(llm: GigaChat):
    """Парсит ответ пользователя в список индексов кандидатов.

    Непонятный ответ → пустой выбор → ``commit_assignments`` отработает как
    cancel и сбросит pending, чтобы диалог не залипал.
    """

    def select(state: OrchestratorState) -> dict:
        pending = state.get("pending_assignments") or []
        user_text = _last_user_text(state)
        if not pending or not user_text:
            return {"selected_assignments": []}

        listing = "\n".join(
            f"{i}. {p.get('title', '').strip() or '(без названия)'}"
            for i, p in enumerate(pending, 1)
        )
        try:
            ai = llm.invoke([
                SystemMessage(content=SELECT_ASSIGNMENTS_PROMPT),
                HumanMessage(
                    content=f"Кандидаты (N={len(pending)}):\n{listing}\n\n"
                    f"Ответ пользователя: {user_text}"
                ),
            ])
            indices = _parse_indices_json(ai.content, max_n=len(pending))
        except Exception:  # noqa: BLE001 — LLM-вызов, сужать нечем
            indices = []

        selected = [pending[i - 1] for i in indices]
        return {
            "selected_assignments": selected,
            "reasoning_trace": _append_trace(state, [{
                "stage": "assignments", "kind": "decision",
                "summary": f"Распознал выбор пользователя: позиции {indices or '—'}.",
                "detail": {"indices": indices},
            }]),
        }

    return select


def make_commit_assignments_node():
    """Отправляет выбранные поручения в mock-сервис и закрывает ветку выбора.

    Всегда чистит ``pending_assignments``/``selected_assignments`` — независимо
    от исхода: пустой выбор = cancel, ошибка отправки = сообщение и сброс,
    успех = подтверждение и обновлённый ``last_committed_assignments``.
    """

    def commit(state: OrchestratorState, config: RunnableConfig) -> dict:
        selected = state.get("selected_assignments") or []
        boss = (state.get("boss_tabnum") or "").strip()
        employee = (state.get("employee_tabnum") or "").strip()

        if not selected:
            new_trace = _append_trace(state, [{
                "stage": "assignments", "kind": "decision",
                "summary": "Выбор пуст — поручения не фиксирую (отмена).",
            }])
            text = ("Хорошо, поручения сейчас не фиксирую. "
                    "Если передумаете — скажите «оформи поручения».")
            return {
                "messages": [AIMessage(content=_with_description(
                    text, {"reasoning_trace": new_trace}, config))],
                "pending_assignments": [],
                "selected_assignments": [],
                "reasoning_trace": new_trace,
            }

        try:
            component = SendAssignmentsComponent(
                boss_tabnum=boss,
                employee_tabnum=employee,
                assignments=selected,
            )
            component.submit()
            committed = selected
            err: str | None = None
        except Exception as exc:  # noqa: BLE001 — внешний клиент, сужать нечем
            committed = []
            err = f"{type(exc).__name__}: {exc}"[:300]

        if err:
            text = f"Не удалось отправить поручения: {err}"
            step: TraceStep = {"stage": "assignments", "kind": "error",
                               "summary": f"Ошибка отправки поручений: {err}"}
        else:
            lines = [
                f"Зафиксировал {len(committed)} поручение(й) "
                f"для сотрудника {employee} (от руководителя {boss}):"
            ]
            for s in committed:
                lines.append(f"- {s.get('title', '').strip() or '(без названия)'}")
            text = "\n".join(lines)
            step = {"stage": "assignments", "kind": "decision",
                    "summary": f"Отправил {len(committed)} поручение(й) в сервис.",
                    "detail": {"committed": [s.get("title") for s in committed]}}

        new_trace = _append_trace(state, [step])
        return {
            "messages": [AIMessage(content=_with_description(
                text, {"reasoning_trace": new_trace}, config))],
            "pending_assignments": [],
            "selected_assignments": [],
            "last_committed_assignments": committed,
            "reasoning_trace": new_trace,
        }

    return commit


def need_load(state: OrchestratorState) -> str:
    return "route" if state.get("loaded") else "load_data"


def after_route(state: OrchestratorState) -> str:
    intent = state.get("intent")
    if intent == "analytics":
        return "call_json_analyzer"
    if intent == "wiki":
        return "call_easyrag"
    if intent == "assignments":
        return "extract_assignments"
    if intent == "assignments_select":
        return "select_assignments"
    # chat, done и любой неожиданный intent — прямо к респондеру без подграфов.
    # done тоже идёт сюда: респондер коротко прощается (RESPONDER_PROMPT), но
    # диалог не закрывается — пользователь может продолжить следующим вызовом.
    return "respond"


def _last_user_text(state: OrchestratorState) -> str:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def _last_ai_text(state: OrchestratorState) -> str:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, AIMessage):
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


def _metrics_payload(metrics: Any) -> str:
    try:
        payload = json.dumps(metrics, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        payload = repr(metrics)
    return f"JSON с метриками сотрудника:\n{payload[:_METRICS_PREVIEW_LIMIT]}"


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


def _describe_enabled(config: RunnableConfig | None) -> bool:
    """Флаг describe_answer из configurable (bool или строка 'true'/'1'/'yes'/'да')."""
    cfg = (config or {}).get("configurable") or {}
    val = cfg.get("describe_answer")
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


def render_trace(trace: list[TraceStep]) -> str:
    """Детерминированный markdown-раздел «Как я пришёл к выводу» из трассы.

    Без LLM — строго по накопленным шагам, поэтому без галлюцинаций (Блок B.3.a).
    """
    steps = [s for s in (trace or []) if (s.get("summary") or "").strip()]
    if not steps:
        return ""
    lines = ["---", "### Как я пришёл к выводу", ""]
    for i, step in enumerate(steps, 1):
        label = _KIND_LABELS.get(step.get("kind", ""), step.get("stage") or "Шаг")
        lines.append(f"{i}. **{label}.** {step['summary'].strip()}")
    return "\n".join(lines)


def _with_description(
    text: str, state: OrchestratorState, config: RunnableConfig | None
) -> str:
    """Дописывает раздел «Как я пришёл к выводу», если describe_answer=true."""
    if not _describe_enabled(config):
        return text
    section = render_trace(state.get("reasoning_trace") or [])
    return f"{text}\n\n{section}" if section else text


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
    # Sticky-приоритет: если json_analyzer когда-либо в этом thread'е дал
    # ответ без ошибки — респондер видит именно его, а не сырой JSON. Это
    # экономит токены и снимает обрезание под _METRICS_PREVIEW_LIMIT.
    if state.get("analytics_answer") and not state.get("analytics_error"):
        return f"Ответ аналитика метрик:\n{state['analytics_answer']}"
    if state.get("metrics_error"):
        return f"Контекст: {state['metrics_error']}"
    metrics = state.get("metrics")
    if metrics is None:
        return None
    return _metrics_payload(metrics)


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


def _parse_assignments_json(text: Any) -> list[dict]:
    raw = text if isinstance(text, str) else str(text or "")
    cleaned = _strip_code_fence(raw)
    if not cleaned:
        return []
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    result: list[dict] = []
    for item in data[:_MAX_CANDIDATES]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        result.append({
            "title": title,
            "problem": str(item.get("problem") or "").strip(),
            "action": str(item.get("action") or "").strip(),
        })
    return result


def _parse_indices_json(text: Any, max_n: int) -> list[int]:
    raw = text if isinstance(text, str) else str(text or "")
    cleaned = _strip_code_fence(raw)
    if not cleaned:
        return []
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return []
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
