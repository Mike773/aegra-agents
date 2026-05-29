from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_gigachat import GigaChat
from langgraph.types import interrupt

from ..shared.agent_dataset import GetBatchAgentDatasetByFiltersComponent
from ..shared.assignments_service import SendAssignmentsComponent
from ..shared.orgstructure import IsuEmployeeOrgstructureInfo
from .prompts import (
    ASK_USER_PROMPT,
    EXTRACT_ASSIGNMENTS_PROMPT,
    INITIAL_ANALYSIS_PROMPT,
    LOAD_ERROR_PROMPT,
    PROPOSE_NO_CANDIDATES_PROMPT,
    RESPONDER_PROMPT,
    ROUTER_PROMPT,
    SELECT_ASSIGNMENTS_PROMPT,
)
from .state import OrchestratorState

_DEFAULT_DATASET = "metrics_for_agent_analyst"
_METRICS_PREVIEW_LIMIT = 8000
_DEFAULT_EASYRAG_TOP_K = 5
_EASYRAG_SNIPPET_PREVIEW = 400
_MAX_CANDIDATES = 5


def make_load_data_node():
    def load_data(state: OrchestratorState, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable") or {}
        boss = (cfg.get("boss_tabnum") or "").strip()
        employee = (cfg.get("employee_tabnum") or "").strip()
        position = cfg.get("position")
        dataset_name = cfg.get("dataset_name") or _DEFAULT_DATASET

        if not boss or not employee:
            return {
                "loaded": True,
                "metrics": None,
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
            "loaded": True,
        }

    return load_data


def make_initial_analysis_node(llm: GigaChat):
    def initial_analysis(state: OrchestratorState) -> dict:
        if state.get("metrics_error") or state.get("metrics") is None:
            return {"messages": [AIMessage(content=LOAD_ERROR_PROMPT)]}

        ai = llm.invoke([
            SystemMessage(content=INITIAL_ANALYSIS_PROMPT),
            HumanMessage(content=_metrics_payload(state.get("metrics"))),
        ])
        return {"messages": [ai]}

    return initial_analysis


def make_ask_user_node():
    def ask_user(state: OrchestratorState) -> dict:
        user_text = interrupt({"prompt": ASK_USER_PROMPT})
        return {"messages": [HumanMessage(content=str(user_text))]}

    return ask_user


_ROUTE_LABELS = {"analytics", "wiki", "chat", "done", "assignments"}


def make_route_node(llm: GigaChat):
    def route(state: OrchestratorState) -> dict:
        last_text = _last_user_text(state)
        if not last_text:
            return {"intent": "chat"}

        # Пока висит pending-список, ЛЮБАЯ реплика интерпретируется как выбор
        # по нему — это и есть «отдельная ветвь диалога». Выход из неё —
        # «никакие/отмена», что в commit_assignments отрисуется как cancel.
        if state.get("pending_assignments"):
            return {"intent": "assignments_select"}

        ai = llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=last_text),
        ])
        label = (ai.content or "").strip().lower()
        if label not in _ROUTE_LABELS:
            label = "chat"
        return {"intent": label}

    return route


def make_respond_node(llm: GigaChat):
    def respond(state: OrchestratorState, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable") or {}
        system_prompt = cfg.get("system_prompt_override") or RESPONDER_PROMPT

        parts: list[str] = [system_prompt]
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
        return {"messages": [ai]}

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
            return {
                "analytics_question": None,
                "analytics_answer": None,
                "analytics_error": "Нет реплики пользователя для аналитического запроса.",
            }
        if metrics is None:
            return {
                "analytics_question": last_text,
                "analytics_answer": None,
                "analytics_error": (
                    state.get("metrics_error")
                    or "Метрики не загружены — нечем кормить json_analyzer."
                ),
            }
        if not direction_key:
            return {
                "analytics_question": last_text,
                "analytics_answer": None,
                "analytics_error": "Пустой direction_key — json_analyzer не сможет изолировать pgvector-кэш.",
            }

        try:
            result = await json_analyzer_graph.ainvoke({
                "raw_json": metrics,
                "question": last_text,
                "direction_key": direction_key,
            })
            return {
                "analytics_question": last_text,
                "analytics_answer": result.get("answer") or None,
                "analytics_error": None,
            }
        except Exception as exc:  # noqa: BLE001 — внешний подграф (LLM/БД), сужать нечем
            return {
                "analytics_question": last_text,
                "analytics_answer": None,
                "analytics_error": f"{type(exc).__name__}: {exc}"[:300],
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
            return {"easyrag_snippets": [], "easyrag_error": None}

        direction_key = (state.get("direction_key") or "").strip()
        last_text = _last_user_text(state)
        if not direction_key or not last_text:
            return {
                "easyrag_snippets": [],
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
            return {
                "easyrag_query": last_text,
                "easyrag_snippets": result.get("snippets") or [],
                "easyrag_error": None,
            }
        except Exception as exc:  # noqa: BLE001 — внешний подграф (сеть/БД), сужать нечем
            return {
                "easyrag_query": last_text,
                "easyrag_snippets": [],
                "easyrag_error": f"{type(exc).__name__}: {exc}"[:300],
            }

    return call_easyrag


def make_extract_assignments_node(llm: GigaChat):
    """Извлекает кандидатов-поручений из метрик и последнего AI-анализа.

    Запускается автоматически после ``initial_analysis`` и повторно — когда
    роутер ловит явный intent ``assignments``. Пишет результат и в
    ``candidate_assignments`` (история), и в ``pending_assignments`` (то,
    что сейчас на выборе). Пустой список — нормальный исход.
    """

    def extract(state: OrchestratorState) -> dict:
        metrics = state.get("metrics")
        if metrics is None or state.get("metrics_error"):
            return {"candidate_assignments": [], "pending_assignments": []}

        last_ai = _last_ai_text(state)
        context = _metrics_payload(metrics)
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

        return {
            "candidate_assignments": candidates,
            "pending_assignments": candidates,
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
        return {"selected_assignments": selected}

    return select


def make_commit_assignments_node():
    """Отправляет выбранные поручения в mock-сервис и закрывает ветку выбора.

    Всегда чистит ``pending_assignments``/``selected_assignments`` — независимо
    от исхода: пустой выбор = cancel, ошибка отправки = сообщение и сброс,
    успех = подтверждение и обновлённый ``last_committed_assignments``.
    """

    def commit(state: OrchestratorState) -> dict:
        selected = state.get("selected_assignments") or []
        boss = (state.get("boss_tabnum") or "").strip()
        employee = (state.get("employee_tabnum") or "").strip()

        if not selected:
            return {
                "messages": [AIMessage(
                    content="Хорошо, поручения сейчас не фиксирую. "
                    "Если передумаете — скажите «оформи поручения»."
                )],
                "pending_assignments": [],
                "selected_assignments": [],
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
        else:
            lines = [
                f"Зафиксировал {len(committed)} поручение(й) "
                f"для сотрудника {employee} (от руководителя {boss}):"
            ]
            for s in committed:
                lines.append(f"- {s.get('title', '').strip() or '(без названия)'}")
            text = "\n".join(lines)

        return {
            "messages": [AIMessage(content=text)],
            "pending_assignments": [],
            "selected_assignments": [],
            "last_committed_assignments": committed,
        }

    return commit


def need_load(state: OrchestratorState) -> str:
    return "ask_user" if state.get("loaded") else "load_data"


def after_route(state: OrchestratorState) -> str:
    intent = state.get("intent")
    if intent == "done":
        return "__end__"
    if intent == "analytics":
        return "call_json_analyzer"
    if intent == "wiki":
        return "call_easyrag"
    if intent == "assignments":
        return "extract_assignments"
    if intent == "assignments_select":
        return "select_assignments"
    # chat и любой неожиданный intent — прямо к респондеру без подграфов.
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
