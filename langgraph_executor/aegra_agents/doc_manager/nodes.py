"""Узлы графа doc_manager.

classify → (upload | list | delete | help). Все узлы отвечают ``AIMessage`` и
кладут краткий ``report``. ``direction_key`` читается из config.configurable,
затем из state (как в json_analyzer._resolve_inputs).
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from .config import get_settings
from .intent import classify
from .repository import delete_doc, insert_pending_doc, list_docs
from .resolve import resolve_delete_target


def _last_human_text(messages: list[Any]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            content = m.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "".join(parts)
    return ""


def _resolve_direction_key(state: dict, config: RunnableConfig) -> str:
    cfg = (config or {}).get("configurable", {}) if config else {}
    return (
        (cfg.get("direction_key") if isinstance(cfg, dict) else None)
        or state.get("direction_key")
        or ""
    ).strip()


def _derive_title(text: str, hint: str = "") -> str:
    """uri документа: первая непустая строка (обрезанная), иначе подсказка LLM."""
    limit = get_settings().title_max_len
    for line in (text or "").splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:limit]
    hint = (hint or "").strip()
    if hint:
        return hint[:limit]
    return "Без названия"


def _reply(text: str, **extra: Any) -> dict:
    return {"messages": [AIMessage(content=text)], "report": text, **extra}


def _format_listing(docs: list[dict]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        status = "обработан" if d.get("processed_at") else "в очереди"
        chunks = d.get("chunks") or 0
        chunk_str = f", чанков: {chunks}" if status == "обработан" else ""
        short_id = str(d.get("id"))[:8]
        lines.append(
            f"{i}. {d.get('uri')} — {status}{chunk_str} (id: {short_id}…)"
        )
    return "\n".join(lines)


async def classify_node(state: dict, config: RunnableConfig) -> dict:
    direction_key = _resolve_direction_key(state, config)
    if not direction_key:
        return _reply(
            "Не задан direction_key — не знаю, к какому направлению относить документ. "
            "Передайте его в configurable.",
            intent="unknown",
        )

    text = _last_human_text(state.get("messages") or [])
    if not text.strip():
        return _reply(
            "Пустое сообщение. Пришлите текст документа для загрузки, "
            "либо команду «список» / «удали …».",
            intent="unknown",
            direction_key=direction_key,
        )

    parsed = await classify(text)
    intent = parsed["intent"]
    out: dict = {"intent": intent, "direction_key": direction_key}
    if intent == "upload":
        out["upload_content"] = text
        out["upload_title"] = parsed.get("title") or ""
    elif intent == "delete":
        out["delete_reference"] = parsed.get("reference") or ""
    return out


def after_classify(state: dict) -> str:
    intent = state.get("intent") or "unknown"
    if intent in {"upload", "list", "delete"}:
        return intent
    return "unknown"


async def do_upload(state: dict, config: RunnableConfig) -> dict:
    cfg = get_settings()
    direction_key = state.get("direction_key") or _resolve_direction_key(state, config)
    content = state.get("upload_content") or _last_human_text(state.get("messages") or [])
    if not (content or "").strip():
        return _reply("Нечего загружать — текст документа пуст.")

    uri = _derive_title(content, state.get("upload_title") or "")
    result = await insert_pending_doc(
        direction_key=direction_key,
        uri=uri,
        content=content,
        mime=cfg.default_mime,
        dedup=cfg.upload_dedup,
    )
    if result["status"] == "duplicate":
        where = "уже обработан" if result.get("processed") else "уже в очереди"
        return _reply(
            f"Такой документ {where} (id: {result['id'][:8]}…, «{result['uri']}»). "
            "Повторная загрузка пропущена."
        )
    return _reply(
        f"Документ «{uri}» принят и поставлен в очередь на обработку "
        f"(id: {result['id'][:8]}…). Он будет проиндексирован при следующем "
        "прогоне wiki_ingest."
    )


async def do_list(state: dict, config: RunnableConfig) -> dict:
    direction_key = state.get("direction_key") or _resolve_direction_key(state, config)
    docs = await list_docs(direction_key)
    if not docs:
        return _reply(
            "По этому направлению документов пока нет.", last_listed=[]
        )
    listing = _format_listing(docs)
    return _reply(
        f"Загруженные документы ({len(docs)}):\n{listing}",
        last_listed=docs,
    )


async def do_delete(state: dict, config: RunnableConfig) -> dict:
    direction_key = state.get("direction_key") or _resolve_direction_key(state, config)
    listed = state.get("last_listed") or []
    if not listed:
        # Подтянуть актуальный список, чтобы разрешить ссылку.
        listed = await list_docs(direction_key)

    reference = state.get("delete_reference") or ""
    doc_id, status = resolve_delete_target(reference, listed)

    if status == "empty":
        return _reply(
            "По этому направлению документов нет — удалять нечего.",
            last_listed=[],
        )
    if status == "not_found":
        return _reply(
            f"Не нашёл документ по ссылке «{reference}». Вот текущий список:\n"
            f"{_format_listing(listed)}\nУкажите номер для удаления.",
            last_listed=listed,
        )
    if status == "ambiguous":
        return _reply(
            f"Ссылка «{reference}» подходит сразу нескольким документам. "
            f"Текущий список:\n{_format_listing(listed)}\nУкажите точный номер.",
            last_listed=listed,
        )

    result = await delete_doc(direction_key=direction_key, doc_id=doc_id)
    if result["status"] == "not_found":
        return _reply(
            "Документ не найден (возможно, уже удалён). Запросите список заново.",
            last_listed=[],
        )

    msg = f"Документ «{result['uri']}» удалён (id: {result['id'][:8]}…)."
    if result.get("was_processed"):
        msg += (
            " Документ был уже обработан: его чанки и сущности-кандидаты удалены "
            "каскадом, но wiki-страницы, построенные из него ранее, остаются — "
            "при необходимости их нужно править/удалять отдельно."
        )
    # Сбросить кэш списка: индексы устарели после удаления.
    return _reply(msg, last_listed=[])


__all__ = [
    "classify_node",
    "after_classify",
    "do_upload",
    "do_list",
    "do_delete",
]
