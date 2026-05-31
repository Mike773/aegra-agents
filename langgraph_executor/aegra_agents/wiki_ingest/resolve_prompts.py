"""Промпты резолвера: merge тела страницы, relink (backlink) и LLM-судья.

Порт нужной части easyRag/query/prompts.py. Answer-промпты (query-сторона) сюда
НЕ портируются — они живут в подагенте ``easyrag``.

* ``save_wiki_page`` — merge: LLM перезаписывает ``wiki_page.body_md`` с учётом
  новых ``statements`` от кандидатов, сохраняя все ранее зафиксированные факты.
* ``relink_wiki_page`` — backlink: проставить ``[[…]]`` на сущности из каталога,
  не меняя контент.
* ``decide_entity_target`` — судья в ambiguous-зоне: кандидат == одна из страниц
  или новая сущность.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Слияние тела wiki-страницы (resolver)
# ---------------------------------------------------------------------------

WIKI_MERGE_TOOL_NAME = "save_wiki_page"
WIKI_MERGE_TOOL_DESCRIPTION = (
    "Сохранить итоговое тело wiki-страницы (markdown) и набор алиасов. "
    "Вызывать ровно один раз. Содержимое body_md полностью заменяет текущее."
)

WIKI_MERGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "body_md": {
            "type": "string",
            "description": (
                "Полный markdown страницы. Структурируй H2-секциями (## Заголовок). "
                "Каждое утверждение — самодостаточное предложение. "
                "При упоминании других именованных сущностей оборачивай их в [[Имя]] "
                "(имена собственные, идентифицируемые сущности; не даты, числа и общие слова). "
                "НЕ выдумывай факты — используй только то, что есть в текущем теле "
                "или в новых утверждениях."
            ),
        },
        "aliases": {
            "type": "array",
            "description": (
                "Альтернативные имена этой сущности (без основного заголовка). "
                "Включи существующие алиасы и добавь варианты написания, "
                "встретившиеся среди новых имён."
            ),
            "items": {"type": "string"},
        },
    },
    "required": ["body_md", "aliases"],
    "additionalProperties": False,
}


WIKI_MERGE_SYSTEM = (
    "Ты ведёшь wiki по сущностям, упомянутым в исходных документах. Тебе дают "
    "существующее тело страницы (может быть пустым — тогда страница создаётся "
    "впервые) и список новых утверждений, извлечённых из исходных документов. "
    "Твоя задача — выдать обновлённое тело страницы целиком.\n"
    "\n"
    "ЖЁСТКИЕ ПРАВИЛА:\n"
    "1. Сохрани ВСЕ факты из текущего тела. Не удаляй и не сокращай ранее "
    "зафиксированные утверждения, даже если они кажутся избыточными. "
    "Если возникает прямое противоречие — оставь оба варианта и пометь "
    "коротким комментарием, какой из источников их даёт.\n"
    "2. Влей новые утверждения в подходящие секции. Если темы нет — создай "
    "новую H2-секцию с осмысленным заголовком.\n"
    "3. Оборачивай в [[Имя]] упоминания других именованных сущностей — любых "
    "идентифицируемых объектов с собственным именем, которые могут иметь "
    "отдельную wiki-страницу. НЕ ссылайся на даты, числа и общие слова. "
    "Не ограничивай себя каким-то одним классом сущностей — действуй по смыслу "
    "документа.\n"
    "4. КАТЕГОРИЧЕСКИ запрещено оборачивать в [[…]] упоминания самой текущей "
    "страницы (её заголовок указан в user-сообщении первым). Никогда не "
    "пиши [[Заголовок]], если это заголовок именно этой страницы — повторяй "
    "имя как обычный текст. Самореференция ломает граф ссылок.\n"
    "5. Каталог «Существующие сущности» (если он есть в user-сообщении) — это "
    "ОРФОГРАФИЧЕСКАЯ ПОДСКАЗКА, как правильно записать ссылку на сущность, "
    "которая УЖЕ реально упомянута в текущем теле или в новых утверждениях. "
    "Если сущность из каталога действительно встречается в материале — "
    "оборачивай её упоминание в [[Точное имя из каталога]] (не сокращённый "
    "вариант). НЕ выдумывай упоминания и не вставляй сущности из каталога, "
    "которых нет в исходном материале, только ради того, чтобы добавить "
    "ссылок. Каталог не является списком обязательных тем.\n"
    "6. Не выдумывай факты. Если новое утверждение продублировано — не дублируй "
    "его в выводе.\n"
    "7. Заголовки секций — короткие, в именительном падеже, по смыслу "
    "содержимого секции.\n"
    "8. Имена и обозначения на латинице сохраняй в оригинальном написании.\n"
    "9. Ответ — только через вызов tool save_wiki_page. Свободный текст игнорируется."
)


def build_merge_user_prompt(
    *,
    title: str,
    current_body: str,
    current_aliases: Sequence[str],
    new_descriptors: Sequence[str],
    new_statements: Sequence[str],
    source_uris: Sequence[str] = (),
    existing_entities: Sequence[tuple[str, Sequence[str]]] = (),
) -> str:
    """Собрать user-сообщение для merge-вызова."""
    title_clean = title.strip()
    parts: list[str] = [
        f"Заголовок страницы: {title_clean}\n"
        f"ВАЖНО: НЕ оборачивай «{title_clean}» в [[…]] в тексте — это заголовок "
        "САМОЙ страницы, на себя не ссылаемся."
    ]

    aliases_clean = [a.strip() for a in current_aliases if a and a.strip()]
    if aliases_clean:
        parts.append("Текущие алиасы: " + ", ".join(aliases_clean))

    descriptors_clean = [d.strip() for d in new_descriptors if d and d.strip()]
    if descriptors_clean:
        bullets = "\n".join(f"- {d}" for d in descriptors_clean)
        parts.append("Контекст новых вкладов (descriptor'ы кандидатов):\n" + bullets)

    catalog_lines: list[str] = []
    for ent_title, ent_aliases in existing_entities:
        name = (ent_title or "").strip()
        if not name:
            continue
        cleaned_aliases = [a.strip() for a in (ent_aliases or ()) if a and a.strip()]
        if cleaned_aliases:
            catalog_lines.append(f"- {name} (псевдонимы: {', '.join(cleaned_aliases)})")
        else:
            catalog_lines.append(f"- {name}")
    if catalog_lines:
        parts.append(
            "Существующие сущности (используй [[Имя]] точно по этому списку, "
            "если упоминаешь любую из них):\n" + "\n".join(catalog_lines)
        )

    body_block = (current_body or "").strip()
    if body_block:
        parts.append(
            "Текущее тело страницы (между тегами):\n"
            f"<current_body>\n{body_block}\n</current_body>"
        )
    else:
        parts.append("Текущего тела страницы нет — создаёшь с нуля.")

    statements_clean = [s.strip() for s in new_statements if s and s.strip()]
    if not statements_clean:
        parts.append("Новых утверждений нет — верни текущее тело и алиасы без изменений.")
    else:
        bullets = "\n".join(f"- {s}" for s in statements_clean)
        parts.append("Новые утверждения для интеграции:\n" + bullets)

    sources_clean = [u.strip() for u in source_uris if u and u.strip()]
    if sources_clean:
        parts.append("Источники этого раунда: " + ", ".join(sources_clean))

    parts.append(
        "Сформируй обновлённое тело страницы и набор алиасов по правилам "
        "системного сообщения и верни их вызовом tool save_wiki_page."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Back-link: добавить [[…]] ссылки на новые сущности в существующую страницу
# ---------------------------------------------------------------------------

WIKI_RELINK_TOOL_NAME = "relink_wiki_page"
WIKI_RELINK_TOOL_DESCRIPTION = (
    "Сохранить обновлённое тело wiki-страницы (markdown) после простановки "
    "[[…]] ссылок на сущности из переданного каталога. Вызывать ровно один раз."
)

WIKI_RELINK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "body_md": {
            "type": "string",
            "description": (
                "Полный markdown страницы. Содержимое и структура должны "
                "совпадать с current_body, отличие — только в добавленных "
                "[[Имя]] ссылках на сущности из каталога, упоминания которых "
                "уже есть в тексте."
            ),
        },
        "aliases": {
            "type": "array",
            "description": (
                "Алиасы страницы. ВЕРНИ ТЕ ЖЕ, что переданы в current_aliases — "
                "поле обязательно по схеме."
            ),
            "items": {"type": "string"},
        },
    },
    "required": ["body_md", "aliases"],
    "additionalProperties": False,
}


WIKI_RELINK_SYSTEM = (
    "Ты ведёшь wiki по сущностям. Тебе дают тело уже существующей страницы и "
    "каталог других страниц wiki (title + aliases). Твоя задача — оборачивать "
    "в [[Имя]] упоминания этих сущностей, которые УЖЕ присутствуют в текущем "
    "теле страницы, и больше ничего не менять.\n"
    "\n"
    "ЖЁСТКИЕ ПРАВИЛА:\n"
    "1. НЕ меняй формулировки, порядок слов, структуру секций, заголовки H2.\n"
    "2. НЕ добавляй новых фактов, предложений, секций.\n"
    "3. НЕ удаляй ничего из текущего тела — ни предложений, ни ссылок.\n"
    "4. НЕ оборачивай в [[…]] упоминания САМОЙ этой страницы (её заголовок "
    "указан в user-сообщении первым). Самореференция ломает граф ссылок.\n"
    "5. Уже существующие [[…]] оставь как есть.\n"
    "6. Если упоминание сущности из каталога стоит в тексте в склонённой форме, "
    "оборачивай весь упомянутый токен в [[Имя из каталога|склонённая форма]] — "
    "так читатель видит исходный текст, а граф знает целевой slug. Не "
    "переписывай склонение в именительный.\n"
    "7. Если ни одной сущности из каталога в теле нет — верни current_body "
    "ровно как был передан.\n"
    "8. НЕ выдумывай упоминаний и не вставляй сущности из каталога, которых "
    "нет в исходном теле.\n"
    "9. Поле aliases в ответе — те же значения, что в current_aliases.\n"
    "10. Ответ — только через вызов tool relink_wiki_page."
)


def build_relink_user_prompt(
    *,
    title: str,
    current_body: str,
    current_aliases: Sequence[str],
    catalog: Sequence[tuple[str, Sequence[str]]],
) -> str:
    """Собрать user-сообщение для relink-вызова."""
    title_clean = title.strip()
    parts: list[str] = [
        f"Заголовок страницы: {title_clean}\n"
        f"ВАЖНО: НЕ оборачивай «{title_clean}» в [[…]] в тексте — это заголовок "
        "САМОЙ страницы, на себя не ссылаемся."
    ]

    aliases_clean = [a.strip() for a in current_aliases if a and a.strip()]
    if aliases_clean:
        parts.append("Текущие алиасы (передай в ответе без изменений): " + ", ".join(aliases_clean))
    else:
        parts.append("Текущих алиасов нет — верни в ответе пустой массив.")

    catalog_lines: list[str] = []
    for ent_title, ent_aliases in catalog:
        name = (ent_title or "").strip()
        if not name:
            continue
        cleaned_aliases = [a.strip() for a in (ent_aliases or ()) if a and a.strip()]
        if cleaned_aliases:
            catalog_lines.append(f"- {name} (псевдонимы: {', '.join(cleaned_aliases)})")
        else:
            catalog_lines.append(f"- {name}")
    if catalog_lines:
        parts.append(
            "Каталог сущностей (используй [[Имя]] точно по этому списку, если "
            "упоминание встретилось в теле страницы):\n" + "\n".join(catalog_lines)
        )
    else:
        parts.append("Каталог пуст — оборачивать нечего, верни тело без изменений.")

    body_block = (current_body or "").strip()
    if body_block:
        parts.append(
            "Текущее тело страницы (между тегами, итоговое тело должно "
            "совпадать дословно, кроме добавленных [[…]] ссылок):\n"
            f"<current_body>\n{body_block}\n</current_body>"
        )
    else:
        parts.append("Текущего тела страницы нет — верни пустую строку.")

    parts.append(
        "Верни обновлённое тело и тот же набор алиасов вызовом tool "
        f"{WIKI_RELINK_TOOL_NAME}."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Решение резолвера в ambiguous-зоне (LLM-судья)
# ---------------------------------------------------------------------------

RESOLVE_JUDGE_TOOL_NAME = "decide_entity_target"
RESOLVE_JUDGE_TOOL_DESCRIPTION = (
    "Решить, является ли кандидат-сущность той же, что одна из показанных "
    "wiki-страниц, либо новой сущностью."
)

RESOLVE_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["existing", "new", "ambiguous"],
            "description": (
                "existing — кандидат и одна из показанных страниц — одна и "
                "та же сущность; new — новая сущность; ambiguous — нельзя "
                "решить уверенно."
            ),
        },
        "slug": {
            "type": "string",
            "description": (
                "Slug выбранной страницы. Обязателен ТОЛЬКО если "
                "decision=existing — и должен быть строго одним из slug'ов, "
                "перечисленных в user-сообщении."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": "1-2 коротких предложения с обоснованием решения.",
        },
    },
    "required": ["decision", "reasoning"],
    "additionalProperties": False,
}


RESOLVE_JUDGE_SYSTEM = (
    "Тебе дают одного кандидата-сущность (имя, описание, факты) и небольшой "
    "список существующих wiki-страниц. Реши, ссылается ли кандидат на ту же "
    "реальную сущность, что одна из показанных страниц.\n"
    "\n"
    "ПРАВИЛА:\n"
    "1. existing — выбирай только если ты УВЕРЕН, что это синонимы/варианты "
    "одного и того же объекта: разные имена одного и того же персонажа, "
    "разные написания одного и того же названия, полное и сокращённое имя "
    "одной и той же сущности. Укажи slug этой страницы.\n"
    "2. new — если ни одна из показанных страниц явно не та же сущность.\n"
    "3. ambiguous — если данных недостаточно, чтобы решить уверенно. "
    "Лучше отложить, чем слить разные сущности.\n"
    "4. Близость имён ≠ та же сущность. Опирайся на описание и факты, а не "
    "только на буквенное сходство. Однофамильцы, тёзки, разные объекты "
    "с похожими названиями — всё это разные сущности.\n"
    "5. slug в ответе должен быть строго одним из slug'ов, показанных в "
    "user-сообщении. Не выдумывай и не пиши новый slug.\n"
    "6. Ответ — только через вызов tool decide_entity_target."
)


def build_resolve_judge_user_prompt(
    *,
    candidate_name: str,
    candidate_descriptor: str,
    candidate_statements: Sequence[str],
    options: Sequence[tuple[str, str, Sequence[str], str]],
) -> str:
    """Собрать user-сообщение для LLM-судьи.

    ``options`` — список ``(slug, title, aliases, body_excerpt)`` для top-N
    страниц-претендентов, по убыванию похожести.
    """
    parts: list[str] = [
        "Кандидат-сущность:",
        f"- Имя: {candidate_name.strip() or '(пусто)'}",
    ]
    descriptor_clean = (candidate_descriptor or "").strip()
    if descriptor_clean:
        parts.append(f"- Описание: {descriptor_clean}")
    statements_clean = [s.strip() for s in candidate_statements if s and s.strip()]
    if statements_clean:
        bullets = "\n".join(f"  - {s}" for s in statements_clean)
        parts.append("- Факты:\n" + bullets)

    if not options:
        parts.append(
            "Существующих страниц-претендентов нет. Реши: new (если кандидат "
            "осмыслен) или ambiguous (если данных мало)."
        )
    else:
        parts.append("Существующие wiki-страницы-претенденты (в порядке релевантности):")
        for slug, title, aliases, body_excerpt in options:
            slug_clean = slug.strip()
            title_clean = (title or "").strip()
            aliases_clean = [a.strip() for a in (aliases or ()) if a and a.strip()]
            header = f"- slug={slug_clean} | title={title_clean}"
            if aliases_clean:
                header += f" | aliases: {', '.join(aliases_clean)}"
            parts.append(header)
            body_clean = (body_excerpt or "").strip()
            if body_clean:
                parts.append(f"  выдержка: {body_clean}")

    parts.append(
        "Реши: existing (укажи slug) / new / ambiguous и верни решение "
        f"вызовом tool {RESOLVE_JUDGE_TOOL_NAME}."
    )
    return "\n\n".join(parts)


__all__ = [
    "RESOLVE_JUDGE_SCHEMA",
    "RESOLVE_JUDGE_SYSTEM",
    "RESOLVE_JUDGE_TOOL_DESCRIPTION",
    "RESOLVE_JUDGE_TOOL_NAME",
    "WIKI_MERGE_SCHEMA",
    "WIKI_MERGE_SYSTEM",
    "WIKI_MERGE_TOOL_DESCRIPTION",
    "WIKI_MERGE_TOOL_NAME",
    "WIKI_RELINK_SCHEMA",
    "WIKI_RELINK_SYSTEM",
    "WIKI_RELINK_TOOL_DESCRIPTION",
    "WIKI_RELINK_TOOL_NAME",
    "build_merge_user_prompt",
    "build_relink_user_prompt",
    "build_resolve_judge_user_prompt",
]
