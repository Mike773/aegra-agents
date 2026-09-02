"""Сводка составного разбора: результаты задач → один ответ руководителю."""
from __future__ import annotations

from typing import Any

# Полный ответ показываем только тем задачам, от которых зависит текущая;
# остальные ужимаем — иначе промпт растёт с числом задач.
_DEPENDENCY_CAP = 1500
_DIGEST_CAP = 300

SUMMARIZER_PROMPT_TAIL = (
    "Выше — результаты отдельных задач составного разбора. Сведи их в ОДИН "
    "связный ответ руководителю по правилам структуры ответа: главная область "
    "внимания с числами и причинами, затем сводка и сравнение с коллегами, в "
    "конце — предложение продолжить. Поскольку задачи были заданы явно, можно "
    "вести рассказ по задачам, но без служебных подписей и нумерации разделов. "
    "Числа бери только из результатов задач, новых не придумывай."
)


def prior_results_digest(results: list[dict[str, Any]], task: dict[str, Any]) -> str:
    """Контекст предыдущих задач для текущей."""
    if not results:
        return ""
    depends = set(task.get("depends_on") or [])
    lines: list[str] = []
    for r in results:
        answer = (r.get("answer_md") or "").strip()
        if not answer:
            continue
        if r.get("id") in depends:
            body = answer[:_DEPENDENCY_CAP]
            lines.append(f"Результат задачи «{r.get('title')}» (нужен для этой):\n{body}")
        else:
            body = answer[:_DIGEST_CAP] + ("…" if len(answer) > _DIGEST_CAP else "")
            lines.append(f"Кратко из задачи «{r.get('title')}»: {body}")
    return "\n\n".join(lines)


def summary_input(results: list[dict[str, Any]]) -> str:
    """Текст для сводки: все ответы задач по порядку."""
    parts: list[str] = []
    for i, r in enumerate(results or [], 1):
        answer = (r.get("answer_md") or "").strip()
        if not answer:
            continue
        parts.append(f"Задача {i}. {r.get('title')}\n{answer}")
    return "\n\n".join(parts)


__all__ = ["SUMMARIZER_PROMPT_TAIL", "prior_results_digest", "summary_input"]
