"""Свободный SQL от модели (text2sql).

Модель пишет обычный SELECT по схеме из промпта. Безопасность — на стороне
базы (SafeQueryRunner), а не промпта: запись, DDL и мультизапросы отклоняются,
тяжёлый запрос прерывается по таймауту.

Аргумент ``purpose`` обязателен: он идёт в шаговое сообщение руководителю и в
трассу, поэтому по логу видно, ЗАЧЕМ агент лез в данные.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from ..db.sqlrunner import render_markdown
from ..db.text2sql import SafeQueryRunner

# Строк на один запрос: больше в контекст модели тащить бессмысленно.
_ROW_CAP = 60


def make_query_sql(ctx: Any) -> StructuredTool:
    runner = SafeQueryRunner(ctx.db.conn, max_rows=_ROW_CAP)
    runner.install()
    # База должна знать о защите, чтобы наши собственные записи (карта
    # отклонений) не отбивались авторизатором.
    ctx.db.guard = runner

    def query_sql(sql: str, purpose: str) -> str:
        ctx.used_data_tools = True
        res = runner.run(sql)
        if res.error:
            return res.error
        return render_markdown(res, max_rows=_ROW_CAP)

    return StructuredTool.from_function(
        func=query_sql,
        name="query_sql",
        description=(
            "Выполнить один read-only SELECT по схеме данных из системного "
            "промпта, когда нужен срез, которого нет в карточке показателя: "
            "ранжирование, фильтр, сравнение нескольких показателей, история. "
            "Разрез по части имени ищи через LIKE и всегда ставь LIMIT — разрезов "
            "бывают сотни. Аргументы: sql — запрос (только SELECT/WITH, до 60 "
            "строк в выдаче); purpose — обязательная короткая цель запроса "
            "по-русски, её увидит руководитель."
        ),
    )


__all__ = ["make_query_sql"]
