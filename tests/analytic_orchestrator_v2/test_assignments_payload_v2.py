"""Юнит-тест структуры payload SendAssignmentsComponent.submit().

Проверяем целевой формат сервиса инсайтов: title/content.insights + маппинг
идентификаторов (object_id=сотрудник, subject_id=руководитель, session_id=thread)
и наличие метки времени. Зависимостей от LLM/langgraph нет.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langgraph_executor.aegra_agents.shared.assignments_service import (
    SendAssignmentsComponent,
)


def test_submit_payload_shape_and_id_mapping():
    insights = [
        {"type": "main_problem", "metric_id": "90028770",
         "metric_name": "AHT", "text": "AHT вырос на 22.5%."},
        {"type": "achievement", "metric_id": "90028777",
         "metric_name": "Доля переводов", "text": "Доля переводов 7.32%."},
    ]
    payload = SendAssignmentsComponent(
        boss_tabnum="832243",
        employee_tabnum="0932433",
        direction_key="dir-1",
        thread_id="373737",
        insights=insights,
    ).submit()

    assert payload["title"] == "agent_analyst_insights"
    assert payload["content"] == {"insights": insights}
    # object_id — сотрудник, subject_id — руководитель, session_id — thread aegra.
    assert payload["object_id"] == "0932433"
    assert payload["subject_id"] == "832243"
    assert payload["session_id"] == "373737"
    # metric_id каждого инсайта доходит до payload без потерь.
    assert [i["metric_id"] for i in payload["content"]["insights"]] == [
        "90028770", "90028777",
    ]
    # timestamp формата dd.mm.yyyy HH:MM:SS.
    assert re.fullmatch(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}", payload["timestamp"])
