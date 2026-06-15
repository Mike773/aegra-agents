"""Инсайты-поручения нового формата: каталог метрик, парсинг, отправка.

Внешний сервис принимает {employee_tabnum, direction_key, insights}, где каждый
insight — {type, metric_id, metric_name, text}, type ∈
{main_problem, problem, norm, achievement}, причём main_problem СТРОГО один.

Импортируем только чистые функции из nodes.py + заглушку сервиса — без сети/LLM.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from langchain_core.messages import AIMessage, HumanMessage

from langgraph_executor.aegra_agents.analytic_orchestrator.nodes import (
    _STEP_KEY,
    _collect_metric_catalog,
    _format_metric_catalog,
    _gather_agent_answers,
    _parse_insights_json,
    _resolve_insight_metric,
)
from langgraph_executor.aegra_agents.shared.assignments_service import (
    SendAssignmentsComponent,
)


def _dataset():
    return {
        "employees": [{
            "tabnum": 2,
            "metrics": [{
                "id": "90022908", "metric_name": "Производительность",
                "child_metrics": [
                    {"id": "90022910", "metric_name": "Доля переводов",
                     "element": None, "child_metrics": []},
                    # тот же id в другом разрезе element — дедуп по id
                    {"id": "90022910", "metric_name": "Доля переводов",
                     "element": "Банковские счета", "child_metrics": []},
                ],
            }],
        }],
    }


# --- каталог метрик ---------------------------------------------------------

def test_catalog_walks_tree_and_dedups_by_id():
    cat = _collect_metric_catalog(_dataset())
    ids = [c["id"] for c in cat]
    assert ids == ["90022908", "90022910"]  # дедуп: один 90022910
    by_id = {c["id"]: c["metric_name"] for c in cat}
    assert by_id["90022908"] == "Производительность"
    assert all("description" in c for c in cat)  # описание для маппинга LLM


def test_catalog_format_human_readable():
    s = _format_metric_catalog(_collect_metric_catalog(_dataset()))
    assert "90022908 | Производительность" in s
    assert "90022910 | Доля переводов" in s


# --- детерминированный резолвер метрики -------------------------------------

def test_resolve_by_exact_id():
    cat = _collect_metric_catalog(_dataset())
    mid, name = _resolve_insight_metric("текст", "90022908", "", cat)
    assert (mid, name) == ("90022908", "Производительность")


def test_resolve_by_exact_name_casefold():
    cat = _collect_metric_catalog(_dataset())
    mid, name = _resolve_insight_metric("текст", "", "доля переводов", cat)
    assert (mid, name) == ("90022910", "Доля переводов")


def test_resolve_by_fuzzy_name():
    cat = _collect_metric_catalog(_dataset())
    # лёгкая опечатка/склонение имени
    mid, name = _resolve_insight_metric("текст", "", "Производительности", cat)
    assert (mid, name) == ("90022908", "Производительность")


def test_resolve_by_text_scan_when_fields_empty():
    cat = _collect_metric_catalog(_dataset())
    # имя/ид пустые — но в тексте упомянута метрика из каталога
    mid, name = _resolve_insight_metric(
        "Доля переводов выросла до 21% при плане 10%.", "", "", cat,
    )
    assert (mid, name) == ("90022910", "Доля переводов")


def test_resolve_returns_asis_when_no_match():
    cat = _collect_metric_catalog(_dataset())
    mid, name = _resolve_insight_metric("ничего знакомого", "", "", cat)
    assert (mid, name) == ("", "")


def test_catalog_empty_dataset():
    assert _collect_metric_catalog({}) == []
    assert _format_metric_catalog([]) == "(каталог пуст)"


# --- сбор ответов агента из диалога -----------------------------------------

def test_gather_agent_answers_skips_steps_and_human():
    state = {"messages": [
        HumanMessage(content="вопрос"),
        AIMessage(content="📊 шаг", additional_kwargs={_STEP_KEY: True}),
        AIMessage(content="Производительность 25.9, выше плана."),
        HumanMessage(content="оформи поручения"),
    ]}
    answers = _gather_agent_answers(state)
    assert "Производительность 25.9" in answers
    assert "📊 шаг" not in answers
    assert "вопрос" not in answers


# --- парсинг инсайтов -------------------------------------------------------

def test_parse_insights_object_form_and_catalog_id_restore():
    catalog = _collect_metric_catalog(_dataset())
    raw = (
        '{"insights": [{"type": "main_problem", "metric_name": "Доля переводов", '
        '"text": "Доля переводов 21% при плане 10%."}]}'
    )
    out = _parse_insights_json(raw, catalog)
    assert len(out) == 1
    # id восстановлен по имени из каталога
    assert out[0]["metric_id"] == "90022910"
    assert out[0]["type"] == "main_problem"


def test_parse_insights_restores_name_by_id():
    catalog = _collect_metric_catalog(_dataset())
    raw = '[{"type": "norm", "metric_id": "90022908", "text": "В норме."}]'
    out = _parse_insights_json(raw, catalog)
    assert out[0]["metric_name"] == "Производительность"


def test_parse_insights_enforces_single_main_problem():
    raw = (
        '{"insights": ['
        '{"type": "main_problem", "metric_name": "A", "text": "t1"},'
        '{"type": "main_problem", "metric_name": "B", "text": "t2"}]}'
    )
    out = _parse_insights_json(raw, [])
    types = [i["type"] for i in out]
    assert types == ["main_problem", "problem"]  # второй понижен


def test_parse_insights_normalizes_unknown_type_and_drops_empty_text():
    raw = (
        '[{"type": "weird", "metric_name": "A", "text": "есть"},'
        '{"type": "norm", "metric_name": "B", "text": ""}]'
    )
    out = _parse_insights_json(raw, [])
    assert len(out) == 1
    assert out[0]["type"] == "problem"  # неизвестный тип → problem


def test_parse_insights_garbage_returns_empty():
    assert _parse_insights_json("не json", []) == []
    assert _parse_insights_json('{"foo": 1}', []) == []


# --- заглушка сервиса -------------------------------------------------------

def test_service_submit_contract():
    insights = [{"type": "problem", "metric_id": "1",
                 "metric_name": "CSAT", "text": "ниже плана"}]
    out = SendAssignmentsComponent(
        boss_tabnum="999", employee_tabnum="2", direction_key="dir-1",
        thread_id="thread-1", insights=insights,
    ).submit()
    assert out["object_id"] == "2"          # employee_tabnum
    assert out["subject_id"] == "999"       # boss_tabnum
    assert out["session_id"] == "thread-1"  # thread_id
    assert out["content"]["insights"] == insights
