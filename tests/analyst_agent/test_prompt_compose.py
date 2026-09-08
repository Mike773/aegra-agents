"""analyst_agent.prompts.compose_system_prompt: один системный промпт агента.

Порядок блоков фиксирован: бизнес-промпт → как работать инструментами →
схема данных → что в этих данных → карта отклонений → подсказка хода. Бизнес-
промпт заменяется через system_prompt_override, операционная часть остаётся всегда.
"""
from __future__ import annotations

from _fixtures import make_dataset_obj, make_metric, make_synthetic_dataset  # noqa: F401

from langgraph_executor.aegra_agents.analyst_agent import prompts
from langgraph_executor.aegra_agents.analyst_agent.db import analytics, core


def _ctx(**kw):
    base = dict(
        schema_doc="СХЕМА ДАННЫХ\nv_fact(...)",
        enrichment_block="СОСТАВ ДАННЫХ\n- В анализе: Иванов.",
        catalog_block="КАТАЛОГ ПОКАЗАТЕЛЕЙ\n- Продажи",
        deviations_block="КАРТА ОТКЛОНЕНИЙ\n- Продажи — хуже плана.",
        org_block="Кого анализируем: Иванов.",
        briefing="Разбери показатели",
        turn_kind="initial",
        tool_budget=18,
    )
    base.update(kw)
    return prompts.PromptContext(**base)


def test_block_order():
    text = prompts.compose_system_prompt(_ctx())
    positions = [
        text.index("# Роль и Миссия"),
        text.index("# Структура отчета"),
        text.index("КАК РАБОТАТЬ"),
        text.index("СХЕМА ДАННЫХ"),
        text.index("СОСТАВ ДАННЫХ"),
        text.index("КАТАЛОГ ПОКАЗАТЕЛЕЙ"),
        text.index("КАРТА ОТКЛОНЕНИЙ"),
    ]
    assert positions == sorted(positions), positions


def test_business_prompt_is_single_block():
    """Бизнес-промпт — одна переменная и входит в системный промпт целиком."""
    text = prompts.compose_system_prompt(_ctx())
    assert prompts.BUSINESS_PROMPT in text
    for marker in (
        "# Принципы аналитического мышления",
        "# Стиль и Язык",
        "# Диалог и Свободная форма",
    ):
        assert marker in text


def test_override_replaces_business_but_keeps_operational():
    text = prompts.compose_system_prompt(_ctx(system_prompt_override="ТЫ ПРОСТО БОТ"))
    assert "ТЫ ПРОСТО БОТ" in text
    assert "# Роль и Миссия" not in text
    assert "# Структура отчета" not in text
    # Операционные блоки остаются: без них модель не сможет работать с данными.
    assert "СХЕМА ДАННЫХ" in text
    assert "КАТАЛОГ ПОКАЗАТЕЛЕЙ" in text
    assert "КАК РАБОТАТЬ" in text


def test_star_block_only_when_stars_present():
    assert "## Как говорить про звёзды" not in prompts.compose_system_prompt(_ctx())
    assert "## Как говорить про звёзды" in prompts.compose_system_prompt(
        _ctx(has_stars=True)
    )


def test_memory_block_filtered_when_empty():
    assert "Долгосрочная память" not in prompts.compose_system_prompt(
        _ctx(memory_context="Долгосрочная память отсутствует.")
    )
    assert "договорились" in prompts.compose_system_prompt(
        _ctx(memory_context="В прошлый раз договорились смотреть AHT.")
    )


def test_no_turn_hint_outside_dashboard():
    """Первый ход ведёт входное сообщение, последующие — история диалога:
    отдельных подсказок хода у них нет, промпт одинаковый."""
    initial = prompts.compose_system_prompt(_ctx(turn_kind="initial"))
    followup = prompts.compose_system_prompt(_ctx(turn_kind="followup"))
    assert initial == followup
    assert "Брифинг руководителя" in initial
    assert "первичный разбор" not in initial
    assert "реплику руководителя" not in followup
    assert prompts.DASHBOARD_TASK_HINT not in initial
    dashboard = prompts.compose_system_prompt(_ctx(turn_kind="dashboard"))
    assert prompts.DASHBOARD_TASK_HINT in dashboard


def test_dashboard_task_block():
    text = prompts.compose_system_prompt(
        _ctx(
            turn_kind="dashboard",
            task_block="ЗАДАЧА 2 из 3: Разбери качество\nРезультат задачи 1: продажи просели.",
        )
    )
    assert "ЗАДАЧА 2 из 3" in text
    assert "Результат задачи 1" in text


def test_tools_guide_mentions_budget_and_entry_points():
    """Полный список инструментов модель видит в functions; в гиде — бюджет и
    с чего начинать: карта отклонений и карточка показателя."""
    text = prompts.compose_system_prompt(_ctx(tool_budget=12))
    assert "12" in text
    for tool in ("metric_card", "list_deviations"):
        assert tool in text


def test_tools_guide_warns_about_differing_dates():
    text = prompts.compose_system_prompt(_ctx())
    assert "дат" in text.lower()
    assert "is_last_of_series" in text or "последнее значение" in text.lower()


def test_absent_blocks_are_skipped():
    text = prompts.compose_system_prompt(
        _ctx(deviations_block="", catalog_block="", org_block="")
    )
    assert "КАРТА ОТКЛОНЕНИЙ" not in text
    assert "\n\n\n" not in text


def test_fits_budget_on_production_scale():
    from langgraph_executor.aegra_agents.analyst_agent.db import enrich

    db = core.build_run_db(make_synthetic_dataset(n_level1=100, depth=5, periods=6))
    text = prompts.compose_system_prompt(
        _ctx(
            schema_doc=core.schema_doc(db),
            enrichment_block=enrich.enrichment_block(db, person_key="100500"),
            catalog_block=enrich.catalog_block(db),
            has_stars=db.has_stars,
        )
    )
    assert len(text) <= 50_000, len(text)


def test_tools_guide_keeps_only_cross_cutting_rules():
    """Список инструментов модель видит в functions; в гиде остаются правила,
    которые ни к одному инструменту не привязаны."""
    guide = prompts.tools_guide_block(18)
    for rule in (
        "ТОЛЬКО через инструменты",
        "карты отклонений",
        "Один вызов инструмента за шаг",
        "Бюджет вызовов на этот ответ — 18",
        "РАЗНЫЕ даты",
        "Вердикты",
        "своего итога нет",
        "БЕЗ вызова инструментов",
    ):
        assert rule in guide, rule
    # Пер-тульных описаний в гиде больше нет.
    assert "`metric_card` — всё об одном показателе" not in guide
    assert "`search_wiki` — методика" not in guide
    assert "передавай `purpose`" not in guide
    assert len(guide) < 1800
