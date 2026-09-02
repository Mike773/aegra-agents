"""Группы сравнения: название приходит с данными, конфиг задаёт только приоритет.

Название группы берётся из поля ``level_name`` предагрегатов и подставляется НА
УРОВНЕ РЕНДЕРА — служебный код (``level``) до модели не доходит никогда: нет
названия, значит группа описывается обезличенно. Конфиг сервиса (``PEER_LEVELS``)
хранит только порядок кодов, от узкой группы к широкой; первый становится
референсным для расчётов.
"""
import re

import pytest

from langgraph_executor.aegra_agents.json_analyzer_v5.analytics import (
    _level_order,
    compute_analytics,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.loader import (
    load_aggregates_obj,
    load_dataset_obj,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.sqlite_store import SqliteStore
from langgraph_executor.aegra_agents.json_analyzer_v5.store_cache import EmbeddingIndex
from langgraph_executor.aegra_agents.json_analyzer_v5.tools import build_tools
from langgraph_executor.aegra_agents.shared import peer_levels

CUR = "2026-07-30"
ORDER = "OFFICE|TERR|ORG"
NAMES = {"OFFICE": "по офису", "TERR": "по территории", "ORG": "по всему банку"}


@pytest.fixture
def levels(monkeypatch):
    """Ставит порядок уровней в окружение и чистит кэш модуля до и после теста."""
    def _set(value: str | None):
        if value is None:
            monkeypatch.delenv(peer_levels.PEER_LEVELS_ENV, raising=False)
        else:
            monkeypatch.setenv(peer_levels.PEER_LEVELS_ENV, value)
        peer_levels.reset_cache()

    yield _set
    peer_levels.reset_cache()


# --- Разбор конфига: только приоритет --------------------------------------

def test_parse_codes_compact_and_json():
    assert peer_levels.parse_peer_levels(ORDER) == ("OFFICE", "TERR", "ORG")
    assert peer_levels.parse_peer_levels('["OFFICE", "TERR"]') == ("OFFICE", "TERR")
    assert peer_levels.parse_peer_levels('[{"code": "OFFICE"}]') == ("OFFICE",)


def test_parse_legacy_form_drops_titles():
    """Старая форма КОД:название ещё встречается — берём код, название игнорируем."""
    legacy = "OFFICE:по офису|TERR:территория"
    assert peer_levels.parse_peer_levels(legacy) == ("OFFICE", "TERR")


def test_parse_caps_at_three():
    """Бизнес-требование: групп сравнения не больше трёх, лишние отбрасываем."""
    parsed = peer_levels.parse_peer_levels(ORDER + "|REGION|WORLD")
    assert len(parsed) == peer_levels.MAX_PEER_LEVELS == 3
    assert parsed == ("OFFICE", "TERR", "ORG")


def test_parse_empty_and_broken():
    for raw in (None, "", "   ", "[не json", "[]", "[{}]", "|", ":"):
        assert peer_levels.parse_peer_levels(raw) == ()


# --- Порядок уровней для расчётов ------------------------------------------

def _agg_rows(*pairs):
    """(level, total_objects) → строки peer_aggregates в порядке появления."""
    return [
        {"level": lvl, "node_uid": i, "total_objects": objs, "is_current": 1}
        for i, (lvl, objs) in enumerate(pairs, 1)
    ]


def test_level_order_follows_config(levels):
    """Порядок конфига важнее эвристики по размеру группы."""
    rows = _agg_rows(("ORG", 10), ("TERR", 100), ("OFFICE", 500))
    levels(None)
    assert _level_order(rows)[0] == "ORG"  # эвристика: самая маленькая группа
    levels(ORDER)
    assert _level_order(rows) == ["OFFICE", "TERR", "ORG"]


def test_explicit_ref_level_beats_config(levels):
    levels(ORDER)
    rows = _agg_rows(("ORG", 10), ("TERR", 100), ("OFFICE", 500))
    assert _level_order(rows, ref_level="TERR")[0] == "TERR"


def test_levels_outside_config_go_last(levels):
    levels("OFFICE")
    rows = _agg_rows(("ЧУЖОЙ", 5), ("OFFICE", 500))
    assert _level_order(rows) == ["OFFICE", "ЧУЖОЙ"]


# --- Рендер: имя из данных --------------------------------------------------

def _peer_context_output(with_names: bool):
    data = {
        "me": None,
        "employees": [{"fio": "Иванов", "tabnum": 1, "metrics": [{
            "id": "1", "metric_name": "Продажи", "metric_type": "прямая",
            "measure_type": "рубль", "date": CUR, "calc_period": "Месяц",
            "fact": 100.0, "plan": 100.0, "benchmark": None, "element": None,
            "child_metrics": [],
            "rankings": [{"rank": "20 из 500", "level": "ORG", "percentile": 96.0}],
        }]}],
    }
    payload = []
    for lvl, objs in (("OFFICE", 40), ("ORG", 500)):
        ds = {"level": lvl, "metrics": [{
            "metric_id": "1", "metric_name": "Продажи",
            "aggregates": {"dt": CUR, "calc_period": "Месяц", "mean_fact": 800,
                           "median": 700, "top20_mean_fact": 2000, "hit_rate": 26.0,
                           "total_objects": objs},
        }]}
        if with_names:
            ds["level_name"] = NAMES[lvl]
        payload.append({"dataset": ds})

    store = SqliteStore()
    store.load(load_dataset_obj(data))
    store.load_aggregates(load_aggregates_obj(payload))
    compute_analytics(store)
    tool = next(
        t for t in build_tools(store, EmbeddingIndex([]), lambda q: [0.0])
        if t.name == "peer_context"
    )
    return tool.invoke({"metric": "Продажи"})


def test_render_uses_level_name_from_payload(levels):
    levels(ORDER)
    out = _peer_context_output(with_names=True)
    assert "по офису" in out
    assert "по всему банку" in out
    assert "сравнение с группой: по офису" in out  # референсная — первая по конфигу
    assert "OFFICE" not in out and "ORG" not in out


# --- Имя для персональных rankings берётся из общих предагрегатов -----------

def _store_with(metrics, agg_payload):
    data = {"me": None, "employees": [{"fio": "Иванов", "tabnum": 1,
                                       "metrics": metrics}]}
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    store.load_aggregates(load_aggregates_obj(agg_payload))
    compute_analytics(store)
    return store


def _metric(name, rankings=None, date=CUR):
    node = {
        "id": name, "metric_name": name, "metric_type": "прямая",
        "measure_type": "рубль", "date": date, "calc_period": "Месяц",
        "fact": 100.0, "plan": 100.0, "benchmark": None, "element": None,
        "child_metrics": [],
    }
    if rankings is not None:
        node["rankings"] = rankings
    return node


def _agg_dataset(level, level_name, metric_name, objs=500):
    return {"dataset": {"level": level, "level_name": level_name, "metrics": [{
        "metric_id": metric_name, "metric_name": metric_name,
        "aggregates": {"dt": CUR, "calc_period": "Месяц", "mean_fact": 800,
                       "median": 700, "total_objects": objs},
    }]}}


def _peer_context(store, metric):
    tool = next(
        t for t in build_tools(store, EmbeddingIndex([]), lambda q: [0.0])
        if t.name == "peer_context"
    )
    return tool.invoke({"metric": metric})


def test_name_pulled_from_other_metric_aggregates(levels):
    """У метрики только rankings — имя группы берём из агрегатов соседней метрики.

    В персональном датасете level_name нет вообще, поэтому карта имён должна
    строиться по всей таблице агрегатов, а не по срезу текущей метрики.
    """
    levels(ORDER)
    store = _store_with(
        metrics=[
            _metric("Продажи", rankings=[
                {"rank": "20 из 500", "level": "ORG", "percentile": 96.0},
            ]),
        ],
        # Агрегаты есть только у ДРУГОЙ метрики.
        agg_payload=[_agg_dataset("ORG", "по всему банку", "Лиды")],
    )
    out = _peer_context(store, "Продажи")
    assert "по всему банку" in out
    assert "ORG" not in out


def test_name_found_for_level_missing_in_metric_aggregates(levels):
    """Данные приходят неполными: у метрики агрегаты по одному уровню, ранг — по двум."""
    levels(ORDER)
    store = _store_with(
        metrics=[
            _metric("Продажи", rankings=[
                {"rank": "5 из 40", "level": "OFFICE", "percentile": 90.0},
                {"rank": "20 из 500", "level": "ORG", "percentile": 96.0},
            ]),
        ],
        agg_payload=[
            _agg_dataset("OFFICE", "по офису", "Продажи", objs=40),
            _agg_dataset("ORG", "по всему банку", "Лиды"),
        ],
    )
    out = _peer_context(store, "Продажи")
    assert "по офису" in out and "по всему банку" in out
    assert "OFFICE" not in out and "ORG" not in out


def test_level_code_case_mismatch_between_payloads(levels):
    """Коды приходят из двух пайлоадов и могут разойтись регистром."""
    levels(ORDER)
    store = _store_with(
        metrics=[
            _metric("Продажи", rankings=[
                {"rank": "20 из 500", "level": "org", "percentile": 96.0},
            ]),
        ],
        agg_payload=[_agg_dataset("ORG", "по всему банку", "Лиды")],
    )
    assert store.level_names() == {"org": "по всему банку"}
    assert "по всему банку" in _peer_context(store, "Продажи")


def test_no_aggregates_at_all_stays_anonymous(levels):
    """Агрегатов нет вовсе — названий взять неоткуда, кода в выдаче тоже нет."""
    levels(ORDER)
    data = {"me": None, "employees": [{"fio": "Иванов", "tabnum": 1, "metrics": [
        _metric("Продажи", rankings=[
            {"rank": "20 из 500", "level": "ORG", "percentile": 96.0},
        ]),
        _metric("Продажи", date="2026-06-30", rankings=[
            {"rank": "40 из 500", "level": "ORG", "percentile": 80.0},
        ]),
    ]}]}
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    compute_analytics(store)
    out = _peer_context(store, "Продажи")
    assert "ORG" not in out
    assert "Позиция в рейтинге:" in out


def test_render_without_level_name_is_anonymous(levels):
    """Названия не пришли — группа описывается обезличенно, код не показываем."""
    levels(ORDER)
    out = _peer_context_output(with_names=False)
    assert "OFFICE" not in out and "ORG" not in out
    assert "сравнение с группой" not in out
    # Строки различаются порядком и числом объектов — колонка «группа» отпала.
    assert "| группа |" not in out
    assert "40" in out and "500" in out


def test_level_name_reaches_rows_and_store():
    payload = [{"dataset": {"level": "OFFICE", "level_name": "по офису", "metrics": [{
        "metric_id": "1", "metric_name": "Продажи",
        "aggregates": {"dt": CUR, "calc_period": "Месяц", "mean_fact": 800,
                       "children_metrics": []},
        "children_metrics": [{"metric_id": "2", "metric_name": "Лиды",
                              "aggregates": {"dt": CUR, "mean_fact": 400}}],
    }]}}]
    rows = load_aggregates_obj(payload)
    # Название наследуется всем деревом уровня, включая детей.
    assert {r["level_name"] for r in rows} == {"по офису"}
    store = SqliteStore()
    store.load_aggregates(rows)
    stored = store.conn.execute(
        "SELECT DISTINCT level, level_name FROM peer_aggregates").fetchall()
    assert [dict(r) for r in stored] == [{"level": "OFFICE", "level_name": "по офису"}]


def test_level_name_optional_in_payload():
    """Старый пайлоад без level_name грузится как раньше — значение NULL."""
    payload = [{"dataset": {"level": "OFFICE", "metrics": [{
        "metric_id": "1", "metric_name": "Продажи",
        "aggregates": {"dt": CUR, "mean_fact": 800},
    }]}}]
    rows = load_aggregates_obj(payload)
    assert rows and rows[0]["level_name"] is None


# --- Вердикты и служебная лексика ------------------------------------------

def test_verdicts_rendered_as_words():
    """Снейк-кейс вердиктов модель переносит в ответ дословно — переводим в рендере."""
    from langgraph_executor.aegra_agents.json_analyzer_v5.tools import _verdict

    assert _verdict("хуже_плана") == "хуже плана"
    assert _verdict("на_уровне_группы") == "на уровне группы"
    assert _verdict("жёсткий_план") == "жёсткий план"
    # Значения вне закрытого списка не калечим слепой заменой подчёркиваний.
    assert _verdict("какой_то_новый_статус") == "какой_то_новый_статус"
    assert _verdict(None) == ""


@pytest.mark.parametrize("with_names", [True, False])
def test_no_service_names_in_tool_output(levels, with_names):
    """Сквозная проверка: ни снейк-кейса, ни имён полей, ни кодов групп."""
    levels(ORDER)
    out = _peer_context_output(with_names=with_names)
    assert not re.search(r"[а-яё]+_[а-яё]+", out, re.IGNORECASE), out
    for field in ("mean_fact", "hit_rate", "top20_mean_fact", "total_objects", "cv"):
        assert field not in out
