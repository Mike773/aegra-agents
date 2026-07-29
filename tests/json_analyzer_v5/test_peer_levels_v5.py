"""Справочник уровней peer-групп из конфига сервиса (shared/peer_levels.py).

Уровни инстанс-специфичны, поэтому набор задаётся окружением, а не кодом. Имя
группы подставляется НА УРОВНЕ РЕНДЕРА инструментов: модель служебных кодов
(ORG/TERR/OFFICE) вообще не видит и процитировать их не может. Порядок в конфиге
(от узкой группы к широкой) заодно задаёт референсный уровень для расчётов.
"""
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
COMPACT = "OFFICE:по офису|TERR:территория|ORG:вся организация"


@pytest.fixture
def levels(monkeypatch):
    """Ставит справочник в окружение и чистит кэш модуля до и после теста."""
    def _set(value: str | None):
        if value is None:
            monkeypatch.delenv(peer_levels.PEER_LEVELS_ENV, raising=False)
        else:
            monkeypatch.setenv(peer_levels.PEER_LEVELS_ENV, value)
        peer_levels.reset_cache()

    yield _set
    peer_levels.reset_cache()


# --- Разбор конфига ---------------------------------------------------------

def test_parse_compact_and_json():
    compact = peer_levels.parse_peer_levels(COMPACT)
    assert [(lv.code, lv.title) for lv in compact] == [
        ("OFFICE", "по офису"), ("TERR", "территория"), ("ORG", "вся организация"),
    ]
    as_json = peer_levels.parse_peer_levels(
        '[{"code": "OFFICE", "title": "по офису"}, {"level": "ORG", "name": "банк"}]'
    )
    assert [(lv.code, lv.title) for lv in as_json] == [
        ("OFFICE", "по офису"), ("ORG", "банк"),
    ]


def test_parse_caps_at_three():
    """Бизнес-требование: уровней сравнения не больше трёх, лишние отбрасываем."""
    raw = COMPACT + "|REGION:регион|WORLD:мир"
    parsed = peer_levels.parse_peer_levels(raw)
    assert len(parsed) == peer_levels.MAX_PEER_LEVELS == 3
    assert [lv.code for lv in parsed] == ["OFFICE", "TERR", "ORG"]


def test_parse_broken_input_degrades_to_empty():
    for raw in (None, "", "   ", "просто мусор", "[не json", "[]", "[{}]", ":"):
        assert peer_levels.parse_peer_levels(raw) == ()


def test_level_title_falls_back_to_code(levels):
    levels(COMPACT)
    assert peer_levels.level_title("OFFICE") == "по офису"
    assert peer_levels.level_title("office") == "по офису"  # регистр не важен
    # Уровень вне справочника не теряем и не переименовываем.
    assert peer_levels.level_title("SOMETHING") == "SOMETHING"
    assert peer_levels.level_title(None) == ""


def test_without_config_titles_are_codes(levels):
    levels(None)
    assert peer_levels.get_peer_levels() == ()
    assert peer_levels.level_title("OFFICE") == "OFFICE"
    assert peer_levels.level_order() == []


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
    # Без конфига референсом стал бы самый узкий по total_objects — ORG.
    levels(None)
    assert _level_order(rows)[0] == "ORG"
    # С конфигом — первый из справочника.
    levels(COMPACT)
    assert _level_order(rows) == ["OFFICE", "TERR", "ORG"]


def test_explicit_ref_level_beats_config(levels):
    levels(COMPACT)
    rows = _agg_rows(("ORG", 10), ("TERR", 100), ("OFFICE", 500))
    assert _level_order(rows, ref_level="TERR")[0] == "TERR"


def test_levels_outside_config_go_last(levels):
    levels("OFFICE:по офису")
    rows = _agg_rows(("ЧУЖОЙ", 5), ("OFFICE", 500))
    # Настроенный уровень впереди, ненастроенный — по прежней эвристике после него.
    assert _level_order(rows) == ["OFFICE", "ЧУЖОЙ"]


# --- Рендер инструмента -----------------------------------------------------

def _peer_context_output():
    data = {
        "me": {"fio": "Босс", "metrics": []},
        "employees": [{"fio": "Иванов", "tabnum": 1, "metrics": [{
            "id": "1", "metric_name": "Продажи", "metric_type": "прямая",
            "measure_type": "рубль", "date": CUR, "calc_period": "Месяц",
            "fact": 100.0, "plan": 100.0, "benchmark": None, "element": None,
            "child_metrics": [],
            "rankings": [{"rank": "20 из 500", "level": "ORG", "percentile": 96.0}],
        }]}],
    }
    payload = [
        {"dataset": {"level": lvl, "metrics": [{
            "metric_id": "1", "metric_name": "Продажи",
            "aggregates": {"dt": CUR, "calc_period": "Месяц", "mean_fact": 800,
                           "median": 700, "top20_mean_fact": 2000, "hit_rate": 26.0,
                           "total_objects": objs},
        }]}}
        for lvl, objs in (("OFFICE", 40), ("ORG", 500))
    ]
    store = SqliteStore()
    store.load(load_dataset_obj(data))
    store.load_aggregates(load_aggregates_obj(payload))
    compute_analytics(store)
    tool = next(
        t for t in build_tools(store, EmbeddingIndex([]), lambda q: [0.0])
        if t.name == "peer_context"
    )
    return tool.invoke({"metric": "Продажи"})


def test_render_uses_titles_not_codes(levels):
    levels(COMPACT)
    out = _peer_context_output()
    assert "по офису" in out
    assert "вся организация" in out
    # Служебные коды до модели не доходят.
    assert "OFFICE" not in out
    assert "ORG" not in out


def test_render_without_config_keeps_codes(levels):
    """Справочник не задан — поведение прежнее, уровень виден как код."""
    levels(None)
    out = _peer_context_output()
    assert "OFFICE" in out or "ORG" in out


# --- Вердикты и заголовки колонок ------------------------------------------

def test_verdicts_rendered_as_words():
    """Снейк-кейс вердиктов модель переносит в ответ дословно — переводим в рендере."""
    from langgraph_executor.aegra_agents.json_analyzer_v5.tools import _verdict

    assert _verdict("хуже_плана") == "хуже плана"
    assert _verdict("на_уровне_группы") == "на уровне группы"
    assert _verdict("жёсткий_план") == "жёсткий план"
    # Значения вне закрытого списка не калечим слепой заменой подчёркиваний.
    assert _verdict("какой_то_новый_статус") == "какой_то_новый_статус"
    assert _verdict(None) == ""


def test_no_service_names_in_tool_output(levels):
    """Сквозная проверка: в выдаче инструмента нет ни снейк-кейса, ни имён полей."""
    import re

    levels(COMPACT)
    out = _peer_context_output()
    assert not re.search(r"[а-яё]+_[а-яё]+", out, re.IGNORECASE), out
    for field in ("mean_fact", "hit_rate", "top20_mean_fact", "total_objects", "cv"):
        assert field not in out
