"""Каузальный слой json_analyzer_causal (этапы 1-3 из исследования DoWhy).

Детерминированный (без LLM) расчёт ВКЛАДА дочерних метрик в изменение
родительской. Три режима, выбираются автоматически:

  1) algebraic  — алгебраическое waterfall-разложение изменения родителя по
     детям (вес influent_percent, нормированный; иначе равные веса). Работает
     всегда, в т.ч. для одного человека (N=1). Только stdlib. Эвристика.
  2) causal     — DoWhy gcm.distribution_change: Shapley-атрибуция сдвига
     РАСПРЕДЕЛЕНИЯ родительской метрики по когорте сотрудников между двумя
     периодами. Включается, когда сотрудников >= _MIN_PEERS и доступен dowhy.
  3) anomaly    — DoWhy gcm.attribute_anomalies: объяснение аномального
     значения ОДНОГО сотрудника относительно коллег.

DoWhy/pandas/networkx импортируются ЛЕНИВО внутри функций: нет библиотек —
граф продолжает работать в algebraic-режиме (важно для PyInstaller-бандла,
чтобы тяжёлые scipy/sklearn не тянулись, пока каузальный режим не вызван).

Направление метрики ('прямая'/'обратная') учитывается так же, как в analytics.py:
знак вклада — это эффект на ЗНАЧЕНИЕ родителя, а вердикт 'улучшение'/'ухудшение'
выводится с поправкой на metric_type, чтобы LLM не разворачивал его сам.
"""
from __future__ import annotations

from typing import Any

from .sqlite_store import SqliteStore

# Ниже этого числа сотрудников DoWhy-фиты и Shapley нестабильны — падаем в
# алгебраический режим (см. caveat исследования про короткие ряды/малые N).
_MIN_PEERS = 8


def causal_available() -> bool:
    """Доступны ли тяжёлые зависимости каузального режима (ленивая проверка)."""
    try:
        import dowhy  # noqa: F401
        import networkx  # noqa: F401
        import pandas  # noqa: F401
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------- #
# Общие helpers поверх SQLite
# --------------------------------------------------------------------------- #
def _dates(store: SqliteStore) -> list[str]:
    return [
        r["date"]
        for r in store.conn.execute(
            "SELECT DISTINCT date FROM metrics WHERE date IS NOT NULL ORDER BY date"
        )
    ]


def _resolve_dates(
    store: SqliteStore, date_old: str | None, date_new: str | None
) -> tuple[str | None, str | None]:
    dates = _dates(store)
    if not dates:
        return None, None
    new = date_new or dates[-1]
    if date_old:
        old = date_old
    else:
        before = [d for d in dates if d < new]
        old = before[-1] if before else (dates[0] if dates[0] != new else None)
    return old, new


def _direct_children(store: SqliteStore, target: str) -> list[str]:
    """Прямые дочерние метрики target по агрегатным (element IS NULL) строкам."""
    rows = store.conn.execute(
        "SELECT DISTINCT c.metric_name AS name "
        "FROM metrics p JOIN metrics c ON c.parent_uid = p.metric_uid "
        "WHERE p.metric_name = ? AND p.element IS NULL AND c.element IS NULL "
        "AND c.metric_name IS NOT NULL",
        (target,),
    ).fetchall()
    # сохраняем порядок появления, без дублей
    seen: dict[str, None] = {}
    for r in rows:
        seen.setdefault(r["name"], None)
    return list(seen.keys())


def _metric_type(store: SqliteStore, name: str) -> str | None:
    return store.metric_type_of(name)


def _agg_value(
    store: SqliteStore, pkey: Any, metric: str, date: str
) -> float | None:
    # Идентичность человека — по person_key (фолбэк на ФИО при null табельном).
    if pkey is None:
        return None
    # person_key — TEXT; приводим аргумент к строке, чтобы сравнение было
    # детерминированным, даже если кто-то передал табельный как int.
    row = store.conn.execute(
        "SELECT fact FROM metrics WHERE person_key = ? AND metric_name = ? "
        "AND element IS NULL AND date = ? AND fact IS NOT NULL LIMIT 1",
        (str(pkey), metric, date),
    ).fetchone()
    return row["fact"] if row else None


def _employee_tabnums(store: SqliteStore) -> list[Any]:
    # Возвращает person_key сотрудников (фолбэк на ФИО при null табельном).
    return [
        r["person_key"]
        for r in store.conn.execute(
            "SELECT DISTINCT person_key FROM metrics WHERE person_is_me = 0"
        )
    ]


def _self_tabnum(store: SqliteStore) -> Any | None:
    row = store.conn.execute(
        "SELECT person_key FROM metrics WHERE person_is_me = 1 LIMIT 1"
    ).fetchone()
    return row["person_key"] if row else None


def _verdict(effect_on_value: float, parent_type: str | None) -> str:
    """Вклад двигает значение родителя вверх/вниз -> улучшение/ухудшение с учётом
    направления родительской метрики (для 'обратной' рост значения = ухудшение)."""
    if effect_on_value == 0:
        return "нейтрально"
    higher_is_better = parent_type != "обратная"
    value_up = effect_on_value > 0
    return "улучшение" if (value_up == higher_is_better) else "ухудшение"


# --------------------------------------------------------------------------- #
# Этап 1: алгебраическое waterfall-разложение (stdlib-only, работает при N=1)
# --------------------------------------------------------------------------- #
def _algebraic_attribution(
    store: SqliteStore,
    target: str,
    date_old: str,
    date_new: str,
    tabnum: Any,
) -> dict[str, Any]:
    children = _direct_children(store, target)
    parent_type = _metric_type(store, target)
    p_old = _agg_value(store, tabnum, target, date_old)
    p_new = _agg_value(store, tabnum, target, date_new)
    parent_delta = (
        (p_new - p_old) if (p_old is not None and p_new is not None) else None
    )

    raw: list[dict[str, Any]] = []
    for ch in children:
        c_old = _agg_value(store, tabnum, ch, date_old)
        c_new = _agg_value(store, tabnum, ch, date_new)
        if c_old is None or c_new is None:
            continue
        delta = c_new - c_old
        infl = store.conn.execute(
            "SELECT influent_percent FROM metrics WHERE metric_name = ? "
            "AND element IS NULL AND influent_percent IS NOT NULL LIMIT 1",
            (ch,),
        ).fetchone()
        weight = infl["influent_percent"] if infl else None
        raw.append(
            {
                "node": ch,
                "metric_type": _metric_type(store, ch),
                "child_delta": round(delta, 4),
                "weight": weight,
            }
        )

    has_weights = any(r["weight"] for r in raw)
    if has_weights:
        wsum = sum((r["weight"] or 0.0) for r in raw) or 1.0
        for r in raw:
            r["_w"] = (r["weight"] or 0.0) / wsum
    else:
        n = len(raw) or 1
        for r in raw:
            r["_w"] = 1.0 / n

    contributions = []
    for r in raw:
        # «Сила кандидата» = нормированный вес × модуль изменения ребёнка.
        # ВАЖНО: знак связи ребёнок→родитель эвристике неизвестен (influent_percent
        # — только величина), поэтому НЕ заявляем эффект на родителя. Вердикт
        # описывает СОБСТВЕННОЕ движение ребёнка с учётом его metric_type.
        importance = r["_w"] * abs(r["child_delta"])
        contributions.append(
            {
                "node": r["node"],
                "child_delta": r["child_delta"],
                "weight_pct": r["weight"],
                "importance": round(importance, 4),
                "child_verdict": _verdict(r["child_delta"], r["metric_type"]),
            }
        )

    total = sum(c["importance"] for c in contributions) or 1.0
    for c in contributions:
        c["share_pct"] = round(c["importance"] / total * 100.0, 1)
    contributions.sort(key=lambda c: c["importance"], reverse=True)

    return {
        "method": "algebraic",
        "note": (
            "Эвристика ранжирования драйверов: importance = нормированный вес "
            "influent_percent × |изменение ребёнка|. Знак связи ребёнок→родитель "
            "неизвестен, поэтому child_verdict описывает движение САМОГО ребёнка "
            "(с учётом его направления), а не доказанный эффект на родителя. Для "
            "знак-аккуратной атрибуции нужен каузальный режим (когорта)."
        ),
        "target": target,
        "target_type": parent_type,
        "date_old": date_old,
        "date_new": date_new,
        "parent_old": round(p_old, 4) if p_old is not None else None,
        "parent_new": round(p_new, 4) if p_new is not None else None,
        "parent_delta": round(parent_delta, 4) if parent_delta is not None else None,
        "contributions": contributions,
    }


# --------------------------------------------------------------------------- #
# Этап 2: DoWhy distribution_change по когорте сотрудников
# --------------------------------------------------------------------------- #
def _pivot_cohort(store: SqliteStore, nodes: list[str], date: str):
    import pandas as pd

    rows = []
    for t in _employee_tabnums(store):
        rec = {n: _agg_value(store, t, n, date) for n in nodes}
        rows.append(rec)
    df = pd.DataFrame(rows, columns=nodes)
    return df.dropna()


def _causal_attribution(
    store: SqliteStore,
    target: str,
    date_old: str,
    date_new: str,
) -> dict[str, Any] | None:
    """Возвращает None, если режим неприменим (мало данных/нет связей)."""
    import networkx as nx
    from dowhy import gcm

    children = _direct_children(store, target)
    if not children:
        return None
    nodes = [target, *children]
    df_old = _pivot_cohort(store, nodes, date_old)
    df_new = _pivot_cohort(store, nodes, date_new)
    if len(df_old) < _MIN_PEERS or len(df_new) < _MIN_PEERS:
        return None
    # Убираем константные колонки (DoWhy не построит механизм по нулевой дисперсии).
    usable = [
        n
        for n in nodes
        if n == target or (df_old[n].nunique() > 1 and df_new[n].nunique() > 1)
    ]
    children_used = [n for n in usable if n != target]
    if not children_used:
        return None
    df_old = df_old[usable]
    df_new = df_new[usable]

    graph = nx.DiGraph([(ch, target) for ch in children_used])  # причина -> следствие
    model = gcm.ProbabilisticCausalModel(graph)
    gcm.auto.assign_causal_mechanisms(model, df_old)
    attributions = gcm.distribution_change(model, df_old, df_new, target)

    parent_type = _metric_type(store, target)
    # Знак distribution_change — НЕ чистый сигнал «хорошо/плохо» (величина/ранг
    # надёжнее знака). Направление эффекта на родителя выводим из данных:
    # корреляция ребёнок↔родитель в когорте × сдвиг среднего ребёнка между
    # периодами -> в какую сторону ребёнок двигал ЗНАЧЕНИЕ родителя; затем
    # verdict с учётом направления родителя.
    corr = df_old.corr(numeric_only=True)
    items = []
    for k, v in attributions.items():
        if k == target:
            continue
        mean_shift = float(df_new[k].mean() - df_old[k].mean())
        c = float(corr.loc[k, target]) if target in corr.columns else 0.0
        effect_on_parent = c * mean_shift
        items.append(
            {
                "node": k,
                "attribution": round(float(v), 5),
                "mean_shift": round(mean_shift, 4),
                "corr_with_parent": round(c, 3),
                "verdict": _verdict(effect_on_parent, parent_type),
            }
        )
    total_abs = sum(abs(it["attribution"]) for it in items) or 1.0
    for it in items:
        it["share_pct"] = round(abs(it["attribution"]) / total_abs * 100.0, 1)
    items.sort(key=lambda it: abs(it["attribution"]), reverse=True)

    return {
        "method": "causal",
        "note": (
            "DoWhy gcm.distribution_change: Shapley-атрибуция сдвига распределения "
            "родительской метрики по когорте сотрудников между двумя периодами. "
            "Знак attribution — направление эффекта на ЗНАЧЕНИЕ родителя."
        ),
        "target": target,
        "target_type": parent_type,
        "date_old": date_old,
        "date_new": date_new,
        "cohort_size": int(len(df_old)),
        "self_attribution": round(float(attributions.get(target, 0.0)), 5),
        "contributions": items,
    }


# --------------------------------------------------------------------------- #
# Этап 3: DoWhy attribute_anomalies для одного сотрудника
# --------------------------------------------------------------------------- #
def attribute_anomaly(
    store: SqliteStore,
    target: str,
    person_tabnum: Any,
    date: str | None = None,
) -> dict[str, Any]:
    if not causal_available():
        return {
            "method": "unavailable",
            "error": "Каузальный режим недоступен (нет dowhy/pandas/networkx).",
        }
    import networkx as nx
    from dowhy import gcm

    date = date or (_dates(store)[-1] if _dates(store) else None)
    if date is None:
        return {"method": "anomaly", "error": "Нет дат в датасете."}
    children = _direct_children(store, target)
    if not children:
        return {"method": "anomaly", "error": f"У метрики '{target}' нет детей."}
    nodes = [target, *children]
    df_ref = _pivot_cohort(store, nodes, date)  # коллеги как референс
    if len(df_ref) < _MIN_PEERS:
        return {
            "method": "anomaly",
            "error": f"Недостаточно коллег для референса ({len(df_ref)} < {_MIN_PEERS}).",
        }
    usable = [
        n for n in nodes if n == target or df_ref[n].nunique() > 1
    ]
    children_used = [n for n in usable if n != target]
    if not children_used:
        return {"method": "anomaly", "error": "Все дочерние метрики константны."}
    df_ref = df_ref[usable]

    one = {n: _agg_value(store, person_tabnum, n, date) for n in usable}
    if any(v is None for v in one.values()):
        return {
            "method": "anomaly",
            "error": "У сотрудника нет полного набора значений по дереву на эту дату.",
        }

    import pandas as pd

    df_one = pd.DataFrame([one], columns=usable)
    graph = nx.DiGraph([(ch, target) for ch in children_used])
    model = gcm.ProbabilisticCausalModel(graph)
    gcm.auto.assign_causal_mechanisms(model, df_ref)
    gcm.fit(model, df_ref)
    attr = gcm.attribute_anomalies(model, target, anomaly_samples=df_one)

    parent_type = _metric_type(store, target)

    def _score(scores: Any) -> float:
        return float(scores[0]) if hasattr(scores, "__len__") else float(scores)

    items = []
    for node, scores in attr.items():
        val = _score(scores)
        if node == target:
            # Вклад СОБСТВЕННОГО механизма родителя: аномалия НЕ объясняется
            # детьми (значение нетипично при типичных дочерних метриках —
            # сигнал «ищи причину вне дерева / связь изменилась»).
            items.append(
                {
                    "node": f"(собственный механизм «{target}», не объясняется детьми)",
                    "anomaly_score": round(val, 5),
                    "is_self": True,
                }
            )
        else:
            items.append({"node": node, "anomaly_score": round(val, 5), "is_self": False})
    total_abs = sum(abs(it["anomaly_score"]) for it in items) or 1.0
    for it in items:
        it["share_pct"] = round(abs(it["anomaly_score"]) / total_abs * 100.0, 1)
        it["verdict"] = (
            "—" if it["is_self"] else _verdict(it["anomaly_score"], parent_type)
        )
    items.sort(key=lambda it: abs(it["anomaly_score"]), reverse=True)

    return {
        "method": "anomaly",
        "note": (
            "DoWhy gcm.attribute_anomalies: вклад узлов в аномальность значения "
            "сотрудника относительно коллег. Строка «собственный механизм» = "
            "аномалия НЕ объясняется детьми (значение нетипично при нормальных "
            "дочерних метриках — причина вне дерева или связь изменилась)."
        ),
        "target": target,
        "target_type": parent_type,
        "date": date,
        "person_tabnum": person_tabnum,
        "reference_cohort_size": int(len(df_ref)),
        "contributions": items,
    }


# --------------------------------------------------------------------------- #
# Единая точка входа: выбирает causal vs algebraic автоматически
# --------------------------------------------------------------------------- #
def attribute_change(
    store: SqliteStore,
    target: str,
    date_old: str | None = None,
    date_new: str | None = None,
    person: Any | None = None,
) -> dict[str, Any]:
    """Главный вход для tool attribute_change.

    person задан    -> алгебраический разбор для конкретного человека (N=1 ок).
    person не задан -> пытаемся каузальный режим по когорте, иначе алгебраический
                       для руководителя (или первого сотрудника).
    """
    if _metric_type(store, target) is None:
        return {"error": f"Метрика '{target}' не найдена."}
    old, new = _resolve_dates(store, date_old, date_new)
    if old is None or new is None:
        return {"error": "Недостаточно периодов для сравнения (нужно >= 2 дат)."}

    if person is not None:
        tab = _resolve_person_tabnum(store, person)
        if tab is None:
            return {"error": f"Человек '{person}' не найден."}
        return _algebraic_attribution(store, target, old, new, tab)

    if causal_available():
        try:
            causal = _causal_attribution(store, target, old, new)
            if causal is not None:
                return causal
        except Exception as exc:  # каузальный режим хрупок — не валим tool
            fallback_tab = _self_tabnum(store) or (
                _employee_tabnums(store)[0] if _employee_tabnums(store) else None
            )
            result = (
                _algebraic_attribution(store, target, old, new, fallback_tab)
                if fallback_tab is not None
                else {"error": "Нет данных для разбора."}
            )
            result["causal_fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return result

    fallback_tab = _self_tabnum(store) or (
        _employee_tabnums(store)[0] if _employee_tabnums(store) else None
    )
    if fallback_tab is None:
        return {"error": "Нет людей в датасете."}
    result = _algebraic_attribution(store, target, old, new, fallback_tab)
    if not causal_available():
        result["causal_note"] = (
            "Каузальный режим недоступен (нет dowhy) — использован алгебраический."
        )
    return result


def _resolve_person_tabnum(store: SqliteStore, person: Any) -> Any | None:
    # Возвращает person_key (фолбэк на ФИО): на проде табельный приходит null.
    text = str(person).strip()
    if text.isdigit():
        row = store.conn.execute(
            "SELECT person_key FROM metrics "
            "WHERE person_tabnum = ? OR person_key = ? LIMIT 1",
            (int(text), text),
        ).fetchone()
        return row["person_key"] if row else None
    row = store.conn.execute(
        "SELECT person_key FROM metrics WHERE person_fio LIKE ? LIMIT 1",
        (f"%{text}%",),
    ).fetchone()
    return row["person_key"] if row else None
