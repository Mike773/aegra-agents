"""Разбор входного JSON в нормализованные записи (без sqlite).

Вход тот же, что у json_analyzer_v5 (см. его loader.py):
    {"me": <person>, "employees": [<person>, ...]}
    person  = {tabnum, fio, post, depart, [aggregates_ids], [rating], metrics: [<metric>]}
    rating  = [{year, quarter, level, place, staff}, ...] — место сотрудника в
              рейтинге по звёздам на уровне level (код: GOSB/TB/SBER) за квартал;
              приходит только вместе со звёздами. Названия уровней в рейтинге
              нет — они берутся из агрегатов (см. core._insert_ratings).
    metric  = {id, metric_name, metric_description, metric_type, measure_type,
               date, calc_period, fact, plan, benchmark, [ex], [rr],
               [influent_percent], element, [rankings],
               [star_received], [is_star_metric], child_metrics: [...]}

Отличие от v5: вместо одной широкой строки на узел дерева здесь три
независимых результата — КАТАЛОГ метрик (дедуп по имени), РЁБРА дерева
(персона × родитель → ребёнок, вес influent_percent) и ФАКТЫ (персона ×
метрика × дата × разрез). Дерево строится только из структуры JSON: даты узлов
на него не влияют, поэтому дочерний показатель с отстающей датой (родитель на
20-е, ребёнок на 14-е) остаётся и в дереве, и в фактах.

Дерево у КАЖДОГО человека своё: набор показателей и их вложенность у
руководителя и сотрудников могут отличаться, поэтому рёбра привязаны к
person_key. Каталог остаётся общим — показатель с одним именем один и тот же.

Хелперы нормализации (флаги, направление, пустой факт, разбор ранга) —
копия v5: новый пакет не импортирует старые (их удалят).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


# --- нормализация ---------------------------------------------------------

def norm_text(value: Any) -> str:
    """Каноническая форма имени/описания: strip, схлопнутые пробелы, casefold."""
    if value is None:
        return ""
    return " ".join(str(value).split()).casefold()


def name_key(name: Any) -> str:
    return hashlib.sha256(norm_text(name).encode("utf-8")).hexdigest()


def metric_key(name: Any, description: Any) -> str:
    raw = f"{norm_text(name)}\x00{norm_text(description)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Виды показателей: относительные проценты осмысленны только у «уровня».
# У «вклада» (знаковая величина, центр у нуля) и «индекса» (ранг, место) деление
# на базу даёт бессмыслицу и даже переворачивает вердикт динамики.
VALID_KINDS = ("уровень", "вклад", "индекс")
DEFAULT_KIND = "уровень"

_INDEX_RE = re.compile(r"\bранг\b|\bместо\b|\bпозици|\bиндекс")
_SHARE_RE = re.compile(r"\bвклад\b|\bвлияни|\bразниц|\bдельт|\bприрост\b|\bотклонени")


def guess_kind(name: Any, description: Any = None) -> str:
    """Вид показателя по его названию и описанию.

    Работает без базы знаний, поэтому проценты у рангов подавляются даже когда
    кэш трактовок недоступен. Трактовка из wiki, если она есть, вид уточняет.
    """
    text = f"{name or ''} {description or ''}".casefold()
    if _INDEX_RE.search(text):
        return "индекс"
    if _SHARE_RE.search(text):
        return "вклад"
    return DEFAULT_KIND


def _is_empty_fact(value: Any) -> bool:
    """Факт «пустой» = None или пустая строка. Ноль — валидный факт."""
    if value is None:
        return True
    return isinstance(value, str) and value.strip() == ""


_TRUE_TOKENS = frozenset({"true", "1", "да", "yes", "y"})
_FALSE_TOKENS = frozenset({"false", "0", "нет", "no", "n"})


def normalize_flag(value: Any) -> int | None:
    """Тристейт булева поля → 1 / 0 / None (None = поля не было ≠ False)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value else 0
    if isinstance(value, str):
        norm = value.strip().casefold()
        if norm in _TRUE_TOKENS:
            return 1
        if norm in _FALSE_TOKENS:
            return 0
    return None


def normalize_metric_type(value: Any) -> Any:
    """'Обратный'/'ОБРАТНАЯ' → 'обратная', 'прям*' → 'прямая'; иное — как есть."""
    if not isinstance(value, str):
        return value
    norm = value.strip().casefold()
    if norm.startswith("обратн"):
        return "обратная"
    if norm.startswith("прям"):
        return "прямая"
    return value


_RANK_RE = re.compile(r"(\d+)\D+(\d+)")


def parse_rank(raw: Any) -> tuple[int | None, int | None]:
    """«458 из 500» → (458, 500); нераспознанное → (None, None)."""
    if not isinstance(raw, str):
        return None, None
    m = _RANK_RE.search(raw)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --- записи ---------------------------------------------------------------

@dataclass
class PersonRec:
    person_key: str
    tabnum: Any
    fio: str | None
    post: str | None
    depart: str | None
    is_me: bool


@dataclass
class MetricRec:
    name: str
    norm: str
    description: str | None = None
    ext_id: str | None = None
    direction: str | None = None
    unit: str | None = None
    calc_period: str | None = None
    is_star: bool = False
    is_star_metric: bool = False
    star_of: str | None = None
    first_seen: int = 0


@dataclass
class FactRec:
    person_key: str
    metric_norm: str
    date: str | None
    element: str | None
    fact: float | None
    plan: float | None
    benchmark: float | None
    ex: float | None
    rr: float | None
    star_received: int | None
    calc_period: str | None
    src_ord: int
    rankings: list[dict[str, Any]] | None = None


@dataclass
class RatingRec:
    """Место в рейтинге по звёздам: персона × уровень × квартал."""

    person_key: str
    level: str
    year: int
    quarter: int
    place: int
    staff: int | None


@dataclass
class LoadReport:
    skipped_empty_leaves: int = 0
    description_conflicts: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParsedDataset:
    people: list[PersonRec]
    metrics: dict[str, MetricRec]          # norm(name) → запись каталога
    edges: dict[tuple[str, str, str], float | None]  # (person_key, parent_norm, child_norm) → influent_percent
    person_metrics: dict[str, list[str]]   # person_key → norm-имена его показателей в порядке JSON
    facts: list[FactRec]
    person_peer: list[tuple[str, str]]     # (person_key, aggregate_id)
    report: LoadReport
    ratings: list[RatingRec] = field(default_factory=list)


def _to_int(value: Any) -> int | None:
    """'2' → 2, 3.0 → 3; нераспознанное → None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if num != int(num):
        return None
    return int(num)


def parse_ratings(person_key: str, raw: Any) -> list[RatingRec]:
    """Поле ``rating`` персоны → записи рейтинга. Битые элементы (без уровня,
    квартала или с нечисловым местом) пропускаются — рейтинг опционален."""
    out: list[RatingRec] = []
    if not isinstance(raw, list):
        return out
    seen: set[tuple[str, int, int]] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        level = item.get("level")
        level = str(level).strip() if level is not None else ""
        year = _to_int(item.get("year"))
        quarter = _to_int(item.get("quarter"))
        place = _to_int(item.get("place"))
        if not level or year is None or quarter is None or place is None:
            continue
        key = (level, year, quarter)
        if key in seen:
            continue
        seen.add(key)
        out.append(RatingRec(
            person_key=person_key, level=level, year=year, quarter=quarter,
            place=place, staff=_to_int(item.get("staff")),
        ))
    return out


# --- разбор датасета ------------------------------------------------------

def _people(data: dict[str, Any]) -> list[dict[str, Any]]:
    """me + employees с каноническим ключом ``_key``: tabnum → fio → personN."""
    people: list[dict[str, Any]] = []
    me = data.get("me")
    if me is not None:
        people.append({**me, "_is_me": True})
    for emp in data.get("employees", []) or []:
        people.append({**emp, "_is_me": False})
    for i, person in enumerate(people):
        tab = person.get("tabnum")
        fio = person.get("fio")
        person["_key"] = str(tab) if tab is not None else (fio if fio else f"person{i}")
    return people


class _Parser:
    def __init__(self) -> None:
        self.metrics: dict[str, MetricRec] = {}
        self.edges: dict[tuple[str, str, str], float | None] = {}
        self.person_metrics: dict[str, list[str]] = {}
        self._owned: dict[str, set[str]] = {}
        self.facts: list[FactRec] = []
        self.report = LoadReport()
        self._ord = 0

    def _catalog(self, node: dict[str, Any], star_of: str | None) -> MetricRec | None:
        name = node.get("metric_name")
        norm = norm_text(name)
        if not norm:
            self.report.warnings.append("узел без metric_name пропущен")
            return None
        rec = self.metrics.get(norm)
        desc = node.get("metric_description")
        desc = desc if isinstance(desc, str) and desc.strip() else None
        if rec is None:
            rec = MetricRec(
                name=str(name).strip(),
                norm=norm,
                description=desc,
                ext_id=str(node["id"]) if node.get("id") is not None else None,
                direction=normalize_metric_type(node.get("metric_type")),
                unit=node.get("measure_type"),
                calc_period=node.get("calc_period"),
                first_seen=len(self.metrics),
            )
            self.metrics[norm] = rec
        else:
            if desc and rec.description is None:
                rec.description = desc
            elif desc and norm_text(desc) != norm_text(rec.description):
                self.report.description_conflicts += 1
            if rec.ext_id is None and node.get("id") is not None:
                rec.ext_id = str(node["id"])
            if rec.direction is None:
                rec.direction = normalize_metric_type(node.get("metric_type"))
            if rec.unit is None:
                rec.unit = node.get("measure_type")
            if rec.calc_period is None:
                rec.calc_period = node.get("calc_period")
        if normalize_flag(node.get("star_received")) is not None:
            rec.is_star = True
        if normalize_flag(node.get("is_star_metric")) == 1:
            rec.is_star_metric = True
        if star_of and rec.star_of is None:
            rec.star_of = star_of
        return rec

    def walk(
        self,
        metrics: list[dict[str, Any]],
        person_key: str,
        parent: MetricRec | None,
    ) -> None:
        for node in metrics or []:
            if not isinstance(node, dict):
                continue
            children = node.get("child_metrics") or []
            is_star_node = normalize_flag(node.get("star_received")) is not None
            # Лист без факта в базу не идёт (пустая точка затеняла бы предыдущий
            # период); узел-агрегат с детьми сохраняем; звезда факта не имеет.
            if not children and _is_empty_fact(node.get("fact")) and not is_star_node:
                self.report.skipped_empty_leaves += 1
                continue
            star_of = parent.name if (parent is not None and parent.is_star) else None
            rec = self._catalog(node, star_of)
            if rec is None:
                continue
            owned = self._owned.setdefault(person_key, set())
            if rec.norm not in owned:
                owned.add(rec.norm)
                self.person_metrics.setdefault(person_key, []).append(rec.norm)
            if parent is not None and parent.norm != rec.norm:
                key = (person_key, parent.norm, rec.norm)
                infl = _to_float(node.get("influent_percent"))
                if key not in self.edges or (self.edges[key] is None and infl is not None):
                    self.edges[key] = infl
            self._ord += 1
            rankings = node.get("rankings")
            self.facts.append(
                FactRec(
                    person_key=person_key,
                    metric_norm=rec.norm,
                    date=node.get("date"),
                    element=node.get("element") or None,
                    fact=_to_float(node.get("fact")),
                    plan=_to_float(node.get("plan")),
                    benchmark=_to_float(node.get("benchmark")),
                    ex=_to_float(node.get("ex")),
                    rr=_to_float(node.get("rr")),
                    star_received=normalize_flag(node.get("star_received")),
                    calc_period=node.get("calc_period"),
                    src_ord=self._ord,
                    rankings=rankings if isinstance(rankings, list) else None,
                )
            )
            if children:
                self.walk(children, person_key, rec)


def parse_dataset(data: dict[str, Any]) -> ParsedDataset:
    """Уже распарсенный JSON-объект → нормализованные записи."""
    parser = _Parser()
    people: list[PersonRec] = []
    person_peer: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    ratings: list[RatingRec] = []
    for person in _people(data if isinstance(data, dict) else {}):
        key = person["_key"]
        ratings.extend(parse_ratings(key, person.get("rating")))
        people.append(
            PersonRec(
                person_key=key,
                tabnum=person.get("tabnum"),
                fio=person.get("fio"),
                post=person.get("post"),
                depart=person.get("depart"),
                is_me=bool(person.get("_is_me")),
            )
        )
        parser.walk(person.get("metrics", []) or [], key, None)
        for agg_id in person.get("aggregates_ids") or []:
            if agg_id is None:
                continue
            text = str(agg_id).strip()
            if not text or (key, text) in seen_pairs:
                continue
            seen_pairs.add((key, text))
            person_peer.append((key, text))
    return ParsedDataset(
        people=people,
        metrics=parser.metrics,
        edges=parser.edges,
        person_metrics=parser.person_metrics,
        facts=parser.facts,
        person_peer=person_peer,
        report=parser.report,
        ratings=ratings,
    )


# --- агрегаты peer-групп --------------------------------------------------

_AGG_VALUE_KEYS = (
    "mean_fact",
    "mean_plan",
    "mean_ex",
    "median",
    "hit_rate",
    "top20_mean_fact",
    "iqr",
    "cv",
    "total_objects",
)


def _walk_aggregates(
    metrics: list[Any],
    level: Any,
    level_name: Any,
    aggregate_id: Any,
    rows: list[dict[str, Any]],
    counter: list[int],
    parent_uid: int | None,
    depth: int,
) -> None:
    for node in metrics or []:
        if not isinstance(node, dict):
            continue
        uid = counter[0]
        counter[0] += 1
        agg = node.get("aggregates") or {}
        base = {
            "node_uid": uid,
            "parent_node_uid": parent_uid,
            "depth": depth,
            "level": level,
            "level_name": level_name,
            "aggregate_id": aggregate_id,
            "metric_id": node.get("metric_id"),
            "metric_name": node.get("metric_name"),
        }

        def _row(src: dict[str, Any], is_current: int) -> dict[str, Any]:
            dt = src.get("dt")
            return {
                **base,
                "dt": dt,
                "ym": dt[:7] if isinstance(dt, str) and len(dt) >= 7 else None,
                "calc_period": src.get("calc_period"),
                "is_current": is_current,
                **{k: src.get(k) for k in _AGG_VALUE_KEYS},
            }

        if isinstance(agg, dict) and agg:
            rows.append(_row(agg, 1))
            for h in agg.get("history") or []:
                if isinstance(h, dict):
                    rows.append(_row(h, 0))
        children = node.get("children_metrics") or node.get("child_metrics") or []
        _walk_aggregates(children, level, level_name, aggregate_id, rows, counter, uid, depth + 1)


def parse_aggregates(data: Any) -> list[dict[str, Any]]:
    """Batch-агрегаты peer-групп → плоские строки (порт v5 load_aggregates_obj + ym).
    Битый или пустой вход → [] — агрегаты опциональны."""
    rows: list[dict[str, Any]] = []
    counter = [1]
    if not isinstance(data, list):
        return rows
    for entry in data:
        ds = entry.get("dataset") if isinstance(entry, dict) else None
        if isinstance(ds, dict):
            _walk_aggregates(
                ds.get("metrics") or [], ds.get("level"), ds.get("level_name"),
                entry.get("aggregate_id"), rows, counter, None, 1,
            )
    return rows


__all__ = [
    "DEFAULT_KIND",
    "VALID_KINDS",
    "FactRec",
    "LoadReport",
    "MetricRec",
    "ParsedDataset",
    "PersonRec",
    "RatingRec",
    "metric_key",
    "name_key",
    "norm_text",
    "normalize_flag",
    "normalize_metric_type",
    "parse_aggregates",
    "parse_dataset",
    "parse_rank",
    "parse_ratings",
    "guess_kind",
]
