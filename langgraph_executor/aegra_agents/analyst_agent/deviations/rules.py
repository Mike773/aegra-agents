"""Правила карты отклонений: что считать отклонением и насколько это важно.

Приоритет = влияние × масштаб × управляемость (методология ideal_answers.md):
- влияние: где показатель в дереве и какую долю родителя он объясняет;
- масштаб: насколько велико отклонение и подтверждено ли оно динамикой;
- управляемость: есть ли у показателя реальная цель и не является ли просадка
  общей для всей группы (системную просадку сотрудник не выправит).
"""
from __future__ import annotations

from typing import Any

# Отрицательные и положительные виды отклонений. Порядок важен только для
# читаемости — приоритет считается числом.
NEGATIVE_KINDS = (
    "below_plan",
    "declining",
    "anomaly",
    "peer_worse",
    "group_worse",
    "benchmark_gap",
    "star_missed",
    "star_at_risk",
    "segment_worst",
    "task_related",
)
POSITIVE_KINDS = (
    "above_plan",
    "improving",
    "peer_better",
    "star_received",
    "segment_best",
)

KIND_LABELS = {
    "below_plan": "хуже плана",
    "declining": "ухудшается",
    "anomaly": "аномалия на фоне коллег",
    "peer_worse": "хуже коллег",
    "group_worse": "хуже своей группы",
    "benchmark_gap": "отстаёт от сильнейших",
    "star_missed": "звезда не получена",
    "star_at_risk": "влияющий на звезду показатель ниже плана",
    "segment_worst": "худший разрез",
    "task_related": "по задаче руководителя",
    "above_plan": "лучше плана",
    "improving": "улучшается",
    "peer_better": "лучше коллег",
    "star_received": "звезда получена",
    "segment_best": "лучший разрез",
}

# Масштаб: отклонение от плана в 30 % и изменение к прошлому периоду в 20 %
# считаем «полным» масштабом — дальше расти некуда.
_PLAN_SCALE_FULL_PCT = 30.0
_POP_SCALE_FULL_PCT = 20.0
# Разрез попадает в карту, только если он отклоняется как минимум вдвое сильнее
# своего агрегата (иначе это шум внутри и без того проблемного показателя).
ELEMENT_MATERIALITY_FACTOR = 2.0
# ...и не меньше этого по модулю. Без абсолютного порога правило «вдвое сильнее»
# вырождается: у здорового агрегата отклонение около нуля, и любой шум в разрезе
# формально превышает его вдвое, заваливая карту мелочью.
ELEMENT_MIN_DEV_PCT = 10.0
# Сколько разрезов одного показателя попадает в карту. Разрез объясняет
# показатель, а не конкурирует с ним: без капа один показатель с сотнями плохих
# разрезов занимал всю карту и вытеснял агрегатные отклонения других.
MAX_ELEMENT_DEVS_PER_METRIC = 2
# Ниже этого приоритета отклонение не считается существенным для главного вывода.
MIN_MATERIAL_PRIORITY = 0.2

# Множители фокуса на задачах руководителя.
TASK_BOOST = 1.5
NON_TASK_DAMPEN = 0.6


def impact(row: dict[str, Any], share_pct: float | None) -> float:
    """Вклад показателя в общий результат по его месту в дереве."""
    kind = row.get("kind")
    if kind == "индекс":
        # Ранг/место — производная величина, сам по себе рычагом не является.
        return 0.3
    if row.get("is_star_metric"):
        return 0.8
    depth = row.get("depth") or 1
    if depth <= 1:
        return 1.0
    # Доля влияния среди сиблингов известна не всегда — тогда берём половину.
    share = (share_pct / 100.0) if share_pct else 0.5
    # Каждый уровень вглубь ослабляет вклад, но не обнуляет его.
    return max(0.15, share * (0.9 ** (depth - 2)))


def scale(row: dict[str, Any]) -> float:
    """Насколько велико отклонение: план, динамика, аномалия, устойчивость."""
    plan_pct = abs(row.get("plan_dev_pct") or 0.0)
    pop_pct = abs(row.get("pop_change_pct") or 0.0)
    by_plan = min(1.0, plan_pct / _PLAN_SCALE_FULL_PCT) if plan_pct else 0.0
    by_pop = min(1.0, pop_pct / _POP_SCALE_FULL_PCT) if pop_pct else 0.0
    value = max(by_plan, by_pop)
    if by_plan and by_pop:
        value += 0.2
    if row.get("is_anomaly"):
        value += 0.2
    if row.get("peer_status") == "хуже_коллег":
        value += 0.15
    if row.get("persistent"):
        value += 0.2
    return max(0.05, min(1.5, value))


def controllability(row: dict[str, Any]) -> float:
    """Насколько результат в руках сотрудника."""
    if row.get("rel_status") == "на_уровне_группы" and row.get("group_worsening"):
        # Падает вся группа — это системная история, а не личная.
        return 0.4
    if not row.get("has_plan"):
        return 0.6
    return 1.0


def priority(row: dict[str, Any], share_pct: float | None) -> float:
    return round(impact(row, share_pct) * scale(row) * controllability(row), 4)


def signal(row: dict[str, Any]) -> float:
    """Сила сигнала для сортировки внутри равных приоритетов."""
    for key in ("pop_change_pct", "plan_dev_pct", "benchmark_dev_pct"):
        value = row.get(key)
        if value is not None:
            return abs(value)
    return 0.0


def element_is_material(
    element_dev_pct: float | None,
    aggregate_dev_pct: float | None,
    aggregate_flagged: bool,
) -> bool:
    """Разрез существенен, если он заметно хуже своего агрегата либо агрегата
    нет вовсе и разрез сам по себе отклоняется заметно.

    Раньше проблемный агрегат пропускал ЛЮБОЙ свой разрез, и в карту попадали
    строки вроде «худший разрез, к плану +1.3 %» — ни худшие, ни существенные.
    Порог по модулю нужен и здесь: у здорового агрегата отклонение около нуля,
    и любой шум формально превышает его вдвое.
    """
    if element_dev_pct is None:
        return False
    if abs(element_dev_pct) < ELEMENT_MIN_DEV_PCT:
        return False
    if aggregate_dev_pct is None:
        # Собственного итога у показателя нет — сравнивать не с чем, судим по
        # самому разрезу (порог выше уже пройден).
        return True
    # Есть агрегат — разрез интересен, только если заметно хуже него. Флаг
    # агрегата сам по себе разрез не пропускает: иначе у проблемного показателя
    # в карту лезли все его разрезы подряд.
    return abs(element_dev_pct) >= ELEMENT_MATERIALITY_FACTOR * abs(aggregate_dev_pct)


__all__ = [
    "ELEMENT_MATERIALITY_FACTOR",
    "ELEMENT_MIN_DEV_PCT",
    "MAX_ELEMENT_DEVS_PER_METRIC",
    "KIND_LABELS",
    "MIN_MATERIAL_PRIORITY",
    "NEGATIVE_KINDS",
    "NON_TASK_DAMPEN",
    "POSITIVE_KINDS",
    "TASK_BOOST",
    "controllability",
    "element_is_material",
    "impact",
    "priority",
    "scale",
    "signal",
]
