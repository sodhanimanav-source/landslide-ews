"""
Anthropogenic slope disturbance, weighted by how recently it happened.

Why this layer exists
---------------------
SIH26001 names "unplanned hill cutting" as a driver of landslides in the
North East. The world's leading operational global landslide model, NASA's
LHASA version 2, states in its own documentation that it "does not
incorporate any measures of anthropogenic disturbance", and attributes its
weak performance on multitemporal inventories to exactly that omission.
That is an open gap in the state of the art, and it happens to sit directly
on top of the problem we were asked to solve.

The key design decision: presence is not the signal, RECENCY is
------------------------------------------------------------------
A slope cut five years ago has re-vegetated, drained and partially
re-stabilised. The same slope cut four months ago has no root
reinforcement, no drainage, and loose spoil on it. Treating those two as
one binary "disturbed" flag throws away most of the information.

So we model a window of vulnerability. Two processes run at once:

  - residual roots from the removed vegetation decay away
    (time constant ~12 months)
  - new roots from whatever grows back slowly establish
    (time constant ~36 months)

Their combination has a minimum some months after clearance. This is a real
and measured effect: the forestry literature documents the same "window of
vulnerability" after clear-felling, and it is why we can justify the shape
of the curve rather than inventing a weight.

Crucially, this feeds the PHYSICS (through root cohesion in
`physics.stability`) rather than being bolted on as another opaque ML
feature. The disturbance layer changes the factor of safety through a
mechanism we can write down.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Iterable, Sequence

import numpy as np

# Time constants, months. Shared with physics.stability so the two layers
# cannot drift apart.
ROOT_DECAY_TAU_MONTHS = 12.0
ROOT_RECOVERY_TAU_MONTHS = 36.0

# Decay of the purely geometric destabilisation from the cut itself, months.
# Humid-tropical cut faces re-equilibrate and colonise relatively fast.
MECHANICAL_TAU_MONTHS = 30.0

# Beyond this the slope is treated as recovered and no longer flagged.
# At 120 months the combined weight is already about 0.05.
DISTURBANCE_MEMORY_MONTHS = 120.0


class DisturbanceType(str, Enum):
    """What happened to the slope. Severity differs by mechanism."""

    ROAD_CUT = "road_cut"
    BUILDING_CUT = "building_cut"
    QUARRY = "quarry"
    FOREST_LOSS = "forest_loss"
    AGRICULTURE = "agriculture"
    BURN = "burn"
    UNKNOWN = "unknown"

    @property
    def severity(self) -> float:
        """Multiplier on the disturbance weight, 0-1.

        A benched road cut removes toe support and steepens the face, so it
        is worse than losing tree cover alone.
        """
        return {
            DisturbanceType.ROAD_CUT: 1.00,
            DisturbanceType.BUILDING_CUT: 0.95,
            DisturbanceType.QUARRY: 1.00,
            DisturbanceType.FOREST_LOSS: 0.65,
            DisturbanceType.AGRICULTURE: 0.45,
            DisturbanceType.BURN: 0.55,
            DisturbanceType.UNKNOWN: 0.60,
        }[self]


@dataclass(frozen=True)
class DisturbanceRecord:
    """One detected disturbance event on a grid cell."""

    detected_on: date
    kind: DisturbanceType = DisturbanceType.UNKNOWN
    confidence: float = 1.0          # 0-1, from the detector
    area_fraction: float = 1.0       # fraction of the cell affected
    source: str = "unknown"          # sentinel1_coherence, sentinel2_ndvi, ...

    def months_before(self, as_of: date) -> float:
        days = (as_of - self.detected_on).days
        return max(days, 0) / 30.44


def root_strength_shortfall(
    months_since: np.ndarray | float,
    *,
    decay_tau: float = ROOT_DECAY_TAU_MONTHS,
    recovery_tau: float = ROOT_RECOVERY_TAU_MONTHS,
) -> np.ndarray:
    """Loss of root reinforcement relative to mature cover, 0-1.

    Two processes: residual dead roots decay away, new roots establish.

        reinforcement(t) = exp(-t/decay_tau) + (1 - exp(-t/recovery_tau))
        shortfall(t)     = 1 - reinforcement(t)

    The shortfall is zero at the instant of cutting -- dead roots still
    hold -- rises to a maximum after a year or two, then falls as the new
    stand matures. That delayed minimum in strength is the classic
    "window of vulnerability" from the forestry literature.
    """
    t = np.maximum(np.asarray(months_since, dtype=float), 0.0)
    residual = np.exp(-t / decay_tau)
    recovering = 1.0 - np.exp(-t / recovery_tau)
    reinforcement = np.clip(residual + recovering, 0.0, 1.0)
    return np.clip(1.0 - reinforcement, 0.0, 1.0)


def mechanical_effect(
    months_since: np.ndarray | float,
    *,
    tau: float = MECHANICAL_TAU_MONTHS,
) -> np.ndarray:
    """Destabilisation from the cut geometry itself, 0-1.

    Unlike root loss, this is immediate: excavating a bench removes toe
    support, steepens the face and leaves loose spoil the moment the
    machine stops. It then decays as the slope re-equilibrates, surface
    drainage develops and vegetation colonises the face.
    """
    t = np.maximum(np.asarray(months_since, dtype=float), 0.0)
    return np.exp(-t / tau)


def recency_weight(
    months_since: np.ndarray | float,
    *,
    decay_tau: float = ROOT_DECAY_TAU_MONTHS,
    recovery_tau: float = ROOT_RECOVERY_TAU_MONTHS,
    mechanical_tau: float = MECHANICAL_TAU_MONTHS,
) -> np.ndarray:
    """Vulnerability weight in [0, 1] as a function of time since disturbance.

    Combines the two mechanisms with a noisy-OR, because either one alone
    is enough to destabilise a slope:

        weight = 1 - (1 - mechanical) * (1 - root_shortfall)

    The result is 1.0 at the moment of cutting, stays high through the
    first two or three monsoons, and decays to near zero after several
    years. That shape is the entire argument for modelling recency rather
    than a binary disturbed/undisturbed flag.

    Parameters
    ----------
    months_since : months elapsed since the disturbance

    Returns
    -------
    weight : 0 = recovered, 1 = maximum vulnerability
    """
    t = np.maximum(np.asarray(months_since, dtype=float), 0.0)

    mech = mechanical_effect(t, tau=mechanical_tau)
    root = root_strength_shortfall(
        t, decay_tau=decay_tau, recovery_tau=recovery_tau
    )
    weight = 1.0 - (1.0 - mech) * (1.0 - root)

    # Nothing is flagged forever.
    return np.where(t > DISTURBANCE_MEMORY_MONTHS, 0.0, np.clip(weight, 0.0, 1.0))


def peak_vulnerability_months(
    *,
    decay_tau: float = ROOT_DECAY_TAU_MONTHS,
    recovery_tau: float = ROOT_RECOVERY_TAU_MONTHS,
) -> float:
    """When ROOT reinforcement is at its weakest, in months.

    Solves d/dt [residual + recovering] = 0 analytically:

        t* = ln(decay_tau / recovery_tau) / (1/recovery_tau - 1/decay_tau)

    Worth quoting in your report: the combined vulnerability peaks
    immediately after cutting, but root strength alone bottoms out roughly
    a year and a half later -- which tells a planner when a slope that
    survived its first monsoon is still not safe.
    """
    if np.isclose(decay_tau, recovery_tau):
        return float(decay_tau)
    num = np.log(decay_tau / recovery_tau)
    den = (1.0 / recovery_tau) - (1.0 / decay_tau)
    return float(num / den)


def disturbance_index(
    records: Iterable[DisturbanceRecord],
    as_of: date | datetime | None = None,
) -> dict:
    """Combine all detections on a cell into one index and its provenance.

    Multiple detections are combined with a noisy-OR rather than a sum, so
    that three independent detectors seeing the same cut do not triple-count
    it, while genuinely separate events do accumulate.

    Returns
    -------
    dict with keys:
        index               0-1 combined disturbance weight
        months_since        months since the most recent detection, or None
        dominant_kind       the disturbance type driving the index
        n_records           how many detections contributed
        contributions       per-record breakdown, for the explanation panel
    """
    if as_of is None:
        as_of = date.today()
    if isinstance(as_of, datetime):
        as_of = as_of.date()

    recs = list(records)
    if not recs:
        return {
            "index": 0.0,
            "months_since": None,
            "dominant_kind": None,
            "n_records": 0,
            "contributions": [],
        }

    contributions = []
    survival = 1.0
    best = (0.0, None)

    for r in recs:
        months = r.months_before(as_of)
        w = float(recency_weight(months))
        contribution = (
            w
            * r.kind.severity
            * float(np.clip(r.confidence, 0.0, 1.0))
            * float(np.clip(r.area_fraction, 0.0, 1.0))
        )
        survival *= (1.0 - contribution)
        contributions.append(
            {
                "detected_on": r.detected_on.isoformat(),
                "kind": r.kind.value,
                "source": r.source,
                "months_since": round(months, 1),
                "recency_weight": round(w, 3),
                "contribution": round(contribution, 3),
            }
        )
        if contribution > best[0]:
            best = (contribution, r.kind)

    index = float(np.clip(1.0 - survival, 0.0, 1.0))
    most_recent = min(r.months_before(as_of) for r in recs)

    return {
        "index": round(index, 4),
        "months_since": round(most_recent, 1),
        "dominant_kind": best[1].value if best[1] else None,
        "n_records": len(recs),
        "contributions": sorted(
            contributions, key=lambda c: -c["contribution"]
        ),
    }


def effective_months_since(index_result: dict) -> float | None:
    """Months-since value to hand to the physics layer.

    The physics layer wants a single "time since the slope was cut". When
    several events overlap we pass the most recent, because that is the one
    that reset the clock on root reinforcement.
    """
    return index_result.get("months_since")


def landcover_after_disturbance(
    kind: DisturbanceType | str | None,
    months_since: float | None,
) -> str:
    """Best guess at current land cover, for the root-cohesion model.

    Freshly cut ground is bare; a few years on, scrub has usually taken
    hold unless the surface is sealed (road, building, quarry floor).
    """
    if kind is None or months_since is None:
        return "dense_forest"
    if isinstance(kind, str):
        try:
            kind = DisturbanceType(kind)
        except ValueError:
            kind = DisturbanceType.UNKNOWN

    if kind in (
        DisturbanceType.ROAD_CUT,
        DisturbanceType.BUILDING_CUT,
        DisturbanceType.QUARRY,
    ):
        return "built" if months_since > 12 else "bare"

    if months_since < 6:
        return "bare"
    if months_since < 36:
        return "shrub"
    return "open_forest"


def summarise_curve(
    months: Sequence[float] = (0, 3, 6, 9, 12, 18, 24, 36, 48, 60, 84),
) -> list[dict]:
    """The vulnerability curve as plottable rows.

    Handy for the dashboard and for the slide that explains why recency
    matters -- the shape of this curve IS the argument.
    """
    return [
        {"months_since": float(m), "weight": round(float(recency_weight(m)), 4)}
        for m in months
    ]
