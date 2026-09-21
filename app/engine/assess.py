"""
The single entry point: rainfall in, warning decision out.

This orchestrates all four layers for one site:

    Layer B  disturbance index from detected cuts, weighted by recency
       |     -> land cover and months-since, which change root cohesion
       v
    Layer A  Monte Carlo infiltration + slope stability
       |     -> probability of failure, delta pore pressure, effective stress
       v
    Layer C  rate-and-state chi -> failure regime -> hours of warning left
       |
       v
    Layer D  cost-calibrated threshold -> alert level, explanation, CAP

Runtime is a few tens of milliseconds per site at the default draw count,
so the API runs the physics live rather than needing a trained surrogate.
The surrogate in `model/` is a scaling optimisation for grid-wide inference,
not a prerequisite.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

import numpy as np

from .decision import alerts as alert_mod
from .decision.cost import ASSET_CLASSES, bayes_threshold, ensemble_interval
from .disturbance.recency import (
    DisturbanceRecord,
    disturbance_index,
    landcover_after_disturbance,
)
from .friction import rate_state
from .physics.ensemble import SlopeUnit, scenario_from_series, simulate


def assess_site(
    *,
    slope_deg: float,
    lithology: str = "colluvium",
    rainfall_mm: Sequence[float],
    interval_hours: float = 3.0,
    antecedent_wetness: float = 0.3,
    disturbance_records: Sequence[DisturbanceRecord] | None = None,
    surcharge_kpa: float = 0.0,
    asset_class: str = "default",
    is_forecast: bool = False,
    location_name: str = "",
    latitude: float | None = None,
    longitude: float | None = None,
    as_of: date | None = None,
    n_draws: int = 96,
    seed: int | None = None,
    threshold_override: float | None = None,
) -> dict:
    """Full assessment for one slope.

    Parameters
    ----------
    slope_deg : slope angle in degrees
    lithology : key from `physics.params.LITHOLOGY_LIBRARY`
    rainfall_mm : rainfall per interval over the assessment window,
        oldest first. A 7-day window at 3-hourly resolution is 56 values.
    interval_hours : spacing of `rainfall_mm`
    antecedent_wetness : long-term infiltration as a fraction of Ksat,
        a proxy for SMAP soil wetness. 0.05 dry, 0.6 saturated.
    disturbance_records : detected cuts/clearances on this cell
    surcharge_kpa : loading from structures on the slope
    asset_class : what is exposed here; sets the cost ratio and threshold
    is_forecast : True when `rainfall_mm` includes forecast values. This
        matters: a synchronous-regime slope is only actionable on a forecast.
    threshold_override : bypass the cost-derived threshold

    Returns
    -------
    dict with `hazard`, `warning`, `disturbance`, `decision` and `inputs`.
    """
    rng = np.random.default_rng(seed)
    as_of = as_of or date.today()

    # ---- Layer B: disturbance -------------------------------------------
    dist = disturbance_index(disturbance_records or [], as_of=as_of)
    months_since = dist["months_since"] if dist["index"] > 0.01 else None
    landcover = landcover_after_disturbance(dist["dominant_kind"], months_since)

    # ---- Layer A: physics ------------------------------------------------
    unit = SlopeUnit(
        slope_deg=float(slope_deg),
        lithology=lithology,
        landcover=landcover,
        months_since_disturbance=months_since,
        surcharge_kpa=float(surcharge_kpa),
        antecedent_wetness=float(np.clip(antecedent_wetness, 0.01, 0.95)),
    )
    scenario = scenario_from_series(rainfall_mm, interval_hours)
    phys = simulate(unit, scenario, n_draws=n_draws, rng=rng)

    probability = float(phys["prob_failure"])

    # Interval from the ensemble spread. This reflects parameter
    # uncertainty, NOT model error -- see decision.cost.ensemble_interval.
    # Replace with ConformalCalibrator once a held-out set exists.
    lo, hi = _binomial_interval(probability, n_draws)

    # ---- Layer C: warning time -------------------------------------------
    warning = rate_state.assess(
        phys["delta_pressure_kpa"], phys["sigma_eff_kpa"]
    )

    # ---- Layer D: decision ------------------------------------------------
    asset = ASSET_CLASSES.get(asset_class, ASSET_CLASSES["default"])
    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else bayes_threshold(asset.cost_ratio)
    )

    desc = scenario.descriptors()
    drivers = alert_mod.build_drivers(
        rain_5d_mm=desc["rain_5d_mm"],
        rain_24h_mm=desc["rain_1d_mm"],
        soil_wetness=unit.antecedent_wetness,
        slope_deg=unit.slope_deg,
        disturbance_index=dist["index"],
        months_since_disturbance=months_since,
    )

    decision = alert_mod.decide(
        probability=probability,
        threshold=threshold,
        regime=warning.regime,
        lead_time_hours=(
            None if not np.isfinite(warning.lead_time_hours)
            else warning.lead_time_hours
        ),
        lead_time_label=warning.lead_time_label,
        probability_interval=(lo, hi),
        location_name=location_name,
        latitude=latitude,
        longitude=longitude,
        drivers=drivers,
        is_forecast=is_forecast,
    )

    return {
        "inputs": {
            "slope_deg": unit.slope_deg,
            "lithology": unit.lithology,
            "landcover": unit.landcover,
            "antecedent_wetness": unit.antecedent_wetness,
            "surcharge_kpa": unit.surcharge_kpa,
            "asset_class": asset.name,
            "cost_ratio": asset.cost_ratio,
            "is_forecast": is_forecast,
            "n_draws": n_draws,
            "rainfall": {
                k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in desc.items()
            },
        },
        "hazard": {
            "probability_of_failure": round(probability, 4),
            "probability_interval": [round(lo, 4), round(hi, 4)],
            "factor_of_safety_median": round(phys["fs_median"], 3),
            "factor_of_safety_p10": round(phys["fs_p10"], 3),
            "critical_depth_m": round(phys["critical_depth_m"], 2),
            "delta_pore_pressure_kpa": round(phys["delta_pressure_kpa"], 3),
            "effective_stress_kpa": round(phys["sigma_eff_kpa"], 2),
            "simulated_failure_hours": (
                None if not np.isfinite(phys["failure_hours"])
                else round(phys["failure_hours"], 1)
            ),
        },
        "warning": warning.as_dict(),
        "disturbance": dist,
        "decision": decision.as_dict(),
        "_decision_obj": decision,   # for CAP rendering; not serialised
    }


def _binomial_interval(p: float, n: int, z: float = 1.645) -> tuple[float, float]:
    """Wilson interval on the Monte Carlo failure fraction (90% by default).

    The probability is a binomial proportion over `n` draws, so its
    sampling uncertainty has a closed form. This is honest about how much
    of the interval comes purely from having run a finite ensemble.
    """
    if n <= 0:
        return (p, p)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def to_cap(assessment: dict, **kwargs) -> str:
    """Render an assessment as CAP 1.2 XML."""
    decision = assessment.get("_decision_obj")
    if decision is None:
        raise ValueError("assessment has no decision object")
    return alert_mod.to_cap_xml(decision, **kwargs)


def public(assessment: dict) -> dict:
    """Strip private keys for JSON serialisation."""
    return {k: v for k, v in assessment.items() if not k.startswith("_")}
