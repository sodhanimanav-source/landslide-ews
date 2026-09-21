"""
BHU-KAAL v2 API: warning-time-aware landslide assessment.

The v1 endpoints are left untouched so nothing that depends on them breaks.
Everything new lives under /api/v2.

What is different from v1
-------------------------
v1 answers "how risky is this slope, 0-100?" from a Random Forest trained
on a hand-written synthetic formula. v2 answers:

  - what is the probability of failure, with an interval
  - how many HOURS of warning remain, from rate-and-state friction
  - was this slope cut recently, and how much does that matter
  - what should someone actually do, at a threshold derived from the cost
    of being wrong for THIS asset
  - and it emits the result as CAP 1.2 for national alerting infrastructure
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.engine import rainfall as rain_mod
from app.engine.assess import assess_site, public, to_cap
from app.engine.decision.cost import ASSET_CLASSES, bayes_threshold
from app.engine.disturbance.recency import (
    DisturbanceRecord,
    DisturbanceType,
    peak_vulnerability_months,
    summarise_curve,
)
from app.engine.friction.rate_state import (
    CHI_DELAYED,
    CHI_SYNCHRONOUS,
    Regime,
    inverse_velocity_failure_time,
)
from app.engine.physics.params import LITHOLOGY_LIBRARY
from app.engine.sites import all_sites, get_site

router = APIRouter(prefix="/api/v2", tags=["BHU-KAAL v2"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DisturbanceIn(BaseModel):
    detected_on: date
    kind: str = "unknown"
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    area_fraction: float = Field(1.0, ge=0.0, le=1.0)
    source: str = "manual"


class AssessIn(BaseModel):
    slope_deg: float = Field(..., ge=0.0, le=89.0)
    lithology: str = "colluvium"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_name: str = ""
    asset_class: str = "default"
    surcharge_kpa: float = Field(0.0, ge=0.0)
    # Either supply rainfall directly, or give lat/lon and let the API fetch it
    rainfall_mm: Optional[List[float]] = None
    interval_hours: float = 3.0
    antecedent_wetness: Optional[float] = Field(None, ge=0.01, le=0.95)
    forecast_hours: int = Field(0, ge=0, le=48)
    disturbances: List[DisturbanceIn] = Field(default_factory=list)
    n_draws: int = Field(96, ge=16, le=512)


class InverseVelocityIn(BaseModel):
    times_hours: List[float]
    displacement_mm: List[float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _records(items) -> list[DisturbanceRecord]:
    out = []
    for d in items:
        try:
            kind = DisturbanceType(d.kind)
        except ValueError:
            kind = DisturbanceType.UNKNOWN
        out.append(
            DisturbanceRecord(
                detected_on=d.detected_on,
                kind=kind,
                confidence=d.confidence,
                area_fraction=d.area_fraction,
                source=d.source,
            )
        )
    return out


def _assess_site_object(site, *, forecast_hours: int = 0, n_draws: int = 96):
    series = rain_mod.fetch(
        site.latitude, site.longitude, forecast_hours=forecast_hours
    )
    result = assess_site(
        slope_deg=site.slope_deg,
        lithology=site.lithology,
        rainfall_mm=series.mm_per_interval,
        interval_hours=series.interval_hours,
        antecedent_wetness=series.soil_wetness or 0.3,
        disturbance_records=list(site.disturbances),
        surcharge_kpa=site.surcharge_kpa,
        asset_class=site.asset_class,
        is_forecast=series.includes_forecast,
        location_name=site.name,
        latitude=site.latitude,
        longitude=site.longitude,
        n_draws=n_draws,
    )
    result["site"] = site.as_dict()
    result["rainfall_source"] = series.as_dict()
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/sites")
def list_sites():
    """Registry of monitored corridors and urban zones."""
    return [s.as_dict() for s in all_sites()]


@router.get("/corridors")
def assess_all(
    forecast_hours: int = Query(
        0, ge=0, le=48,
        description="Hours of forecast rainfall to include. Non-zero is the "
                    "only mode that can help a synchronous-regime slope.",
    ),
    n_draws: int = Query(64, ge=16, le=256),
):
    """Live assessment of every monitored site.

    This replaces the hardcoded risk strings that used to sit in the
    dashboard template. Every number here is computed from live rainfall.
    """
    out = []
    for site in all_sites():
        try:
            result = _assess_site_object(
                site, forecast_hours=forecast_hours, n_draws=n_draws
            )
            d = result["decision"]
            w = result["warning"]
            h = result["hazard"]
            out.append({
                **site.as_dict(),
                "alert_level": d["level"],
                "alert_rank": d["rank"],
                "probability_of_failure": h["probability_of_failure"],
                "probability_interval": h["probability_interval"],
                "threshold": d["threshold"],
                "regime": w["regime"],
                "lead_time_hours": w["lead_time_hours"],
                "lead_time_label": w["lead_time_label"],
                "disturbance_index": result["disturbance"]["index"],
                "months_since_disturbance": result["disturbance"]["months_since"],
                "headline": d["headline"],
                "rainfall": result["rainfall_source"],
            })
        except Exception as exc:  # one bad site must not break the map
            out.append({**site.as_dict(), "error": str(exc)})
    return out


@router.get("/corridors/{site_id}")
def assess_one(
    site_id: str,
    forecast_hours: int = Query(0, ge=0, le=48),
    n_draws: int = Query(128, ge=16, le=512),
):
    """Full assessment for one registered site, including all four layers."""
    site = get_site(site_id)
    if site is None:
        raise HTTPException(404, f"unknown site: {site_id}")
    return public(_assess_site_object(
        site, forecast_hours=forecast_hours, n_draws=n_draws
    ))


@router.post("/assess")
def assess_arbitrary(payload: AssessIn):
    """Assess any slope, with supplied or fetched rainfall."""
    if payload.rainfall_mm:
        series = rain_mod.from_values(
            payload.rainfall_mm,
            interval_hours=payload.interval_hours,
            includes_forecast=payload.forecast_hours > 0,
        )
    elif payload.latitude is not None and payload.longitude is not None:
        series = rain_mod.fetch(
            payload.latitude, payload.longitude,
            forecast_hours=payload.forecast_hours,
        )
    else:
        raise HTTPException(
            422, "supply either rainfall_mm, or latitude and longitude"
        )

    wetness = (
        payload.antecedent_wetness
        if payload.antecedent_wetness is not None
        else (series.soil_wetness or 0.3)
    )

    result = assess_site(
        slope_deg=payload.slope_deg,
        lithology=payload.lithology,
        rainfall_mm=series.mm_per_interval,
        interval_hours=series.interval_hours,
        antecedent_wetness=wetness,
        disturbance_records=_records(payload.disturbances),
        surcharge_kpa=payload.surcharge_kpa,
        asset_class=payload.asset_class,
        is_forecast=series.includes_forecast,
        location_name=payload.location_name,
        latitude=payload.latitude,
        longitude=payload.longitude,
        n_draws=payload.n_draws,
    )
    result["rainfall_source"] = series.as_dict()
    return public(result)


@router.get("/corridors/{site_id}/cap.xml")
def cap_for_site(
    site_id: str,
    forecast_hours: int = Query(0, ge=0, le=48),
    status: str = Query(
        "Exercise",
        description="CAP <status>. Must stay 'Exercise' until this system "
                    "is authorised by a competent authority.",
    ),
):
    """CAP 1.2 alert for a site, ready for SACHET or a cell-broadcast gateway."""
    site = get_site(site_id)
    if site is None:
        raise HTTPException(404, f"unknown site: {site_id}")
    result = _assess_site_object(site, forecast_hours=forecast_hours)
    xml = to_cap(result, status=status)
    return Response(content=xml, media_type="application/xml")


@router.get("/hindcast/cyclone-remal")
def hindcast_remal(n_draws: int = Query(256, ge=32, le=1024)):
    """Validation benchmark: Cyclone Remal over Aizawl, 28 May 2024.

    205 mm in 24 hours, eight near-simultaneous failures. This event is
    held out of all training and calibration. Running it shows what the
    system would have published, and when.
    """
    site = get_site("aizawl-ridge")
    series = rain_mod.cyclone_remal_aizawl()
    result = assess_site(
        slope_deg=site.slope_deg,
        lithology=site.lithology,
        rainfall_mm=series.mm_per_interval,
        interval_hours=series.interval_hours,
        antecedent_wetness=series.soil_wetness or 0.45,
        disturbance_records=list(site.disturbances),
        surcharge_kpa=site.surcharge_kpa,
        asset_class=site.asset_class,
        is_forecast=True,
        location_name=f"{site.name} (Cyclone Remal hindcast)",
        latitude=site.latitude,
        longitude=site.longitude,
        n_draws=n_draws,
        seed=20240528,
    )
    result["rainfall_source"] = series.as_dict()
    result["event"] = {
        "name": "Cyclone Remal",
        "date": "2024-05-28",
        "location": "Aizawl, Mizoram",
        "observed_rainfall_mm_24h": 205,
        "observed_landslides": 8,
        "note": (
            "Held out of training and calibration. The published analysis "
            "places this event in the synchronous regime, meaning no "
            "observation-based system could have provided a warning window."
        ),
    }
    return public(result)


@router.post("/inverse-velocity")
def inverse_velocity(payload: InverseVelocityIn):
    """Fukuzono countdown from a displacement time series.

    Rate-and-state tells you which regime a slope is in. This tells you how
    long is left once it has actually started moving.
    """
    if len(payload.times_hours) != len(payload.displacement_mm):
        raise HTTPException(422, "times and displacement must be equal length")
    return inverse_velocity_failure_time(
        payload.times_hours, payload.displacement_mm
    )


@router.get("/vulnerability-curve")
def vulnerability_curve():
    """The disturbance recency curve -- the shape that IS the argument.

    Plot this on the dashboard. It shows why a slope cut four months ago is
    treated differently from the same slope cut six years ago, which is the
    gap NASA's LHASA v2 documents in itself.
    """
    return {
        "curve": summarise_curve(),
        "root_strength_minimum_months": round(peak_vulnerability_months(), 1),
        "note": (
            "Weight 1.0 = maximum vulnerability. Combines the immediate "
            "mechanical effect of the cut with the delayed loss of root "
            "reinforcement."
        ),
    }


@router.get("/regimes")
def regimes():
    """The three failure regimes and what each one means operationally."""
    return {
        "chi_definition": "chi = mu_0 * delta_P / (a * sigma_eff)",
        "thresholds": {
            "synchronous_at_or_above": CHI_SYNCHRONOUS,
            "delayed_at_or_above": CHI_DELAYED,
        },
        "regimes": [
            {
                "name": r.value,
                "description": r.description,
            }
            for r in (Regime.SYNCHRONOUS, Regime.DELAYED, Regime.CREEP, Regime.STABLE)
        ],
        "source": (
            "Frictional timescales and the impact of climate change-driven "
            "extreme weather on rainfall-triggered landslides in Mizoram, "
            "NE India (2026), arXiv:2606.23281"
        ),
    }


@router.get("/thresholds")
def thresholds():
    """Cost-derived alert thresholds per asset class.

    A hospital trips at a far lower probability than a footpath. This is
    what distinguishes risk from hazard, and it is a policy input a
    district officer can change -- not a constant buried in the code.
    """
    return [
        {
            "asset_class": name,
            "cost_ratio_missed_vs_false_alarm": a.cost_ratio,
            "threshold": round(bayes_threshold(a.cost_ratio), 4),
            "population_weight": a.population_weight,
        }
        for name, a in sorted(
            ASSET_CLASSES.items(), key=lambda kv: -kv[1].cost_ratio
        )
    ]


@router.get("/lithologies")
def lithologies():
    """Lithology library with its geotechnical parameter distributions."""
    return [
        {
            "code": code,
            "name": lp.name,
            "strength_rank": lp.strength_rank,
            "cohesion_kpa_mean_sd": lp.cohesion_kpa,
            "friction_deg_mean_sd": lp.friction_deg,
            "soil_depth_m_mean_sd": lp.soil_depth_m,
        }
        for code, lp in LITHOLOGY_LIBRARY.items()
    ]
