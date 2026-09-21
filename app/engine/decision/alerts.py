"""
Alert ladder, plain-language explanation, and CAP 1.2 output.

Three jobs:

1. Decide WHICH step of the alert ladder a slope is on, using probability,
   exposure and -- critically -- the warning-time regime. A slope that the
   physics says will fail with no precursor cannot be handled by a "watch
   and see" step; it has to jump straight to pre-emptive action on the
   rainfall forecast.

2. Explain the decision in a sentence a district officer can act on. The
   2026 NHESS review lists the interpretability gap as one of four unsolved
   challenges in this field, for the obvious reason that nobody evacuates a
   village on the say-so of a black box.

3. Emit the alert as CAP 1.2 XML -- the Common Alerting Protocol that
   NDMA's SACHET portal and India's Cell Broadcast Alert System speak.
   Being CAP-compliant means this system feeds national alerting
   infrastructure directly rather than requiring a parallel rollout, which
   is worth more than any bespoke SMS gateway.

Alert fatigue is a design constraint, not an afterthought: every step up
the ladder has to be justified by the joint condition, and the ladder is
deliberately short.
"""

from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Sequence

from ..friction.rate_state import Regime

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"


class AlertLevel(str, Enum):
    """The four-step ladder. Short by design."""

    NONE = "none"
    WATCH = "watch"
    ADVISORY = "advisory"
    WARNING = "warning"
    EVACUATE = "evacuate"

    @property
    def rank(self) -> int:
        return {
            AlertLevel.NONE: 0,
            AlertLevel.WATCH: 1,
            AlertLevel.ADVISORY: 2,
            AlertLevel.WARNING: 3,
            AlertLevel.EVACUATE: 4,
        }[self]

    @property
    def action(self) -> str:
        return {
            AlertLevel.NONE: "No action required.",
            AlertLevel.WATCH: (
                "Monitor. No public alert. Review again at the next update."
            ),
            AlertLevel.ADVISORY: (
                "Inform local officials and residents. Avoid non-essential "
                "travel on affected stretches. Prepare response teams."
            ),
            AlertLevel.WARNING: (
                "Issue public warning. Restrict traffic on affected "
                "stretches. Move response teams into position and prepare "
                "evacuation of the most exposed structures."
            ),
            AlertLevel.EVACUATE: (
                "Evacuate exposed structures and close affected roads now. "
                "Do not wait for visible signs of movement."
            ),
        }[self]

    # CAP 1.2 controlled vocabularies
    @property
    def cap_urgency(self) -> str:
        return {
            AlertLevel.NONE: "Unknown",
            AlertLevel.WATCH: "Future",
            AlertLevel.ADVISORY: "Future",
            AlertLevel.WARNING: "Expected",
            AlertLevel.EVACUATE: "Immediate",
        }[self]

    @property
    def cap_severity(self) -> str:
        return {
            AlertLevel.NONE: "Unknown",
            AlertLevel.WATCH: "Minor",
            AlertLevel.ADVISORY: "Moderate",
            AlertLevel.WARNING: "Severe",
            AlertLevel.EVACUATE: "Extreme",
        }[self]

    @property
    def cap_certainty(self) -> str:
        return {
            AlertLevel.NONE: "Unknown",
            AlertLevel.WATCH: "Possible",
            AlertLevel.ADVISORY: "Possible",
            AlertLevel.WARNING: "Likely",
            AlertLevel.EVACUATE: "Likely",
        }[self]


@dataclass
class AlertDecision:
    """The full decision, with everything needed to justify it."""

    level: AlertLevel
    probability: float
    probability_interval: tuple[float, float]
    threshold: float
    regime: Regime
    lead_time_hours: float | None
    lead_time_label: str
    forecast_driven: bool
    headline: str
    explanation: str
    drivers: list[dict] = field(default_factory=list)
    location_name: str = ""
    latitude: float | None = None
    longitude: float | None = None

    def as_dict(self) -> dict:
        return {
            "level": self.level.value,
            "rank": self.level.rank,
            "action": self.level.action,
            "probability": round(self.probability, 4),
            "probability_interval": [
                round(self.probability_interval[0], 4),
                round(self.probability_interval[1], 4),
            ],
            "threshold": round(self.threshold, 4),
            "regime": self.regime.value,
            "lead_time_hours": self.lead_time_hours,
            "lead_time_label": self.lead_time_label,
            "forecast_driven": self.forecast_driven,
            "headline": self.headline,
            "explanation": self.explanation,
            "drivers": self.drivers,
            "location": {
                "name": self.location_name,
                "lat": self.latitude,
                "lon": self.longitude,
            },
        }


def decide(
    *,
    probability: float,
    threshold: float,
    regime: Regime,
    lead_time_hours: float | None,
    lead_time_label: str,
    probability_interval: tuple[float, float] | None = None,
    location_name: str = "",
    latitude: float | None = None,
    longitude: float | None = None,
    drivers: Sequence[dict] | None = None,
    is_forecast: bool = False,
) -> AlertDecision:
    """Choose the alert level from probability, threshold and regime.

    The rule that makes this system different
    ------------------------------------------
    For a SYNCHRONOUS slope (chi >= 4) there is no precursor to observe, so
    an observation-driven ladder is useless. Such a slope goes straight to
    EVACUATE once its probability clears threshold on a FORECAST, and is
    otherwise held at WATCH. This is the operational consequence of the
    physics, not a heuristic.
    """
    interval = probability_interval or (probability, probability)
    drivers = list(drivers or [])

    ratio = probability / threshold if threshold > 0 else 0.0

    if regime is Regime.SYNCHRONOUS:
        # No observable warning window. Only a forecast can help.
        if probability >= threshold and is_forecast:
            level = AlertLevel.EVACUATE
        elif probability >= threshold:
            # Threshold cleared on observed data alone: by the time we see
            # it, the window may already be gone. Still the highest action.
            level = AlertLevel.EVACUATE
        elif interval[1] >= threshold:
            # The upper bound clears even though the point estimate does
            # not -- worth a warning given there will be no second chance.
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.WATCH
        forecast_driven = True

    elif regime is Regime.DELAYED:
        forecast_driven = False
        if ratio >= 1.5:
            level = AlertLevel.EVACUATE
        elif probability >= threshold:
            level = AlertLevel.WARNING
        elif interval[1] >= threshold:
            level = AlertLevel.ADVISORY
        elif ratio >= 0.5:
            level = AlertLevel.WATCH
        else:
            level = AlertLevel.NONE

    elif regime is Regime.CREEP:
        forecast_driven = False
        if probability >= threshold:
            level = AlertLevel.ADVISORY
        elif ratio >= 0.5:
            level = AlertLevel.WATCH
        else:
            level = AlertLevel.NONE

    else:  # STABLE
        forecast_driven = False
        level = AlertLevel.WATCH if probability >= threshold else AlertLevel.NONE

    headline = _headline(level, location_name, regime, lead_time_label)
    explanation = _explain(
        level, probability, interval, threshold, regime,
        lead_time_label, drivers, forecast_driven,
    )

    return AlertDecision(
        level=level,
        probability=probability,
        probability_interval=interval,
        threshold=threshold,
        regime=regime,
        lead_time_hours=lead_time_hours,
        lead_time_label=lead_time_label,
        forecast_driven=forecast_driven,
        headline=headline,
        explanation=explanation,
        drivers=drivers,
        location_name=location_name,
        latitude=latitude,
        longitude=longitude,
    )


def _headline(
    level: AlertLevel,
    place: str,
    regime: Regime,
    lead_label: str,
) -> str:
    where = place or "this slope"
    if level is AlertLevel.NONE:
        return f"No landslide alert for {where}."
    if regime is Regime.SYNCHRONOUS:
        if level.rank >= AlertLevel.WARNING.rank:
            return (
                f"{level.value.upper()}: {where} - failure may occur with no "
                f"warning. Act on the rainfall forecast, not on observed "
                f"movement."
            )
        return (
            f"{level.value.upper()}: {where} - below the action threshold, "
            f"but this slope would fail without warning. Watch the forecast."
        )
    return f"{level.value.upper()}: {where} - estimated warning time {lead_label}."


def _explain(
    level: AlertLevel,
    probability: float,
    interval: tuple[float, float],
    threshold: float,
    regime: Regime,
    lead_label: str,
    drivers: Sequence[dict],
    forecast_driven: bool,
) -> str:
    parts = [
        f"Failure probability {probability:.0%} "
        f"({interval[0]:.0%}-{interval[1]:.0%} interval) against an "
        f"action threshold of {threshold:.0%}."
    ]

    if drivers:
        top = ", ".join(
            f"{d['label']} ({d['value']})" for d in list(drivers)[:3]
        )
        parts.append(f"Driven by {top}.")

    parts.append(f"Failure regime: {regime.value}; warning time {lead_label}.")

    if forecast_driven and level.rank >= AlertLevel.WARNING.rank:
        parts.append(
            "This slope is in the synchronous regime, meaning failure "
            "coincides with peak rainfall and produces no precursory "
            "movement to observe. Action must be taken ahead of the storm."
        )

    parts.append(level.action)
    return " ".join(parts)


def build_drivers(
    *,
    rain_5d_mm: float | None = None,
    rain_24h_mm: float | None = None,
    soil_wetness: float | None = None,
    slope_deg: float | None = None,
    disturbance_index: float | None = None,
    months_since_disturbance: float | None = None,
    shap_values: dict[str, float] | None = None,
) -> list[dict]:
    """Plain-language drivers for the explanation panel.

    If SHAP values are supplied (from the trained surrogate) they are used
    to rank the drivers. Otherwise the ranking falls back to a physically
    sensible ordering, so the explanation still works before a model has
    been trained.
    """
    candidates: list[dict] = []

    if rain_5d_mm is not None:
        candidates.append({
            "key": "rain_5d_mm",
            "label": "5-day cumulative rainfall",
            "value": f"{rain_5d_mm:.0f} mm",
            "raw": rain_5d_mm,
            "weight": rain_5d_mm / 300.0,
        })
    if rain_24h_mm is not None:
        candidates.append({
            "key": "rain_24h_mm",
            "label": "24-hour rainfall",
            "value": f"{rain_24h_mm:.0f} mm",
            "raw": rain_24h_mm,
            "weight": rain_24h_mm / 200.0,
        })
    if soil_wetness is not None:
        candidates.append({
            "key": "soil_wetness",
            "label": "soil wetness",
            "value": f"{soil_wetness:.2f}",
            "raw": soil_wetness,
            "weight": soil_wetness,
        })
    if slope_deg is not None:
        candidates.append({
            "key": "slope_deg",
            "label": "slope",
            "value": f"{slope_deg:.0f} degrees",
            "raw": slope_deg,
            "weight": slope_deg / 60.0,
        })
    if disturbance_index is not None and disturbance_index > 0.01:
        if months_since_disturbance is not None:
            val = (
                f"index {disturbance_index:.2f}, cut "
                f"{months_since_disturbance:.0f} months ago"
            )
        else:
            val = f"index {disturbance_index:.2f}"
        candidates.append({
            "key": "disturbance_index",
            "label": "recent slope cutting",
            "value": val,
            "raw": disturbance_index,
            "weight": disturbance_index,
        })

    if shap_values:
        for c in candidates:
            if c["key"] in shap_values:
                c["weight"] = abs(float(shap_values[c["key"]]))
                c["shap"] = round(float(shap_values[c["key"]]), 4)

    return sorted(candidates, key=lambda c: -c["weight"])


# ---------------------------------------------------------------------------
# CAP 1.2
# ---------------------------------------------------------------------------

def to_cap_xml(
    decision: AlertDecision,
    *,
    sender: str = "bhukaal@ner-lews.in",
    sender_name: str = "BHU-KAAL Landslide Early Warning System",
    radius_km: float = 2.0,
    expires_hours: float = 12.0,
    identifier: str | None = None,
    language: str = "en-IN",
    status: str = "Exercise",
) -> str:
    """Render the decision as CAP 1.2 XML.

    Parameters
    ----------
    status : CAP <status>. Defaults to "Exercise" deliberately -- a system
        that has not been authorised by a competent authority must NOT emit
        "Actual" alerts. Change this only when operating under NDMA
        authorisation.

    Returns
    -------
    str : CAP 1.2 XML document
    """
    now = datetime.now(timezone.utc)
    ident = identifier or f"BHUKAAL-{uuid.uuid4()}"

    alert = ET.Element("alert", {"xmlns": CAP_NS})
    ET.SubElement(alert, "identifier").text = ident
    ET.SubElement(alert, "sender").text = sender
    ET.SubElement(alert, "sent").text = _cap_time(now)
    ET.SubElement(alert, "status").text = status
    ET.SubElement(alert, "msgType").text = "Alert"
    ET.SubElement(alert, "scope").text = "Public"

    info = ET.SubElement(alert, "info")
    ET.SubElement(info, "language").text = language
    ET.SubElement(info, "category").text = "Geo"
    ET.SubElement(info, "event").text = "Landslide"
    ET.SubElement(info, "responseType").text = (
        "Evacuate" if decision.level is AlertLevel.EVACUATE else "Prepare"
    )
    ET.SubElement(info, "urgency").text = decision.level.cap_urgency
    ET.SubElement(info, "severity").text = decision.level.cap_severity
    ET.SubElement(info, "certainty").text = decision.level.cap_certainty

    onset = now
    if decision.lead_time_hours and decision.lead_time_hours > 0:
        onset = now + timedelta(hours=float(decision.lead_time_hours))
    ET.SubElement(info, "onset").text = _cap_time(onset)
    ET.SubElement(info, "expires").text = _cap_time(
        now + timedelta(hours=expires_hours)
    )

    ET.SubElement(info, "senderName").text = sender_name
    ET.SubElement(info, "headline").text = decision.headline
    ET.SubElement(info, "description").text = decision.explanation
    ET.SubElement(info, "instruction").text = decision.level.action

    _param(info, "failure_probability", f"{decision.probability:.4f}")
    _param(
        info, "probability_interval",
        f"{decision.probability_interval[0]:.4f}-"
        f"{decision.probability_interval[1]:.4f}",
    )
    _param(info, "action_threshold", f"{decision.threshold:.4f}")
    _param(info, "failure_regime", decision.regime.value)
    _param(info, "warning_time_hours", str(decision.lead_time_hours))
    _param(info, "warning_time_label", decision.lead_time_label)
    _param(info, "forecast_driven", str(decision.forecast_driven).lower())
    for d in decision.drivers[:5]:
        _param(info, f"driver_{d['key']}", str(d["value"]))

    area = ET.SubElement(info, "area")
    ET.SubElement(area, "areaDesc").text = (
        decision.location_name or "Unnamed slope"
    )
    if decision.latitude is not None and decision.longitude is not None:
        ET.SubElement(area, "circle").text = (
            f"{decision.latitude:.5f},{decision.longitude:.5f} {radius_km:.1f}"
        )

    ET.indent(alert, space="  ")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        + ET.tostring(alert, encoding="unicode")
    )


def _param(info: ET.Element, name: str, value: str) -> None:
    p = ET.SubElement(info, "parameter")
    ET.SubElement(p, "valueName").text = name
    ET.SubElement(p, "value").text = value


def _cap_time(dt: datetime) -> str:
    """CAP requires ISO 8601 with an explicit offset, not a trailing Z."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
