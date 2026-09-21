"""
Rainfall ingestion for the physics engine.

The engine needs a hyetograph -- rainfall per interval over an assessment
window -- not a single "current rain" number. A single instantaneous value
cannot distinguish 50 mm falling in one hour from the same 50 mm spread
over a day, and those two produce completely different slope responses.
That distinction is the whole reason the physics layer exists.

Source here is Open-Meteo (already used elsewhere in this project, free, no
key). In production this should be IMD gridded rainfall and automatic rain
gauges, with GPM IMERG Early as the satellite fallback -- IMERG has roughly
four hours of latency and is known to underestimate short intense
convective cells, which is exactly why gauge data matters in the NER.

Everything degrades safely: if the network is unavailable the caller gets a
clearly-labelled synthetic series rather than an exception, and the label
travels with the result so the UI can show data provenance honestly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

WINDOW_DAYS = 7
INTERVAL_HOURS = 3.0
N_INTERVALS = int(WINDOW_DAYS * 24 / INTERVAL_HOURS)  # 56


@dataclass(frozen=True)
class RainfallSeries:
    """A hyetograph with its provenance attached."""

    mm_per_interval: np.ndarray
    interval_hours: float
    source: str                  # observed | forecast | observed+forecast | synthetic
    includes_forecast: bool
    elevation_m: float | None = None
    soil_wetness: float | None = None

    @property
    def total_mm(self) -> float:
        return float(self.mm_per_interval.sum())

    @property
    def last_24h_mm(self) -> float:
        k = int(24 / self.interval_hours)
        return float(self.mm_per_interval[-k:].sum())

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "includes_forecast": self.includes_forecast,
            "interval_hours": self.interval_hours,
            "n_intervals": int(self.mm_per_interval.size),
            "total_mm": round(self.total_mm, 1),
            "last_24h_mm": round(self.last_24h_mm, 1),
            "elevation_m": self.elevation_m,
            "soil_wetness": (
                None if self.soil_wetness is None else round(self.soil_wetness, 3)
            ),
        }


def _to_intervals(hourly_mm: Sequence[float], interval_hours: float) -> np.ndarray:
    """Aggregate an hourly series into the engine's interval resolution."""
    a = np.asarray(hourly_mm, dtype=float)
    a = np.nan_to_num(a, nan=0.0)
    step = int(interval_hours)
    n_full = (a.size // step) * step
    if n_full == 0:
        return np.zeros(1, dtype=float)
    return a[:n_full].reshape(-1, step).sum(axis=1)


def estimate_soil_wetness(mm_per_interval: np.ndarray, interval_hours: float) -> float:
    """Antecedent wetness proxy from the rainfall series itself.

    A stand-in for SMAP L4 root-zone soil wetness, built as a decaying
    antecedent precipitation index over the window and squashed to the
    0.05-0.9 range the physics layer expects. Replace with real SMAP as
    soon as you ingest it -- LHASA v2 found soil wetness worth including
    even though current rainfall dominated.
    """
    mm = np.asarray(mm_per_interval, dtype=float)
    if mm.size == 0:
        return 0.2
    # Exponential decay with a ~3-day memory, excluding the most recent day
    # so this represents ANTECEDENT state rather than the current storm.
    k = int(24 / interval_hours)
    antecedent = mm[:-k] if mm.size > k else mm
    if antecedent.size == 0:
        return 0.2
    age = np.arange(antecedent.size)[::-1] * interval_hours / 24.0
    api = float(np.sum(antecedent * np.exp(-age / 3.0)))
    # 150 mm of decayed antecedent rain maps to near-saturation.
    return float(np.clip(0.05 + 0.85 * (1.0 - np.exp(-api / 80.0)), 0.05, 0.9))


def fetch(
    latitude: float,
    longitude: float,
    *,
    past_days: int = WINDOW_DAYS,
    forecast_hours: int = 0,
    interval_hours: float = INTERVAL_HOURS,
    timeout: float = 6.0,
) -> RainfallSeries:
    """Fetch a rainfall hyetograph for one location.

    Parameters
    ----------
    forecast_hours : hours of FORECAST rainfall to append. Non-zero makes
        the assessment forecast-driven, which is the only mode that can
        help a synchronous-regime slope.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "precipitation",
        "past_days": min(max(past_days, 1), 92),
        "forecast_days": 2 if forecast_hours > 0 else 1,
        "timezone": "UTC",
    }
    try:
        res = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
        res.raise_for_status()
        data = res.json()
        hourly = data.get("hourly", {}).get("precipitation") or []
        elevation = data.get("elevation")
        if not hourly:
            raise ValueError("no precipitation array returned")

        # Open-Meteo returns past_days + forecast_days of hourly values.
        # Keep the trailing window ending at now + forecast_hours.
        n_keep = past_days * 24 + int(forecast_hours)
        series_h = np.asarray(hourly[-n_keep:], dtype=float)
        mm = _to_intervals(series_h, interval_hours)

        source = "observed+forecast" if forecast_hours > 0 else "observed"
        return RainfallSeries(
            mm_per_interval=mm,
            interval_hours=interval_hours,
            source=source,
            includes_forecast=forecast_hours > 0,
            elevation_m=elevation,
            soil_wetness=estimate_soil_wetness(mm, interval_hours),
        )

    except Exception:
        return synthetic(interval_hours=interval_hours)


def synthetic(
    total_mm: float = 30.0,
    *,
    interval_hours: float = INTERVAL_HOURS,
    n_intervals: int = N_INTERVALS,
) -> RainfallSeries:
    """Clearly-labelled fallback series for when the network is unavailable.

    The UI must show this as synthetic. A system that silently substitutes
    made-up rainfall for real rainfall is worse than one that reports an
    outage.
    """
    mm = np.zeros(n_intervals, dtype=float)
    k = min(8, n_intervals)
    mm[-k:] = total_mm / k
    return RainfallSeries(
        mm_per_interval=mm,
        interval_hours=interval_hours,
        source="synthetic",
        includes_forecast=False,
        elevation_m=None,
        soil_wetness=estimate_soil_wetness(mm, interval_hours),
    )


def from_values(
    mm_per_interval: Sequence[float],
    *,
    interval_hours: float = INTERVAL_HOURS,
    includes_forecast: bool = False,
    source: str = "user_supplied",
) -> RainfallSeries:
    """Wrap a caller-supplied series, e.g. for hindcasting a known event."""
    mm = np.maximum(np.asarray(mm_per_interval, dtype=float), 0.0)
    return RainfallSeries(
        mm_per_interval=mm,
        interval_hours=interval_hours,
        source=source,
        includes_forecast=includes_forecast,
        soil_wetness=estimate_soil_wetness(mm, interval_hours),
    )


def cyclone_remal_aizawl() -> RainfallSeries:
    """The 28 May 2024 Cyclone Remal hyetograph over Aizawl.

    205 mm in 24 hours, which triggered eight near-simultaneous landslides.
    Held out of all training and calibration, this is the system's
    validation benchmark: run it and show what the system would have
    published, and when.
    """
    mm = np.zeros(N_INTERVALS, dtype=float)
    # Two days of moderate antecedent rain, then the cyclone.
    mm[-24:-8] = 4.0
    x = np.linspace(-2.0, 2.0, 8)
    burst = np.exp(-(x ** 2))
    mm[-8:] = burst / burst.sum() * 205.0
    return RainfallSeries(
        mm_per_interval=mm,
        interval_hours=INTERVAL_HOURS,
        source="hindcast:cyclone_remal_2024-05-28",
        includes_forecast=False,
        soil_wetness=estimate_soil_wetness(mm, INTERVAL_HOURS),
    )
