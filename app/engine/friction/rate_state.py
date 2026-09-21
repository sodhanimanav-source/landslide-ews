"""
Rate-and-state friction: how much warning time is left.

This is the layer that distinguishes BHU-KAAL from every other landslide
dashboard. Conventional systems answer "is this slope dangerous?". This one
answers "how long do you have?", which is the question an evacuation
decision actually turns on.

The physics
-----------
Treat a slope as a block on a pre-existing surface whose frictional
strength follows rate-and-state theory. Rainfall raises pore pressure by
delta_P, which reduces effective normal stress and therefore shear
resistance. The dimensionless group that controls the response is

    chi = mu_0 * delta_P / (a * sigma_eff)

where mu_0 is the reference friction coefficient, a is the rate-and-state
direct-effect parameter, and sigma_eff is effective normal stress on the
failure plane. chi is the pore-pressure perturbation measured in units of
the slope's own frictional stiffness -- it is Dieterich's normalised stress
step applied to a hillslope.

Three regimes follow, and they were established for Mizoram specifically:

    chi >~ 4     SYNCHRONOUS. Failure coincides with peak pore pressure.
                 There is no observable precursor and no warning window.
                 Cyclone Remal delivered 205 mm in 24 h over Aizawl on
                 28 May 2024 and produced eight near-simultaneous failures.
    chi ~ 1-3    DELAYED. Hours to about ten days of accelerating creep
                 before runaway. This is the regime in which monitoring and
                 staged evacuation actually work.
    chi < 1      CREEP. No imminent runaway; watchlist only.

Time to failure follows Dieterich's nucleation result, in which the delay
shortens exponentially with the normalised stress step:

    t_failure = t_a * exp(-chi)

t_a is the characteristic nucleation time, modulated by the
velocity-weakening ratio a/b. It is the single calibratable constant in
this module and SHOULD be fitted against a regional inventory -- see
`calibrate_characteristic_time`. The default is set so that the regime
boundaries reproduce the published Mizoram behaviour.

What this means for system design
---------------------------------
For a chi >~ 4 slope, no observation-driven system can help, because the
slope gives nothing to observe before it fails. The only useful action is
pre-emptive, triggered by a rainfall FORECAST. Under high-emission climate
scenarios the share of such zero-warning failures in Mizoram is projected
to rise from roughly 56% to 72%, which makes forecast-driven action the
strategic centre of the design rather than a nice-to-have.

References
----------
Dieterich, J.H. (1994). A constitutive law for rate of earthquake
    production and its application to earthquake clustering. JGR 99(B2).
Frictional timescales and the impact of climate change-driven extreme
    weather on rainfall-triggered landslides in Mizoram, NE India (2026).
    arXiv:2606.23281.
Fukuzono, T. (1985). A new method for predicting the failure time of a
    slope. Proc. 4th Int. Conf. on Landslides, Tokyo.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

# Regime boundaries in chi, after the Mizoram study.
CHI_SYNCHRONOUS = 4.0
CHI_DELAYED = 1.0

# Characteristic nucleation time, hours. With the default a/b this places
# chi = 1 at roughly 3.5 days of lead time and chi = 3 at roughly 12 hours,
# matching the published "hours to ten days" range for the delayed regime.
T_A_REFERENCE_HOURS = 240.0
AB_RATIO_REFERENCE = 0.7

# Reference friction coefficient for the failure surface. 0.6 is the
# standard Byerlee-type value and is appropriate for the clay-rich
# materials inverted in the Mizoram study.
MU_0_DEFAULT = 0.6

# Rate-and-state direct effect. Laboratory values for clay-rich gouge sit
# in the 0.005-0.015 band; 0.01 is the usual working default.
A_DEFAULT = 0.01

# All 18 natural failures inverted in the Mizoram study were
# velocity-weakening, i.e. b > a, so a/b < 1.
AB_RATIO_DEFAULT = 0.7


class Regime(str, Enum):
    """Failure regime, which determines what a warning can possibly achieve."""

    SYNCHRONOUS = "synchronous"
    DELAYED = "delayed"
    CREEP = "creep"
    STABLE = "stable"

    @property
    def description(self) -> str:
        return {
            Regime.SYNCHRONOUS: (
                "No warning window. Failure coincides with peak rainfall. "
                "Act on forecast rainfall before the storm arrives."
            ),
            Regime.DELAYED: (
                "Hours to days of accelerating creep before failure. "
                "Staged response and monitored evacuation are viable."
            ),
            Regime.CREEP: (
                "Slow deformation without imminent runaway. "
                "Monitoring watchlist only."
            ),
            Regime.STABLE: (
                "No meaningful pore-pressure perturbation. No action."
            ),
        }[self]


@dataclass(frozen=True)
class WarningAssessment:
    """What the warning-time layer returns for one slope."""

    chi: float
    regime: Regime
    lead_time_hours: float
    lead_time_label: str
    actionable: bool
    description: str

    def as_dict(self) -> dict:
        return {
            "chi": round(self.chi, 3),
            "regime": self.regime.value,
            "lead_time_hours": (
                None if not np.isfinite(self.lead_time_hours)
                else round(self.lead_time_hours, 1)
            ),
            "lead_time_label": self.lead_time_label,
            "actionable_by_observation": self.actionable,
            "description": self.description,
        }


def chi(
    delta_pressure_kpa: np.ndarray | float,
    sigma_eff_kpa: np.ndarray | float,
    *,
    mu_0: float = MU_0_DEFAULT,
    a: float = A_DEFAULT,
) -> np.ndarray:
    """Normalised pore-pressure perturbation.

    chi = mu_0 * delta_P / (a * sigma_eff)

    Parameters
    ----------
    delta_pressure_kpa : rainfall-driven pore-pressure rise, kPa. This comes
        straight out of the physics layer -- no extra computation needed.
    sigma_eff_kpa : effective normal stress on the failure plane, kPa.
    mu_0, a : rate-and-state parameters.

    Returns
    -------
    chi : dimensionless, >= 0
    """
    dp = np.maximum(np.asarray(delta_pressure_kpa, dtype=float), 0.0)
    sig = np.maximum(np.asarray(sigma_eff_kpa, dtype=float), 1e-6)
    return mu_0 * dp / (a * sig)


def classify_regime(chi_value: np.ndarray | float) -> np.ndarray:
    """Map chi onto the failure regimes. Returns an array of Regime values."""
    c = np.asarray(chi_value, dtype=float)
    out = np.full(c.shape, Regime.STABLE, dtype=object)
    out[c >= 0.1] = Regime.CREEP
    out[c >= CHI_DELAYED] = Regime.DELAYED
    out[c >= CHI_SYNCHRONOUS] = Regime.SYNCHRONOUS
    return out


def lead_time_hours(
    chi_value: np.ndarray | float,
    *,
    ab_ratio: float = AB_RATIO_DEFAULT,
    t_a_hours: float = T_A_REFERENCE_HOURS,
) -> np.ndarray:
    """Hours of warning remaining, from Dieterich nucleation scaling.

        t_failure = t_a * (a/b) / (a/b)_ref * exp(-chi)

    Returns 0 in the synchronous regime (no usable window) and +inf where
    chi is negligible.

    Notes
    -----
    `ab_ratio < 1` means velocity-weakening, which is what the Mizoram
    inversion found for every natural failure. Stronger velocity weakening
    (smaller a/b) shortens the nucleation time.
    """
    c = np.atleast_1d(np.asarray(chi_value, dtype=float))
    t_a = t_a_hours * (ab_ratio / AB_RATIO_REFERENCE)

    out = t_a * np.exp(-c)
    out = np.where(c >= CHI_SYNCHRONOUS, 0.0, out)
    out = np.where(c < 0.1, np.inf, out)

    if np.isscalar(chi_value) or np.ndim(chi_value) == 0:
        return out[0]
    return out


def lead_time_label(hours: float) -> str:
    """Human-readable warning window, for an alert message."""
    if not np.isfinite(hours):
        return "no imminent failure"
    if hours <= 0.5:
        return "none - failure may be simultaneous with peak rainfall"
    if hours < 6:
        return f"under {int(np.ceil(hours))} hours"
    if hours < 48:
        return f"about {int(round(hours))} hours"
    return f"about {hours / 24.0:.1f} days"


def assess(
    delta_pressure_kpa: float,
    sigma_eff_kpa: float,
    *,
    mu_0: float = MU_0_DEFAULT,
    a: float = A_DEFAULT,
    ab_ratio: float = AB_RATIO_DEFAULT,
    t_a_hours: float = T_A_REFERENCE_HOURS,
) -> WarningAssessment:
    """Full warning-time assessment for a single slope."""
    c = float(chi(delta_pressure_kpa, sigma_eff_kpa, mu_0=mu_0, a=a))
    regime = classify_regime(c).item()
    hours = float(lead_time_hours(c, ab_ratio=ab_ratio, t_a_hours=t_a_hours))

    # A synchronous slope is NOT actionable by observation -- that is the
    # whole point. It is actionable only on a forecast.
    actionable = regime in (Regime.DELAYED, Regime.CREEP)

    return WarningAssessment(
        chi=c,
        regime=regime,
        lead_time_hours=hours,
        lead_time_label=lead_time_label(hours),
        actionable=actionable,
        description=regime.description,
    )


def calibrate_characteristic_time(
    chi_observed: Sequence[float],
    hours_to_failure: Sequence[float],
    *,
    ab_ratio: float = AB_RATIO_DEFAULT,
) -> float:
    """Fit t_a against observed events.

    Takes chi values computed for real recorded landslides and the measured
    delay between peak rainfall and failure, and returns the t_a that best
    reproduces them in a least-squares sense on log time.

    Use this the moment you have even a handful of NER events with known
    timing. Reporting a fitted t_a with its residuals is far stronger than
    quoting the default.
    """
    c = np.asarray(chi_observed, dtype=float)
    t = np.asarray(hours_to_failure, dtype=float)

    ok = np.isfinite(c) & np.isfinite(t) & (t > 0)
    if ok.sum() < 2:
        raise ValueError("need at least two finite, positive observations")

    # log t = log t_a' - chi   =>   log t_a' = mean(log t + chi)
    log_ta_scaled = float(np.mean(np.log(t[ok]) + c[ok]))
    t_a_scaled = float(np.exp(log_ta_scaled))
    return t_a_scaled / (ab_ratio / AB_RATIO_REFERENCE)


def inverse_velocity_failure_time(
    times_hours: Sequence[float],
    displacement_mm: Sequence[float],
    *,
    min_points: int = 4,
    smooth_window: int = 3,
) -> dict:
    """Fukuzono inverse-velocity extrapolation from a displacement series.

    Once a slope is actually moving, 1/v tends to decline linearly toward
    zero at the moment of failure. Fitting that line gives a countdown.
    Rate-and-state tells you which regime a slope is in; this tells you how
    long is left once it has started to go.

    Parameters
    ----------
    times_hours : monotonically increasing observation times
    displacement_mm : cumulative displacement at those times
    min_points : minimum samples before a prediction is attempted
    smooth_window : moving-average window applied to velocity, to stop
        sensor noise from producing a spurious countdown

    Returns
    -------
    dict with keys: predicted_failure_hours (absolute, on the same clock as
    `times_hours`), hours_remaining, r_squared, accelerating, n_points.
    `predicted_failure_hours` is None when the slope is not accelerating or
    the fit is too poor to act on.
    """
    t = np.asarray(times_hours, dtype=float)
    d = np.asarray(displacement_mm, dtype=float)

    if t.size != d.size:
        raise ValueError("times and displacement must be the same length")
    if t.size < min_points + 1:
        return {
            "predicted_failure_hours": None,
            "hours_remaining": None,
            "r_squared": None,
            "accelerating": False,
            "n_points": int(t.size),
            "note": "not enough observations",
        }

    dt = np.diff(t)
    v = np.diff(d) / np.where(dt > 0, dt, np.nan)      # mm/h
    t_mid = 0.5 * (t[1:] + t[:-1])

    if smooth_window > 1 and v.size >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        v = np.convolve(v, kernel, mode="valid")
        t_mid = t_mid[smooth_window - 1:]

    ok = np.isfinite(v) & (v > 1e-9)
    v, t_mid = v[ok], t_mid[ok]
    if v.size < min_points:
        return {
            "predicted_failure_hours": None,
            "hours_remaining": None,
            "r_squared": None,
            "accelerating": False,
            "n_points": int(v.size),
            "note": "slope is not moving measurably",
        }

    inv_v = 1.0 / v
    slope, intercept = np.polyfit(t_mid, inv_v, 1)

    pred = slope * t_mid + intercept
    ss_res = float(np.sum((inv_v - pred) ** 2))
    ss_tot = float(np.sum((inv_v - inv_v.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    accelerating = slope < 0
    if not accelerating or r2 < 0.5:
        return {
            "predicted_failure_hours": None,
            "hours_remaining": None,
            "r_squared": round(r2, 3),
            "accelerating": bool(accelerating),
            "n_points": int(v.size),
            "note": (
                "decelerating" if not accelerating
                else "fit too poor to act on (R2 < 0.5)"
            ),
        }

    t_fail = float(-intercept / slope)
    remaining = t_fail - float(t[-1])

    return {
        "predicted_failure_hours": round(t_fail, 2),
        "hours_remaining": round(remaining, 2),
        "r_squared": round(r2, 3),
        "accelerating": True,
        "n_points": int(v.size),
        "note": "Fukuzono inverse-velocity extrapolation",
    }
