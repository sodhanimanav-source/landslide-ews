"""
Infinite-slope stability with transient pore pressure, root cohesion and
structural surcharge.

The factor of safety at depth Z and time t:

    FS = [ c' + c_r + ((gamma_s * Z + q) * cos^2(beta) - psi * gamma_w) * tan(phi') ]
         / [ (gamma_s * Z + q) * sin(beta) * cos(beta) ]

where c' is effective soil cohesion (kPa), c_r is root cohesion (kPa),
q is surcharge from structures (kPa), psi is pressure head (m of water),
gamma_w is the unit weight of water and beta is the slope angle. With
q = 0 and c_r = 0 this reduces to the standard TRIGRS formulation.

Why root cohesion is a first-class term here
--------------------------------------------
This is the physical hinge between the disturbance layer and the hazard
model. Cutting or clearing a slope does not merely correlate with failure --
it removes root reinforcement, which is a real, quantified strength term,
and that reinforcement takes years to return. Modelling it explicitly means
our anthropogenic layer feeds the physics rather than being bolted on as
another opaque ML feature, and it is why our disturbance signal decays with
time since the cut.

References
----------
Sidle, R.C. & Ochiai, H. (2006). Landslides: Processes, Prediction and
    Land Use. AGU Water Resources Monograph 18. (root strength decay and
    recovery, the "window of vulnerability")
Schmidt et al. (2001). The variability of root cohesion as an influence on
    shallow landslide susceptibility. Canadian Geotechnical Journal 38.
"""

from __future__ import annotations

import numpy as np

from .params import GAMMA_W

# Maximum root cohesion by land cover, kPa. Mature forest sits at the top of
# the plausible range for tropical broadleaf; bare ground contributes none.
ROOT_COHESION_MAX_KPA: dict[str, float] = {
    "dense_forest": 10.0,
    "open_forest": 6.0,
    "shrub": 3.0,
    "plantation": 4.0,
    "grass": 1.5,
    "agriculture": 1.0,
    "bare": 0.0,
    "built": 0.0,
}

# Time constant for root system re-establishment after clearance, months.
# Roughly 3 years to reach ~63% of mature reinforcement on humid tropical
# hillslopes; tune against your own inventory if you can.
ROOT_RECOVERY_TAU_MONTHS = 36.0

# Time constant for decay of residual roots left in the ground after
# clearance. Dead roots keep contributing for a while, then rot.
ROOT_DECAY_TAU_MONTHS = 12.0


def root_cohesion_kpa(
    landcover: str | np.ndarray,
    months_since_disturbance: np.ndarray | float | None = None,
    *,
    pre_disturbance_cover: str = "dense_forest",
) -> np.ndarray:
    """Root cohesion, accounting for time since the slope was cut or cleared.

    Undisturbed ground simply returns the land-cover maximum. For a
    disturbed slope the result is the sum of two processes: decaying
    residual roots from the removed vegetation, and slowly recovering roots
    from whatever is growing back. Their sum has a minimum some months after
    clearance -- the "window of vulnerability" -- which is precisely the
    period our disturbance layer is designed to flag.

    Parameters
    ----------
    landcover : str or array of str
        Current land cover class.
    months_since_disturbance : float, array, or None
        Months since the slope was cut or cleared. None means undisturbed.
    pre_disturbance_cover : str
        What was there before. Controls how much residual root strength
        there was to lose.

    Returns
    -------
    c_r : ndarray, kPa
    """
    current_max = _cover_lookup(landcover)

    if months_since_disturbance is None:
        return current_max

    t = np.asarray(months_since_disturbance, dtype=float)
    t = np.maximum(t, 0.0)

    prior_max = _cover_lookup(pre_disturbance_cover)

    residual = prior_max * np.exp(-t / ROOT_DECAY_TAU_MONTHS)
    recovering = current_max * (1.0 - np.exp(-t / ROOT_RECOVERY_TAU_MONTHS))

    # Never credit more reinforcement than the mature cover could provide.
    return np.minimum(residual + recovering, np.maximum(prior_max, current_max))


def _cover_lookup(cover: str | np.ndarray) -> np.ndarray:
    if isinstance(cover, str):
        return np.asarray(ROOT_COHESION_MAX_KPA.get(cover, 0.0), dtype=float)
    arr = np.asarray(cover)
    out = np.zeros(arr.shape, dtype=float)
    for key, val in ROOT_COHESION_MAX_KPA.items():
        out[arr == key] = val
    return out


def factor_of_safety(
    pressure_head_m: np.ndarray,
    depths_m: np.ndarray,
    *,
    cohesion_kpa: np.ndarray | float,
    friction_rad: np.ndarray | float,
    unit_weight_knm3: np.ndarray | float,
    slope_rad: float,
    root_cohesion_kpa: np.ndarray | float = 0.0,
    surcharge_kpa: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Factor of safety at every depth and time.

    Parameters
    ----------
    pressure_head_m : (..., n_t, n_z)
        Pressure head from the infiltration solver, metres of water.
    depths_m : (n_z,)
        Depths matching the last axis of `pressure_head_m`.
    cohesion_kpa, friction_rad, unit_weight_knm3 : scalar or (n_draw,)
        Soil parameters. If arrays, they align with the leading axis of
        `pressure_head_m`.
    slope_rad : float
        Slope angle in radians.
    root_cohesion_kpa : scalar or (n_draw,)
        Additional cohesion from root reinforcement.
    surcharge_kpa : scalar or (n_draw,)
        Vertical stress added by structures on the slope.

    Returns
    -------
    fs : ndarray, same shape as `pressure_head_m`
        Factor of safety. Values are clipped to a large finite ceiling so
        that near-zero-slope cells do not produce infinities downstream.
    """
    psi = np.asarray(pressure_head_m, dtype=float)
    z = np.asarray(depths_m, dtype=float)

    sin_b = np.sin(slope_rad)
    cos_b = np.cos(slope_rad)

    if sin_b < 1e-6:
        # Flat ground never fails by this mechanism.
        return np.full(psi.shape, 1e3, dtype=float)

    def _shape(p):
        p = np.asarray(p, dtype=float)
        if p.ndim == 0:
            return p
        # align a per-draw parameter with the leading axis
        return p.reshape(p.shape + (1,) * (psi.ndim - 1))

    c = _shape(cohesion_kpa)
    c_r = _shape(root_cohesion_kpa)
    tan_phi = np.tan(_shape(friction_rad))
    gamma_s = _shape(unit_weight_knm3)
    q = _shape(surcharge_kpa)

    # Total vertical stress at depth, kPa
    sigma_v = gamma_s * z + q

    numerator = (
        c + c_r
        + (sigma_v * cos_b * cos_b - psi * GAMMA_W) * tan_phi
    )
    denominator = sigma_v * sin_b * cos_b

    with np.errstate(divide="ignore", invalid="ignore"):
        fs = numerator / denominator

    fs = np.nan_to_num(fs, nan=1e3, posinf=1e3, neginf=0.0)
    return np.clip(fs, 0.0, 1e3)


def critical_factor_of_safety(fs: np.ndarray) -> np.ndarray:
    """Minimum FS over depth and time -- the value that decides failure.

    Reduces a (..., n_t, n_z) field to (...,) by taking the worst case over
    both the time axis and the depth axis.
    """
    fs = np.asarray(fs, dtype=float)
    if fs.ndim < 2:
        raise ValueError("expected at least (n_t, n_z)")
    return fs.reshape(fs.shape[:-2] + (-1,)).min(axis=-1)


def failure_time_index(fs: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """First time index at which FS drops below `threshold` at any depth.

    Returns -1 where the slope never fails within the simulated window.
    This is what feeds the lead-time metric: the gap between when an alert
    would have fired and when the slope actually goes.
    """
    fs = np.asarray(fs, dtype=float)
    min_over_depth = fs.min(axis=-1)          # (..., n_t)
    failed = min_over_depth < threshold
    any_fail = failed.any(axis=-1)
    first = np.argmax(failed, axis=-1)
    return np.where(any_fail, first, -1)
