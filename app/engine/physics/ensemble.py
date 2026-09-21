"""
Monte Carlo ensemble: turn physics into a training corpus.

This module is the answer to the single hardest problem in Indian landslide
modelling -- there are not enough recorded events to train on. The best
published rainfall-threshold study for Kerala worked from 64 events since
2000 and achieved an intensity-duration fit of R^2 = 0.068. No amount of
model tuning rescues a dataset that small.

So we do not learn from the inventory. We simulate.

For each slope unit (a combination of gradient, lithology, land cover,
disturbance state and structural loading) we sweep a designed set of
rainfall scenarios. For each scenario we draw geotechnical parameters from
the per-lithology distributions and solve infiltration and stability for
every draw. The fraction of draws that fail is a genuine probability of
failure -- not a classifier score that happens to lie between 0 and 1.

The real GSI/NRSC inventory is then used to CALIBRATE this surface (see
`model.calibrate`), never to train it from scratch.

Output columns are documented in `CORPUS_COLUMNS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from .infiltration import pressure_head_iverson
from .params import GAMMA_W, get_lithology, lithology_codes, sample_parameters
from .stability import (
    critical_factor_of_safety,
    factor_of_safety,
    failure_time_index,
    root_cohesion_kpa,
)

# Design-storm temporal patterns. The literature is consistent that
# cumulative rainfall AND its temporal evolution both matter, so we sweep
# shape as an explicit dimension rather than assuming uniform intensity.
TEMPORAL_PATTERNS = ("uniform", "front_loaded", "back_loaded", "centre_peaked")


@dataclass(frozen=True)
class SlopeUnit:
    """One terrain unit: what the ensemble is conditioned on."""

    slope_deg: float
    lithology: str
    landcover: str = "dense_forest"
    months_since_disturbance: float | None = None
    pre_disturbance_cover: str = "dense_forest"
    surcharge_kpa: float = 0.0
    antecedent_wetness: float = 0.1  # steady infiltration as fraction of Ksat

    def as_features(self) -> dict:
        lp = get_lithology(self.lithology)
        disturbed = self.months_since_disturbance is not None
        return {
            "slope_deg": self.slope_deg,
            "lithology": self.lithology,
            "lithology_strength_rank": lp.strength_rank,
            "landcover": self.landcover,
            "is_disturbed": int(disturbed),
            "months_since_disturbance": (
                float(self.months_since_disturbance) if disturbed else -1.0
            ),
            "surcharge_kpa": self.surcharge_kpa,
            "antecedent_wetness": self.antecedent_wetness,
        }


@dataclass(frozen=True)
class RainfallScenario:
    """A design storm, described both as a series and as scalar descriptors."""

    hyetograph_mm: np.ndarray = field(repr=False)  # per interval, mm
    interval_hours: float
    pattern: str
    label: str = ""
    storm_duration_hours: float | None = None

    @property
    def window_hours(self) -> float:
        """Length of the whole simulated window, including the dry lead-in."""
        return self.hyetograph_mm.size * self.interval_hours

    @property
    def duration_hours(self) -> float:
        """Duration of the storm itself -- the 'D' in an intensity-duration
        threshold. Falls back to the count of wet intervals when not set."""
        if self.storm_duration_hours is not None:
            return float(self.storm_duration_hours)
        wet = int((self.hyetograph_mm > 0).sum())
        return max(wet, 1) * self.interval_hours

    @property
    def total_mm(self) -> float:
        return float(self.hyetograph_mm.sum())

    def descriptors(self) -> dict:
        """Scalar rainfall features the surrogate model learns from."""
        mm = self.hyetograph_mm
        per_hour = mm / self.interval_hours
        n_per_day = int(round(24.0 / self.interval_hours))

        def tail_sum(days: int) -> float:
            k = n_per_day * days
            return float(mm[-k:].sum()) if k <= mm.size else float(mm.sum())

        # Temporal centroid, normalised to [0, 1]: 0.5 is symmetric, below
        # 0.5 is front-loaded, above is back-loaded. This one number carries
        # most of what "temporal evolution" means for slope response.
        total = mm.sum()
        if total > 0:
            idx = np.arange(mm.size) + 0.5
            centroid = float((mm * idx).sum() / total / mm.size)
        else:
            centroid = 0.5

        return {
            "rain_total_mm": self.total_mm,
            "rain_1d_mm": tail_sum(1),
            "rain_3d_mm": tail_sum(3),
            "rain_5d_mm": tail_sum(5),
            "rain_7d_mm": tail_sum(7),
            "rain_duration_h": self.duration_hours,
            "rain_peak_mmh": float(per_hour.max()) if mm.size else 0.0,
            "rain_mean_mmh": float(per_hour.mean()) if mm.size else 0.0,
            "rain_centroid": centroid,
            "rain_pattern": self.pattern,
        }


def make_scenario(
    total_mm: float,
    duration_hours: float,
    pattern: str = "uniform",
    *,
    interval_hours: float = 3.0,
    window_days: int = 7,
) -> RainfallScenario:
    """Build a design storm of given depth, duration and shape.

    The storm is placed at the END of a `window_days` window, with zero
    rainfall before it, so that the multi-day cumulative descriptors have
    the correct meaning relative to the moment of assessment.
    """
    n_storm = max(1, int(round(duration_hours / interval_hours)))
    n_window = max(n_storm, int(round(window_days * 24.0 / interval_hours)))

    x = (np.arange(n_storm) + 0.5) / n_storm
    if pattern == "uniform":
        w = np.ones(n_storm)
    elif pattern == "front_loaded":
        w = np.exp(-3.0 * x)
    elif pattern == "back_loaded":
        w = np.exp(3.0 * (x - 1.0))
    elif pattern == "centre_peaked":
        w = np.exp(-12.0 * (x - 0.5) ** 2)
    else:
        raise ValueError(f"unknown temporal pattern: {pattern!r}")

    w = w / w.sum()
    storm = w * total_mm

    hyeto = np.zeros(n_window, dtype=float)
    hyeto[-n_storm:] = storm

    return RainfallScenario(
        hyetograph_mm=hyeto,
        interval_hours=interval_hours,
        pattern=pattern,
        label=f"{total_mm:.0f}mm/{duration_hours:.0f}h/{pattern}",
        storm_duration_hours=n_storm * interval_hours,
    )


def simulate(
    unit: SlopeUnit,
    scenario: RainfallScenario,
    *,
    n_draws: int = 64,
    n_depths: int = 10,
    max_depth_m: float = 4.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run one (unit, scenario) pair through the full physics.

    Returns
    -------
    dict with keys:
        prob_failure         fraction of Monte Carlo draws with min FS < 1
        fs_median            median critical factor of safety
        fs_p10               10th percentile critical FS (the pessimistic tail)
        delta_pressure_kpa   median pore-pressure rise at the critical depth,
                             which is what the rate-and-state layer consumes
        sigma_eff_kpa        median effective normal stress at that depth
        failure_hours        median hours from start of window to failure,
                             NaN where no draw fails
    """
    if rng is None:
        rng = np.random.default_rng()

    params = sample_parameters(unit.lithology, n_draws, rng=rng)
    slope_rad = np.deg2rad(unit.slope_deg)

    depths = np.linspace(max_depth_m / n_depths, max_depth_m, n_depths)

    interval_s = scenario.interval_hours * 3600.0
    edges = np.arange(scenario.hyetograph_mm.size + 1, dtype=float) * interval_s
    intensity_ms = scenario.hyetograph_mm / 1000.0 / interval_s

    # Water table starts just below the soil column unless the unit is wet.
    wt_depth = params["soil_depth_m"] * (1.0 + 0.5 * (1.0 - unit.antecedent_wetness))

    psi = pressure_head_iverson(
        depths,
        edges,
        intensity_ms,
        ksat_ms=params["ksat_ms"],
        diffusivity_m2s=params["diffusivity_m2s"],
        slope_rad=slope_rad,
        water_table_depth_m=wt_depth,
        steady_ratio=unit.antecedent_wetness,
    )  # (n_draws, n_t, n_z)

    c_root = root_cohesion_kpa(
        unit.landcover,
        unit.months_since_disturbance,
        pre_disturbance_cover=unit.pre_disturbance_cover,
    )

    fs = factor_of_safety(
        psi,
        depths,
        cohesion_kpa=params["cohesion_kpa"],
        friction_rad=params["friction_rad"],
        unit_weight_knm3=params["unit_weight_knm3"],
        slope_rad=slope_rad,
        root_cohesion_kpa=float(np.asarray(c_root)),
        surcharge_kpa=unit.surcharge_kpa,
    )

    # The infinite-slope model only applies within the soil veneer: the
    # failure plane sits at or above the soil-bedrock contact. Without this
    # mask the ensemble "fails" at 4 m depth in soils that are 2 m thick,
    # and it fails at t=0 from the initial profile rather than from rain.
    below_contact = depths[None, None, :] > params["soil_depth_m"][:, None, None]
    fs = np.where(below_contact, 1e3, fs)

    fs_crit = critical_factor_of_safety(fs)          # (n_draws,)
    prob_failure = float((fs_crit < 1.0).mean())

    # Pore-pressure rise at the depth where FS is minimised -- this is the
    # delta_P that the rate-and-state layer needs.
    flat = fs.reshape(fs.shape[0], -1)
    crit_idx = flat.argmin(axis=1)
    t_idx, z_idx = np.unravel_index(crit_idx, fs.shape[1:])
    draw_idx = np.arange(fs.shape[0])

    psi_crit = psi[draw_idx, t_idx, z_idx]
    psi_initial = psi[draw_idx, 0, z_idx]
    delta_pressure = np.maximum(psi_crit - psi_initial, 0.0) * GAMMA_W  # kPa

    z_crit = depths[z_idx]
    sigma_eff = (
        (params["unit_weight_knm3"] * z_crit + unit.surcharge_kpa)
        * np.cos(slope_rad) ** 2
        - psi_crit * GAMMA_W
    )
    sigma_eff = np.maximum(sigma_eff, 1e-3)

    fail_idx = failure_time_index(fs)
    failed = fail_idx >= 0
    if failed.any():
        failure_hours = float(
            np.median(fail_idx[failed].astype(float) * scenario.interval_hours)
        )
    else:
        failure_hours = float("nan")

    return {
        "prob_failure": prob_failure,
        "fs_median": float(np.median(fs_crit)),
        "fs_p10": float(np.percentile(fs_crit, 10)),
        "delta_pressure_kpa": float(np.median(delta_pressure)),
        "sigma_eff_kpa": float(np.median(sigma_eff)),
        "critical_depth_m": float(np.median(z_crit)),
        "failure_hours": failure_hours,
    }


def default_scenario_grid(
    *,
    totals_mm: Sequence[float] = (25, 50, 80, 120, 175, 250, 350, 500),
    durations_h: Sequence[float] = (6, 12, 24, 48, 72, 120),
    patterns: Sequence[str] = TEMPORAL_PATTERNS,
    interval_hours: float = 3.0,
) -> Iterator[RainfallScenario]:
    """The rainfall design space the corpus is swept over.

    Totals span light rain to cyclone-scale events -- Cyclone Remal
    delivered 205 mm in 24 hours over Aizawl on 28 May 2024, which sits
    comfortably inside this grid.
    """
    for total in totals_mm:
        for dur in durations_h:
            # Reject physically absurd combinations (500 mm in 6 hours).
            if total / dur > 60.0:
                continue
            for pattern in patterns:
                yield make_scenario(
                    total, dur, pattern, interval_hours=interval_hours
                )


def default_unit_grid(
    *,
    slopes_deg: Sequence[float] = (15, 25, 30, 35, 40, 45, 55, 65),
    lithologies: Sequence[str] | None = None,
    disturbance_months: Sequence[float | None] = (None, 3, 12, 30, 60),
    wetness: Sequence[float] = (0.05, 0.3, 0.6),
    surcharges_kpa: Sequence[float] = (0.0, 20.0),
) -> Iterator[SlopeUnit]:
    """The terrain design space.

    `disturbance_months = None` is undisturbed forest; the numeric values
    walk through the window of vulnerability after a cut.
    """
    liths = list(lithologies) if lithologies else lithology_codes()
    for slope in slopes_deg:
        for lith in liths:
            for months in disturbance_months:
                cover = "dense_forest" if months is None else "bare"
                for w in wetness:
                    for q in surcharges_kpa:
                        yield SlopeUnit(
                            slope_deg=slope,
                            lithology=lith,
                            landcover=cover,
                            months_since_disturbance=months,
                            surcharge_kpa=q,
                            antecedent_wetness=w,
                        )


CORPUS_COLUMNS = [
    # terrain
    "slope_deg", "lithology", "lithology_strength_rank", "landcover",
    "is_disturbed", "months_since_disturbance", "surcharge_kpa",
    "antecedent_wetness",
    # rainfall
    "rain_total_mm", "rain_1d_mm", "rain_3d_mm", "rain_5d_mm", "rain_7d_mm",
    "rain_duration_h", "rain_peak_mmh", "rain_mean_mmh", "rain_centroid",
    "rain_pattern",
    # physics targets
    "prob_failure", "fs_median", "fs_p10", "delta_pressure_kpa",
    "sigma_eff_kpa", "critical_depth_m", "failure_hours",
]


def generate_corpus(
    units: Iterable[SlopeUnit] | None = None,
    scenarios: Iterable[RainfallScenario] | None = None,
    *,
    n_draws: int = 64,
    seed: int = 0,
    max_rows: int | None = None,
    progress_every: int = 2000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep the design space and return the training corpus.

    Parameters
    ----------
    units, scenarios : iterables, or None to use the default grids
    n_draws : Monte Carlo draws per (unit, scenario) pair
    max_rows : stop early -- useful for a smoke test
    """
    units = list(units) if units is not None else list(default_unit_grid())
    scenarios = (
        list(scenarios) if scenarios is not None
        else list(default_scenario_grid())
    )

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    total = len(units) * len(scenarios)
    if max_rows is not None:
        total = min(total, max_rows)

    if verbose:
        print(
            f"[ensemble] {len(units)} units x {len(scenarios)} scenarios "
            f"= {len(units) * len(scenarios):,} simulations "
            f"({n_draws} draws each)"
        )

    count = 0
    for unit in units:
        base = unit.as_features()
        for scen in scenarios:
            result = simulate(unit, scen, n_draws=n_draws, rng=rng)
            row = {**base, **scen.descriptors(), **result}
            rows.append(row)
            count += 1
            if verbose and progress_every and count % progress_every == 0:
                print(f"[ensemble]   {count:,}/{total:,}")
            if max_rows is not None and count >= max_rows:
                return pd.DataFrame(rows, columns=CORPUS_COLUMNS)

    if verbose:
        print(f"[ensemble] done: {len(rows):,} rows")
    return pd.DataFrame(rows, columns=CORPUS_COLUMNS)


def scenario_from_series(
    mm_per_interval: Sequence[float],
    interval_hours: float = 3.0,
    *,
    pattern: str = "observed",
    label: str = "observed",
) -> RainfallScenario:
    """Wrap a real observed or forecast rainfall series as a scenario.

    This is what the live API uses: the same physics that generated the
    training corpus is driven by the actual hyetograph from IMD, IMERG or a
    weather API, rather than by a design storm.
    """
    mm = np.asarray(mm_per_interval, dtype=float)
    if mm.ndim != 1 or mm.size == 0:
        raise ValueError("mm_per_interval must be a non-empty 1-D series")
    wet = int((mm > 0).sum())
    return RainfallScenario(
        hyetograph_mm=np.maximum(mm, 0.0),
        interval_hours=interval_hours,
        pattern=pattern,
        label=label,
        storm_duration_hours=max(wet, 1) * interval_hours,
    )
