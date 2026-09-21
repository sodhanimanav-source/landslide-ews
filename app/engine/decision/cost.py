"""
Cost-calibrated alert thresholds and conformal uncertainty.

The problem with picking a threshold from a ROC curve
-----------------------------------------------------
Maximising the ROC curve implicitly assumes a missed landslide costs the
same as a needless evacuation. It does not. Published work on cost-
sensitive rainfall thresholds for shallow landslides found missed-alarm
cost to be nearly SEVEN TIMES false-alarm cost, and therefore moved the
optimal threshold DOWN -- deliberately accepting more false alarms --
cutting normalised expected cost from 0.30 to 0.25.

So we minimise normalised expected cost instead:

    NEC = [ (1 - TAr) * p(+) * C(-|+)  +  FAr * p(-) * C(+|-) ]
          / [ p(+) * C(-|+)  +  p(-) * C(+|-) ]

where TAr is the true-alarm rate, FAr the false-alarm rate, p(+) the base
rate of landslide days, C(-|+) the cost of a missed alarm and C(+|-) the
cost of a false one.

Two consequences that matter for the product
--------------------------------------------
1. The threshold becomes a POLICY input a district officer can set, not a
   magic number buried in the code.
2. It can differ per asset. The same 1 km cell may hold a hospital, a
   national highway and a footpath. They do not deserve the same trigger.

Uncertainty
-----------
`conformal_interval` wraps a point probability in a split-conformal
interval with a distribution-free coverage guarantee. An alert that reads
"probability 0.62, 90% interval 0.41-0.80" is auditable. One that reads
"HIGH RISK" is not.

References
----------
Cost-sensitive rainfall thresholds for shallow landslides. Landslides
    (2021), doi:10.1007/s10346-021-01707-4.
Uncertainty quantification for probabilistic machine learning in earth
    observation using conformal prediction. Scientific Reports (2024).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Cost of a missed landslide relative to a false alarm. The published
# reference scenario put this near 7; local disaster managers should be
# invited to set it, and the number should appear in your report.
DEFAULT_COST_RATIO = 7.0

# Base rate of landslide-positive cell-days. Landslides are rare events on
# a daily 1 km grid and pretending otherwise is the single most common
# mistake in this problem space.
DEFAULT_BASE_RATE = 1e-4


@dataclass(frozen=True)
class AssetClass:
    """An exposed asset type and how badly we want to avoid missing it."""

    name: str
    cost_ratio: float           # C(missed) / C(false alarm)
    population_weight: float = 1.0

    def threshold(
        self,
        probabilities: Sequence[float],
        labels: Sequence[int],
        base_rate: float = DEFAULT_BASE_RATE,
    ) -> float:
        return optimal_threshold(
            probabilities, labels,
            cost_ratio=self.cost_ratio,
            base_rate=base_rate,
        )


# Sensible defaults. A hospital trips earlier than a trekking path; that is
# what "risk" means as opposed to "hazard", and most submissions conflate
# the two.
ASSET_CLASSES: dict[str, AssetClass] = {
    "hospital": AssetClass("hospital", cost_ratio=25.0, population_weight=3.0),
    "school": AssetClass("school", cost_ratio=20.0, population_weight=3.0),
    "settlement": AssetClass("settlement", cost_ratio=15.0, population_weight=2.5),
    "national_highway": AssetClass("national_highway", cost_ratio=10.0, population_weight=2.0),
    "state_road": AssetClass("state_road", cost_ratio=7.0, population_weight=1.5),
    "rural_road": AssetClass("rural_road", cost_ratio=4.0, population_weight=1.0),
    "agriculture": AssetClass("agriculture", cost_ratio=2.0, population_weight=0.5),
    "footpath": AssetClass("footpath", cost_ratio=1.5, population_weight=0.3),
    "default": AssetClass("default", cost_ratio=DEFAULT_COST_RATIO),
}


def normalised_expected_cost(
    true_alarm_rate: np.ndarray | float,
    false_alarm_rate: np.ndarray | float,
    *,
    cost_ratio: float = DEFAULT_COST_RATIO,
    base_rate: float = DEFAULT_BASE_RATE,
) -> np.ndarray:
    """Normalised expected cost, in [0, 1]. Lower is better.

    `cost_ratio` is C(missed) / C(false alarm); the absolute costs cancel.
    """
    tar = np.asarray(true_alarm_rate, dtype=float)
    far = np.asarray(false_alarm_rate, dtype=float)

    p_pos = float(base_rate)
    p_neg = 1.0 - p_pos

    miss_term = (1.0 - tar) * p_pos * cost_ratio
    false_term = far * p_neg * 1.0
    denom = p_pos * cost_ratio + p_neg * 1.0

    return (miss_term + false_term) / denom


def cost_curve(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    cost_ratio: float = DEFAULT_COST_RATIO,
    base_rate: float = DEFAULT_BASE_RATE,
    n_thresholds: int = 201,
) -> dict:
    """Sweep the threshold and return the cost curve.

    Returns
    -------
    dict with arrays: thresholds, true_alarm_rate, false_alarm_rate, nec
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)

    if p.size != y.size:
        raise ValueError("probabilities and labels must be the same length")
    if p.size == 0:
        raise ValueError("no data")

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)

    alarms = p[None, :] >= thresholds[:, None]     # (n_thr, n_samples)
    tar = (alarms & (y == 1)[None, :]).sum(axis=1) / n_pos
    far = (alarms & (y == 0)[None, :]).sum(axis=1) / n_neg

    nec = normalised_expected_cost(
        tar, far, cost_ratio=cost_ratio, base_rate=base_rate
    )

    return {
        "thresholds": thresholds,
        "true_alarm_rate": tar,
        "false_alarm_rate": far,
        "nec": nec,
    }


def optimal_threshold(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    cost_ratio: float = DEFAULT_COST_RATIO,
    base_rate: float = DEFAULT_BASE_RATE,
) -> float:
    """Threshold that minimises normalised expected cost."""
    curve = cost_curve(
        probabilities, labels, cost_ratio=cost_ratio, base_rate=base_rate
    )
    return float(curve["thresholds"][int(np.argmin(curve["nec"]))])


def compare_with_roc(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    cost_ratio: float = DEFAULT_COST_RATIO,
    base_rate: float = DEFAULT_BASE_RATE,
) -> dict:
    """Cost-optimal threshold vs the naive ROC-optimal one.

    This comparison belongs in your report. It quantifies, in expected
    cost, what choosing the threshold properly is worth -- and it shows
    that the cost-optimal threshold sits LOWER, accepting more false alarms
    to avoid the far more expensive misses.
    """
    curve = cost_curve(
        probabilities, labels, cost_ratio=cost_ratio, base_rate=base_rate
    )
    tar, far = curve["true_alarm_rate"], curve["false_alarm_rate"]

    youden = tar - far
    roc_idx = int(np.argmax(youden))
    cost_idx = int(np.argmin(curve["nec"]))

    return {
        "cost_optimal_threshold": float(curve["thresholds"][cost_idx]),
        "cost_optimal_nec": float(curve["nec"][cost_idx]),
        "cost_optimal_pod": float(tar[cost_idx]),
        "cost_optimal_far": float(far[cost_idx]),
        "roc_optimal_threshold": float(curve["thresholds"][roc_idx]),
        "roc_optimal_nec": float(curve["nec"][roc_idx]),
        "roc_optimal_pod": float(tar[roc_idx]),
        "roc_optimal_far": float(far[roc_idx]),
        "nec_reduction": float(curve["nec"][roc_idx] - curve["nec"][cost_idx]),
        "cost_ratio": cost_ratio,
        "base_rate": base_rate,
    }


# ---------------------------------------------------------------------------
# Conformal prediction
# ---------------------------------------------------------------------------

@dataclass
class ConformalCalibrator:
    """Split-conformal intervals for a probabilistic predictor.

    Fit on a held-out calibration set that the model never saw. The
    resulting interval has a distribution-free marginal coverage guarantee
    of at least 1 - alpha, which is a far stronger statement than a
    bootstrap band.

    Usage
    -----
    >>> cal = ConformalCalibrator(alpha=0.1).fit(cal_probs, cal_labels)
    >>> lo, hi = cal.interval(0.62)
    """

    alpha: float = 0.1
    quantile_: float | None = None
    n_calibration_: int = 0

    def fit(
        self,
        probabilities: Sequence[float],
        labels: Sequence[int],
    ) -> "ConformalCalibrator":
        p = np.asarray(probabilities, dtype=float)
        y = np.asarray(labels, dtype=float)
        if p.size != y.size:
            raise ValueError("probabilities and labels must be the same length")
        if p.size < 20:
            raise ValueError(
                "conformal calibration needs at least 20 held-out points; "
                f"got {p.size}"
            )

        # Absolute-residual non-conformity score.
        scores = np.abs(y - p)
        n = scores.size
        # Finite-sample corrected quantile level.
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.quantile_ = float(np.quantile(scores, level, method="higher"))
        self.n_calibration_ = int(n)
        return self

    def interval(self, probability: float | np.ndarray) -> tuple:
        """Lower and upper bound at the configured coverage level."""
        if self.quantile_ is None:
            raise RuntimeError("call fit() before interval()")
        p = np.asarray(probability, dtype=float)
        lo = np.clip(p - self.quantile_, 0.0, 1.0)
        hi = np.clip(p + self.quantile_, 0.0, 1.0)
        if p.ndim == 0:
            return float(lo), float(hi)
        return lo, hi

    @property
    def coverage(self) -> float:
        return 1.0 - self.alpha

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "coverage": self.coverage,
            "half_width": self.quantile_,
            "n_calibration": self.n_calibration_,
        }


def ensemble_interval(
    draw_outcomes: Sequence[float],
    alpha: float = 0.1,
) -> tuple[float, float]:
    """Interval straight from the Monte Carlo draws, when no calibration
    set exists yet.

    This is NOT a conformal guarantee -- it is the spread of the physics
    ensemble, which reflects parameter uncertainty but not model error. Say
    so when you report it. Use `ConformalCalibrator` the moment you have
    enough held-out real events to calibrate against.
    """
    d = np.asarray(draw_outcomes, dtype=float)
    lo = float(np.quantile(d, alpha / 2.0))
    hi = float(np.quantile(d, 1.0 - alpha / 2.0))
    return lo, hi


def bayes_threshold(cost_ratio: float = DEFAULT_COST_RATIO) -> float:
    """Cost-optimal decision threshold when no labelled data exists yet.

    For a two-class decision with cost C_fa for a false alarm and C_miss
    for a miss, expected cost is minimised by acting when

        p >= C_fa / (C_fa + C_miss) = 1 / (1 + cost_ratio)

    With the published cost ratio near 7 this puts the trigger at about
    0.125 -- far below the 0.5 that an untuned classifier uses by default,
    which is precisely the point. A hospital (ratio 25) trips at 0.038.

    Use this to bootstrap. Replace it with `optimal_threshold` fitted on a
    held-out set as soon as you have one, and say in your report which of
    the two produced each number.
    """
    r = max(float(cost_ratio), 1e-6)
    return 1.0 / (1.0 + r)
