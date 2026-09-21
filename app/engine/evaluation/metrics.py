"""
Evaluation that survives a domain-expert judge.

Accuracy on a balanced split is the clearest signal that a team has not
understood this problem. Landslides are rare events on a daily 1 km grid:
a model that predicts "no landslide" everywhere, forever, scores above 99%.
One public SIH26001 project reports ROC-AUC of 1.000 and 100% critical-class
recall, which indicates spatial or temporal leakage rather than skill.

This module provides the metrics that actually mean something, plus the
splitters that stop leakage in the first place.

The metric nobody else reports
------------------------------
`lead_time_stats` measures, for every correctly predicted event, how many
hours passed between the alert firing and the slope failing. That is THE
early-warning metric -- a system that detects every landslide five minutes
beforehand is worthless -- and it is essentially absent from the field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ContingencyTable:
    """Event counts at one threshold. Counts, not percentages."""

    hits: int              # predicted yes, observed yes
    misses: int            # predicted no,  observed yes
    false_alarms: int      # predicted yes, observed no
    correct_negatives: int

    @property
    def pod(self) -> float:
        """Probability of detection (recall, true-alarm rate)."""
        d = self.hits + self.misses
        return self.hits / d if d else float("nan")

    @property
    def far(self) -> float:
        """False alarm RATIO: of everything we warned about, what share was
        wrong. This is the number an emergency manager feels."""
        d = self.hits + self.false_alarms
        return self.false_alarms / d if d else float("nan")

    @property
    def pofd(self) -> float:
        """Probability of false detection (false-alarm RATE)."""
        d = self.false_alarms + self.correct_negatives
        return self.false_alarms / d if d else float("nan")

    @property
    def tss(self) -> float:
        """True Skill Statistic = POD - POFD. Robust to class imbalance."""
        return self.pod - self.pofd

    @property
    def csi(self) -> float:
        """Critical success index (threat score)."""
        d = self.hits + self.misses + self.false_alarms
        return self.hits / d if d else float("nan")

    @property
    def precision(self) -> float:
        d = self.hits + self.false_alarms
        return self.hits / d if d else float("nan")

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "false_alarms": self.false_alarms,
            "correct_negatives": self.correct_negatives,
            "pod": _r(self.pod),
            "far": _r(self.far),
            "pofd": _r(self.pofd),
            "tss": _r(self.tss),
            "csi": _r(self.csi),
            "precision": _r(self.precision),
        }


def _r(x: float, n: int = 4) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def contingency(
    probabilities: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> ContingencyTable:
    """Event counts at a threshold."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    alarm = p >= threshold
    return ContingencyTable(
        hits=int(np.sum(alarm & (y == 1))),
        misses=int(np.sum(~alarm & (y == 1))),
        false_alarms=int(np.sum(alarm & (y == 0))),
        correct_negatives=int(np.sum(~alarm & (y == 0))),
    )


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared error of the probabilities. Lower is better."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2))


def reliability_curve(
    probabilities: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> list[dict]:
    """Binned observed frequency vs predicted probability.

    A well-calibrated model sits on the diagonal. Plot this next to your
    ROC curve; it is the one that shows your probabilities mean what they
    say.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not m.any():
            continue
        out.append({
            "bin_lower": round(float(lo), 3),
            "bin_upper": round(float(hi), 3),
            "n": int(m.sum()),
            "mean_predicted": round(float(p[m].mean()), 4),
            "observed_frequency": round(float(y[m].mean()), 4),
        })
    return out


def pr_auc(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the precision-recall curve (average precision).

    Under 1:10,000 imbalance ROC-AUC flatters everything. Report this
    instead -- it is sensitive to exactly the failure mode that matters,
    namely drowning in false positives.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    if y.sum() == 0:
        return float("nan")

    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)

    # Average precision: sum of precision at each recall increase.
    d_recall = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * d_recall))


def lead_time_stats(
    alert_times_h: Sequence[float],
    failure_times_h: Sequence[float],
    *,
    detected: Sequence[bool] | None = None,
) -> dict:
    """Distribution of warning time actually delivered.

    Parameters
    ----------
    alert_times_h : when the alert fired, per event
    failure_times_h : when the slope actually failed, per event
    detected : mask of events the system caught at all. Events not detected
        contribute to the miss count, not to the lead-time distribution.

    Returns
    -------
    dict with median, mean, p10, p90, share of events with usable warning
    (>= 6 h), and the count of zero-or-negative lead times.
    """
    a = np.asarray(alert_times_h, dtype=float)
    f = np.asarray(failure_times_h, dtype=float)
    if a.size != f.size:
        raise ValueError("alert and failure arrays must be the same length")

    mask = np.ones(a.size, dtype=bool) if detected is None else np.asarray(
        detected, dtype=bool
    )
    lead = f[mask] - a[mask]

    if lead.size == 0:
        return {"n_detected": 0, "note": "no detected events"}

    return {
        "n_detected": int(lead.size),
        "median_h": round(float(np.median(lead)), 2),
        "mean_h": round(float(np.mean(lead)), 2),
        "p10_h": round(float(np.percentile(lead, 10)), 2),
        "p90_h": round(float(np.percentile(lead, 90)), 2),
        "share_with_6h_plus": round(float(np.mean(lead >= 6.0)), 3),
        "share_with_24h_plus": round(float(np.mean(lead >= 24.0)), 3),
        "n_zero_or_negative": int(np.sum(lead <= 0)),
    }


def full_report(
    probabilities: Sequence[float],
    labels: Sequence[int],
    threshold: float,
    *,
    alert_times_h: Sequence[float] | None = None,
    failure_times_h: Sequence[float] | None = None,
) -> dict:
    """Everything a judge should be shown, in one dict."""
    table = contingency(probabilities, labels, threshold)
    report = {
        "threshold": round(float(threshold), 4),
        "n_samples": int(len(probabilities)),
        "base_rate": round(float(np.mean(labels)), 6),
        "contingency": table.as_dict(),
        "brier_score": round(brier_score(probabilities, labels), 6),
        "pr_auc": _r(pr_auc(probabilities, labels)),
        "reliability": reliability_curve(probabilities, labels),
    }
    if alert_times_h is not None and failure_times_h is not None:
        report["lead_time"] = lead_time_stats(alert_times_h, failure_times_h)
    return report


# ---------------------------------------------------------------------------
# Splitters -- where leakage is actually prevented
# ---------------------------------------------------------------------------

def leave_one_group_out(groups: Sequence) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Spatial cross-validation: hold out one watershed basin at a time.

    Yields (train_idx, test_idx). Random splits leak badly in this problem,
    because neighbouring cells share terrain, rainfall and often the same
    landslide.
    """
    g = np.asarray(groups)
    for value in np.unique(g):
        test = np.where(g == value)[0]
        train = np.where(g != value)[0]
        yield train, test


def temporal_holdout(
    years: Sequence[int],
    train_through: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Temporal split: train on everything up to a year, test after it.

    Do this AS WELL AS the spatial split, never instead of it. Doing only
    one lets leakage through, and reporting both is rare enough to be a
    differentiator on its own.
    """
    y = np.asarray(years, dtype=int)
    return np.where(y <= train_through)[0], np.where(y > train_through)[0]


def ablation_table(results: dict[str, dict]) -> list[dict]:
    """Format an ablation as rows for the report.

    Pass a dict mapping configuration name -> `full_report` output, in the
    order the layers were added. This is how you prove the novel layers
    earn their place instead of asserting it.
    """
    rows = []
    prev_tss = None
    for name, rep in results.items():
        tss = rep["contingency"]["tss"]
        delta = None if prev_tss is None or tss is None else round(tss - prev_tss, 4)
        rows.append({
            "configuration": name,
            "pod": rep["contingency"]["pod"],
            "far": rep["contingency"]["far"],
            "tss": tss,
            "delta_tss": delta,
            "pr_auc": rep.get("pr_auc"),
            "median_lead_time_h": rep.get("lead_time", {}).get("median_h"),
        })
        prev_tss = tss if tss is not None else prev_tss
    return rows
