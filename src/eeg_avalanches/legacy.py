"""Legacy estimators retained only to reproduce results from versions <=0.3."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def legacy_power_law_exponent(
    values: ArrayLike,
    min_observations: int = 20,
    xmin: float = 1.0,
) -> float:
    """Continuous MLE formerly applied to discrete avalanche counts."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array >= xmin)]
    if array.size < min_observations:
        return float("nan")
    denominator = np.log(array / xmin).sum()
    if denominator <= 0:
        return float("nan")
    return float(1.0 + array.size / denominator)


def legacy_kappa_against_theory(
    values: ArrayLike,
    theory_exponent: float = 1.5,
    min_observations: int = 20,
    evaluation_points: int = 10,
    xmin: float = 1.0,
) -> float:
    """Point-mass CDF and reversed subtraction used in versions <=0.3."""
    array = np.asarray(values, dtype=np.float64)
    array = np.sort(array[np.isfinite(array) & (array >= xmin)])
    if array.size < min_observations:
        return float("nan")
    points = np.unique(
        np.round(
            np.logspace(np.log10(xmin), np.log10(array.max()), evaluation_points)
        ).astype(int)
    )
    empirical_cdf = np.asarray([np.mean(array <= point) for point in points])
    reference_mass = points.astype(np.float64) ** (-theory_exponent)
    reference_cdf = np.cumsum(reference_mass / reference_mass.sum())
    return float(1.0 + np.mean(empirical_cdf - reference_cdf))


def legacy_branching_ratio(event_segments: list[ArrayLike]) -> float:
    """Mean nonterminal pairwise ratio used in versions <=0.3."""
    ratios: list[float] = []
    for raw_events in event_segments:
        events = np.asarray(raw_events, dtype=bool)
        bin_counts = events.sum(axis=0)
        active = bin_counts > 0
        index = 0
        while index < active.size:
            if not active[index]:
                index += 1
                continue
            start = index
            while index < active.size and active[index]:
                index += 1
            counts = bin_counts[start:index].astype(np.float64)
            if counts.size > 1:
                ratios.extend((counts[1:] / counts[:-1]).tolist())
    return float(np.mean(ratios)) if ratios else float("nan")

