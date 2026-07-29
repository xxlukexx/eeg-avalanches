"""Statistical estimators for neuronal-avalanche summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar
from scipy.special import zeta


FloatArray = NDArray[np.float64]


def _integer_values(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size and not np.allclose(array, np.round(array), rtol=0, atol=1e-9):
        raise ValueError(f"{name} must contain integer-valued observations")
    return np.round(array).astype(np.int64)


@dataclass(frozen=True)
class PowerLawFit:
    """Exact discrete power-law fit diagnostics."""

    exponent: float
    xmin: int | None
    n_tail: int
    ks_distance: float
    log_likelihood: float
    candidate_xmins: int
    estimator: str = "exact_discrete_mle_hurwitz_zeta"
    xmin_selection: str = "minimum_ks_distance"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _fit_alpha_exact(values: NDArray[np.int64], xmin: int) -> tuple[float, float]:
    tail = values[values >= xmin]
    log_sum = float(np.log(tail).sum())
    n_tail = int(tail.size)

    def negative_log_likelihood(alpha: float) -> float:
        normalizer = float(zeta(alpha, xmin))
        if not np.isfinite(normalizer) or normalizer <= 0:
            return float("inf")
        return n_tail * np.log(normalizer) + alpha * log_sum

    fit = minimize_scalar(
        negative_log_likelihood,
        method="bounded",
        bounds=(1.000001, 20.0),
        options={"xatol": 1e-10},
    )
    if not fit.success or not np.isfinite(fit.fun):
        return float("nan"), float("nan")
    return float(fit.x), float(-fit.fun)


def _discrete_power_law_ks(
    values: NDArray[np.int64],
    *,
    exponent: float,
    xmin: int,
) -> float:
    tail = np.sort(values[values >= xmin])
    support = np.unique(tail)
    empirical_cdf = np.searchsorted(tail, support, side="right") / tail.size
    theoretical_cdf = 1.0 - zeta(exponent, support + 1) / zeta(exponent, xmin)
    return float(np.max(np.abs(empirical_cdf - theoretical_cdf)))


def fit_discrete_power_law(
    values: ArrayLike,
    *,
    min_observations: int = 50,
    xmin: int | None = None,
) -> PowerLawFit:
    """Fit an unbounded discrete power law and optionally estimate ``xmin``.

    The exponent maximizes the exact Hurwitz-zeta likelihood. When ``xmin`` is
    omitted, the selected value minimizes the Kolmogorov-Smirnov distance
    between the empirical and fitted tail CDFs.
    """
    if min_observations < 2:
        raise ValueError("min_observations must be at least 2")
    if xmin is not None and (not float(xmin).is_integer() or xmin < 1):
        raise ValueError("xmin must be an integer of at least 1")

    array = _integer_values(values, name="power-law values")
    array = array[array >= 1]
    if array.size < min_observations:
        return PowerLawFit(float("nan"), None, int(array.size), float("nan"), float("nan"), 0)

    if xmin is None:
        candidates = [
            int(candidate)
            for candidate in np.unique(array)
            if np.sum(array >= candidate) >= min_observations
            and np.unique(array[array >= candidate]).size >= 2
        ]
    else:
        candidates = [int(xmin)]

    best: PowerLawFit | None = None
    for candidate in candidates:
        tail = array[array >= candidate]
        if tail.size < min_observations or np.unique(tail).size < 2:
            continue
        exponent, log_likelihood = _fit_alpha_exact(array, candidate)
        if not np.isfinite(exponent):
            continue
        ks_distance = _discrete_power_law_ks(
            array,
            exponent=exponent,
            xmin=candidate,
        )
        fit = PowerLawFit(
            exponent=exponent,
            xmin=candidate,
            n_tail=int(tail.size),
            ks_distance=ks_distance,
            log_likelihood=log_likelihood,
            candidate_xmins=len(candidates),
            xmin_selection="fixed" if xmin is not None else "minimum_ks_distance",
        )
        if best is None or fit.ks_distance < best.ks_distance:
            best = fit

    if best is None:
        return PowerLawFit(float("nan"), None, 0, float("nan"), float("nan"), len(candidates))
    return best


def power_law_exponent(
    values: ArrayLike,
    min_observations: int = 50,
    xmin: int | None = None,
) -> float:
    """Return the exponent from an exact discrete power-law fit."""
    return fit_discrete_power_law(
        values,
        min_observations=min_observations,
        xmin=xmin,
    ).exponent


@dataclass(frozen=True)
class KappaResult:
    """Kappa value and the CDF values used to calculate it."""

    value: float
    n_observations: int
    xmin: int
    reference_max_size: int | None
    reference_max_source: str
    evaluation_points: tuple[float, ...]
    empirical_cdf: tuple[float, ...]
    reference_cdf: tuple[float, ...]
    formula: str = "1 + mean(reference_cdf - empirical_cdf)"
    reference_distribution: str = "truncated_discrete_power_law"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calculate_kappa(
    values: ArrayLike,
    *,
    theory_exponent: float = 1.5,
    min_observations: int = 50,
    evaluation_points: int = 10,
    xmin: int = 1,
    reference_max_size: int | None = None,
) -> KappaResult:
    """Calculate kappa against a full-support truncated discrete power law."""
    if not np.isfinite(theory_exponent) or theory_exponent <= 1:
        raise ValueError("theory_exponent must be greater than 1")
    if min_observations < 2:
        raise ValueError("min_observations must be at least 2")
    if evaluation_points < 2:
        raise ValueError("evaluation_points must be at least 2")
    if not float(xmin).is_integer() or xmin < 1:
        raise ValueError("xmin must be an integer of at least 1")
    if reference_max_size is not None and (
        not float(reference_max_size).is_integer() or reference_max_size < xmin
    ):
        raise ValueError("reference_max_size must be an integer of at least xmin")
    xmin = int(xmin)
    reference_max_size = (
        int(reference_max_size) if reference_max_size is not None else None
    )

    array = _integer_values(values, name="avalanche sizes")
    array = array[array >= xmin]
    reference_source = "fixed" if reference_max_size is not None else "observed_maximum"
    if array.size < min_observations:
        return KappaResult(
            float("nan"),
            int(array.size),
            xmin,
            reference_max_size,
            reference_source,
            (),
            (),
            (),
        )

    xmax = int(reference_max_size if reference_max_size is not None else array.max())
    support = np.arange(xmin, xmax + 1, dtype=np.int64)
    reference_mass = support.astype(np.float64) ** (-theory_exponent)
    reference_cdf_full = np.cumsum(reference_mass / reference_mass.sum())
    points = np.geomspace(xmin, xmax, evaluation_points)
    support_indices = np.searchsorted(support, points, side="right") - 1
    reference_cdf = np.where(
        support_indices >= 0,
        reference_cdf_full[np.maximum(support_indices, 0)],
        0.0,
    )
    empirical_cdf = np.asarray(
        [np.mean(array <= point) for point in points],
        dtype=np.float64,
    )
    value = float(1.0 + np.mean(reference_cdf - empirical_cdf))
    return KappaResult(
        value=value,
        n_observations=int(array.size),
        xmin=xmin,
        reference_max_size=xmax,
        reference_max_source=reference_source,
        evaluation_points=tuple(float(point) for point in points),
        empirical_cdf=tuple(float(value) for value in empirical_cdf),
        reference_cdf=tuple(float(value) for value in reference_cdf),
    )


def kappa_against_theory(
    values: ArrayLike,
    theory_exponent: float = 1.5,
    min_observations: int = 50,
    evaluation_points: int = 10,
    xmin: int = 1,
    reference_max_size: int | None = None,
) -> float:
    """Return kappa using the corrected discrete reference CDF."""
    return calculate_kappa(
        values,
        theory_exponent=theory_exponent,
        min_observations=min_observations,
        evaluation_points=evaluation_points,
        xmin=xmin,
        reference_max_size=reference_max_size,
    ).value
