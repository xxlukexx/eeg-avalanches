"""Deterministic simulations validating the corrected estimators."""

from __future__ import annotations

import json

import numpy as np

from eeg_avalanches.estimators import calculate_kappa, fit_discrete_power_law
from eeg_avalanches.legacy import legacy_power_law_exponent


RANDOM_SEED = 20260729


def validate_power_law(rng: np.random.Generator) -> dict[str, float]:
    true_exponent = 1.5
    values = rng.zipf(true_exponent, 50_000)
    corrected = fit_discrete_power_law(values, min_observations=50, xmin=1).exponent
    legacy = legacy_power_law_exponent(values)
    if abs(corrected - true_exponent) >= 0.03:
        raise AssertionError(f"discrete MLE did not recover exponent: {corrected}")
    if legacy - true_exponent <= 0.1:
        raise AssertionError(f"legacy estimator did not exhibit expected bias: {legacy}")
    return {
        "true_exponent": true_exponent,
        "corrected_estimate": corrected,
        "legacy_estimate": legacy,
        "fraction_size_one": float(np.mean(values == 1)),
    }


def validate_kappa(rng: np.random.Generator) -> dict[str, float]:
    support = np.arange(1, 501)
    probability = support.astype(np.float64) ** -1.5
    probability /= probability.sum()
    values = rng.choice(support, size=100_000, p=probability)
    result = calculate_kappa(
        values,
        theory_exponent=1.5,
        min_observations=50,
        reference_max_size=500,
    )
    if abs(result.value - 1.0) >= 0.01:
        raise AssertionError(f"kappa did not recover its reference distribution: {result.value}")
    return {
        "expected_kappa": 1.0,
        "corrected_kappa": result.value,
        "maximum_cdf_error": float(
            np.max(np.abs(np.asarray(result.reference_cdf) - np.asarray(result.empirical_cdf)))
        ),
    }


def validate_branching(rng: np.random.Generator) -> dict[str, float]:
    pairwise_ratios: list[float] = []
    ancestor_total = 0.0
    descendant_total = 0.0
    duration_one = 0
    n_avalanches = 100_000
    for _ in range(n_avalanches):
        counts = [1]
        for _ in range(1_000):
            next_count = int(rng.poisson(counts[-1]))
            if next_count == 0:
                break
            counts.append(next_count)
        array = np.asarray(counts, dtype=np.float64)
        if array.size == 1:
            duration_one += 1
        else:
            pairwise_ratios.extend((array[1:] / array[:-1]).tolist())
        ancestor_total += float(array.sum())
        descendant_total += float(array[1:].sum())

    legacy = float(np.mean(pairwise_ratios))
    corrected = descendant_total / ancestor_total
    if abs(corrected - 1.0) >= 0.02:
        raise AssertionError(f"terminal-inclusive estimator missed critical value: {corrected}")
    if legacy <= 1.03:
        raise AssertionError(f"legacy branching estimator did not show upward bias: {legacy}")
    return {
        "true_branching_ratio": 1.0,
        "corrected_ratio_of_sums": corrected,
        "legacy_mean_nonterminal_ratios": legacy,
        "duration_one_fraction": duration_one / n_avalanches,
    }


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    results = {
        "random_seed": RANDOM_SEED,
        "power_law": validate_power_law(rng),
        "kappa": validate_kappa(rng),
        "branching": validate_branching(rng),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

