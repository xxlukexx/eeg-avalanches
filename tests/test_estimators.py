import numpy as np

from eeg_avalanches.estimators import calculate_kappa, fit_discrete_power_law
from eeg_avalanches.legacy import (
    legacy_power_law_exponent,
)


def test_exact_discrete_mle_recovers_known_exponent() -> None:
    rng = np.random.default_rng(1234)
    values = rng.zipf(1.5, 20_000)

    corrected = fit_discrete_power_law(values, min_observations=50, xmin=1)
    legacy = legacy_power_law_exponent(values)

    assert abs(corrected.exponent - 1.5) < 0.04
    assert legacy - corrected.exponent > 0.1


def test_kappa_recovers_declared_reference_distribution() -> None:
    rng = np.random.default_rng(5678)
    support = np.arange(1, 201)
    probability = support.astype(float) ** -1.5
    probability /= probability.sum()
    values = rng.choice(support, size=30_000, p=probability)

    corrected = calculate_kappa(
        values,
        reference_max_size=200,
    )

    assert abs(corrected.value - 1.0) < 0.015
    assert len(corrected.evaluation_points) == 10


def test_corrected_kappa_has_conventional_direction() -> None:
    rng = np.random.default_rng(9012)
    subcritical_sizes = rng.geometric(0.3, 30_000)
    subcritical_sizes = subcritical_sizes[subcritical_sizes <= 200]

    result = calculate_kappa(
        subcritical_sizes,
        reference_max_size=200,
    )
    reversed_sign = 1 + np.mean(
        np.asarray(result.empirical_cdf) - np.asarray(result.reference_cdf)
    )

    assert result.value < 1
    assert reversed_sign > 1
