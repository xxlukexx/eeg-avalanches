import math

import numpy as np
import pytest

from eeg_avalanches import (
    AvalancheConfig,
    analyze_avalanches,
    detect_avalanches,
    kappa_against_theory,
    power_law_exponent,
    threshold_events,
)


def test_detects_known_avalanches() -> None:
    events = np.array(
        [
            [0, 1, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    sizes, durations, branching = detect_avalanches([events])

    np.testing.assert_array_equal(sizes, [3, 2])
    np.testing.assert_array_equal(durations, [2, 1])
    np.testing.assert_array_equal(branching, [2])


def test_segment_boundaries_split_runs() -> None:
    first = np.array([[0, 1], [0, 0]], dtype=bool)
    second = np.array([[1, 0], [1, 0]], dtype=bool)
    sizes, durations, _ = detect_avalanches([first, second])

    np.testing.assert_array_equal(sizes, [1, 2])
    np.testing.assert_array_equal(durations, [1, 1])


def test_thresholding_pools_zscore_but_preserves_segments() -> None:
    segments = [
        np.array([[0.0, 0.0, 8.0], [2.0, 2.0, 2.0]]),
        np.array([[0.0, -8.0, 0.0], [2.0, 2.0, 2.0]]),
    ]
    events = threshold_events(segments, threshold_z=1.2)

    assert len(events) == 2
    np.testing.assert_array_equal(events[0][0], [False, False, True])
    np.testing.assert_array_equal(events[1][0], [False, True, False])
    assert not events[0][1].any()
    assert not events[1][1].any()


def test_temporal_binning_uses_any_event_per_channel() -> None:
    data = np.array([[0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 10.0, 0.0]])
    events = threshold_events(data, threshold_z=1.0, bin_width_samples=2)

    assert events[0].shape == (2, 2)
    np.testing.assert_array_equal(events[0], [[True, False], [False, True]])


def test_summary_metadata_and_empty_fits() -> None:
    data = np.zeros((2, 100))
    result = analyze_avalanches(data, AvalancheConfig(sampling_rate=200.0))

    assert result.avalanche_count == 0
    assert result.n_channels == 2
    assert result.n_samples == 100
    assert result.n_segments == 1
    assert math.isnan(result.mean_size)
    assert math.isnan(result.size_exponent)


def test_power_law_requires_minimum_observations() -> None:
    assert math.isnan(power_law_exponent([1, 2, 3], min_observations=4))
    assert np.isfinite(power_law_exponent(np.arange(1, 21), min_observations=20))


def test_kappa_parameters_are_configurable() -> None:
    values = np.arange(1, 41)
    default = kappa_against_theory(values)
    changed = kappa_against_theory(
        values,
        theory_exponent=1.7,
        evaluation_points=6,
        xmin=2.0,
    )

    assert np.isfinite(default)
    assert np.isfinite(changed)
    assert default != changed


def test_rejects_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        analyze_avalanches(
            np.array([[0.0, np.nan]]),
            AvalancheConfig(sampling_rate=100.0),
        )
