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
    validate_batch_compatibility,
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
    detection = detect_avalanches([events])

    np.testing.assert_array_equal(detection.sizes, [3, 2])
    np.testing.assert_array_equal(detection.durations, [2, 1])
    np.testing.assert_array_equal(detection.ancestor_counts, [1, 2, 2])
    np.testing.assert_array_equal(detection.descendant_counts, [2, 0, 0])


def test_segment_boundaries_split_runs() -> None:
    first = np.array([[0, 1], [0, 0]], dtype=bool)
    second = np.array([[1, 0], [1, 0]], dtype=bool)
    retained = detect_avalanches([first, second], discard_boundary_avalanches=False)
    discarded = detect_avalanches([first, second])

    np.testing.assert_array_equal(retained.sizes, [1, 2])
    np.testing.assert_array_equal(retained.durations, [1, 1])
    assert discarded.sizes.size == 0
    assert discarded.boundary_avalanches_discarded == 2


def test_thresholding_pools_zscore_but_preserves_segments() -> None:
    segments = [
        np.array([[0.0, 0.0, 8.0], [1.0, -1.0, 1.0]]),
        np.array([[0.0, -8.0, 0.0], [-1.0, 1.0, -1.0]]),
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
    phase = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    data = np.stack([np.sin(phase), np.cos(phase)])
    result = analyze_avalanches(data, AvalancheConfig(sampling_rate=200.0))

    assert result.avalanche_count == 0
    assert result.n_channels == 2
    assert result.n_samples == 100
    assert result.n_segments == 1
    assert math.isnan(result.mean_size)
    assert math.isnan(result.size_exponent)


def test_power_law_requires_minimum_observations() -> None:
    assert math.isnan(power_law_exponent([1, 2, 3], min_observations=4))
    assert np.isfinite(power_law_exponent(np.arange(1, 51), min_observations=50))


def test_kappa_parameters_are_configurable() -> None:
    values = np.arange(1, 201)
    default = kappa_against_theory(values)
    changed = kappa_against_theory(
        values,
        theory_exponent=1.7,
        evaluation_points=6,
        xmin=2,
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


def test_rejects_flat_channels_with_names() -> None:
    with pytest.raises(ValueError, match="CZ"):
        analyze_avalanches(
            np.vstack([np.arange(100), np.ones(100)]),
            AvalancheConfig(sampling_rate=100.0),
            channel_names=["FZ", "CZ"],
        )


def test_bootstrap_intervals_are_seeded_and_recorded() -> None:
    data = np.random.default_rng(44).normal(size=(8, 4_000))
    config = AvalancheConfig(
        sampling_rate=200.0,
        min_events_for_distribution_fit=10,
        bootstrap_iterations=20,
        random_seed=123,
    )

    first = analyze_avalanches(data, config)
    second = analyze_avalanches(data, config)

    assert first.uncertainty == second.uncertainty
    assert first.uncertainty["intervals"]["mean_size"] is not None
    assert first.provenance["uncertainty"]["iterations"] == 20


def test_batch_validation_checks_channel_identity() -> None:
    data = np.random.default_rng(55).normal(size=(2, 1_000))
    config = AvalancheConfig(sampling_rate=200.0)
    first = analyze_avalanches(data, config, channel_names=["FZ", "CZ"])
    second = analyze_avalanches(data, config, channel_names=["FZ", "CZ"])
    incompatible = analyze_avalanches(data, config, channel_names=["FZ", "PZ"])

    validate_batch_compatibility([first, second])
    with pytest.raises(ValueError, match="channel identities"):
        validate_batch_compatibility([first, incompatible])
