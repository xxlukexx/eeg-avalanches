"""Core neuronal-avalanche calculations.

Arrays use channels-by-samples orientation. Multiple discontinuous epochs should
be supplied separately; avalanche runs are never allowed to cross boundaries.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .estimators import calculate_kappa, fit_discrete_power_law


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
PROVENANCE_SCHEMA_VERSION = "eeg-avalanches-provenance-2.0"
AVALANCHE_ALGORITHM_VERSION = "sensor-avalanche-v2"
KAPPA_ALGORITHM_VERSION = "kappa-discrete-full-cdf-v2"
POWER_LAW_ALGORITHM_VERSION = "discrete-mle-ks-xmin-v1"
BRANCHING_ALGORITHM_VERSION = "terminal-ratio-of-sums-v1"


@dataclass(frozen=True)
class AvalancheConfig:
    """Analysis parameters."""

    sampling_rate: float
    threshold_z: float = 2.5
    bin_width_samples: int = 1
    theory_exponent: float = 1.5
    min_events_for_distribution_fit: int = 50
    kappa_evaluation_points: int = 10
    kappa_xmin: int = 1
    kappa_reference_max_size: int | None = None
    size_power_law_xmin: int | None = None
    duration_power_law_xmin: int | None = None
    discard_boundary_avalanches: bool = True
    bootstrap_iterations: int = 0
    bootstrap_confidence_level: float = 0.95
    random_seed: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.sampling_rate) or self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive and finite")
        if not np.isfinite(self.threshold_z) or self.threshold_z <= 0:
            raise ValueError("threshold_z must be positive and finite")
        if self.bin_width_samples < 1:
            raise ValueError("bin_width_samples must be at least 1")
        if not np.isfinite(self.theory_exponent) or self.theory_exponent <= 1:
            raise ValueError("theory_exponent must be greater than 1")
        if self.min_events_for_distribution_fit < 2:
            raise ValueError("min_events_for_distribution_fit must be at least 2")
        if self.kappa_evaluation_points < 2:
            raise ValueError("kappa_evaluation_points must be at least 2")
        if not float(self.kappa_xmin).is_integer() or self.kappa_xmin < 1:
            raise ValueError("kappa_xmin must be an integer of at least 1")
        if self.kappa_reference_max_size is not None and (
            not float(self.kappa_reference_max_size).is_integer()
            or self.kappa_reference_max_size < self.kappa_xmin
        ):
            raise ValueError(
                "kappa_reference_max_size must be an integer of at least kappa_xmin"
            )
        if self.size_power_law_xmin is not None and (
            not float(self.size_power_law_xmin).is_integer()
            or self.size_power_law_xmin < 1
        ):
            raise ValueError("size_power_law_xmin must be an integer of at least 1")
        if self.duration_power_law_xmin is not None and (
            not float(self.duration_power_law_xmin).is_integer()
            or self.duration_power_law_xmin < 1
        ):
            raise ValueError("duration_power_law_xmin must be an integer of at least 1")
        if self.bootstrap_iterations < 0:
            raise ValueError("bootstrap_iterations cannot be negative")
        if not 0 < self.bootstrap_confidence_level < 1:
            raise ValueError("bootstrap_confidence_level must be between 0 and 1")

    def to_dict(self) -> dict[str, object]:
        """Return all configurable analysis parameters."""
        return asdict(self)


@dataclass(frozen=True)
class AvalancheResult:
    """Avalanche distributions, summary metrics, and analysis parameters."""

    avalanche_count: int
    mean_size: float
    mean_duration_bins: float
    mean_duration_seconds: float
    size_exponent: float
    duration_exponent: float
    size_power_law_fit: dict[str, object]
    duration_power_law_fit: dict[str, object]
    kappa: float
    kappa_diagnostics: dict[str, object]
    branching_ratio: float
    sizes: tuple[float, ...]
    durations_bins: tuple[int, ...]
    threshold_z: float
    bin_width_samples: int
    bin_width_seconds: float
    theory_exponent: float
    n_channels: int
    n_samples: int
    n_segments: int
    boundary_avalanches_discarded: int
    channel_names: tuple[str, ...] | None
    uncertainty: dict[str, object]
    provenance: dict[str, object]

    def to_dict(self, include_distributions: bool = True) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""
        result = asdict(self)
        if not include_distributions:
            result.pop("sizes")
            result.pop("durations_bins")
        return result

    def write_json(
        self,
        path: str | Path,
        *,
        include_distributions: bool = True,
    ) -> None:
        """Write standards-compliant JSON, representing non-finite values as null."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            self.to_json(include_distributions=include_distributions),
            encoding="utf-8",
        )

    def to_json(self, *, include_distributions: bool = True) -> str:
        """Return standards-compliant JSON with complete analysis provenance."""
        payload = _json_ready(self.to_dict(include_distributions=include_distributions))
        return json.dumps(payload, indent=2, allow_nan=False) + "\n"


def _package_version() -> str:
    try:
        return version("eeg-avalanches")
    except PackageNotFoundError:
        return "unknown"


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def build_provenance(
    config: AvalancheConfig,
    *,
    n_channels: int | None = None,
    n_samples: int | None = None,
    n_segments: int | None = None,
    channel_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build a complete, JSON-serializable record of kappa analysis choices."""
    bin_width_seconds = config.bin_width_samples / config.sampling_rate
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "software": {
            "package": "eeg-avalanches",
            "package_version": _package_version(),
            "algorithm": "sensor-space absolute-threshold neuronal avalanches",
            "algorithm_version": AVALANCHE_ALGORITHM_VERSION,
        },
        "input": {
            "sampling_rate_hz": config.sampling_rate,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "n_segments": n_segments,
            "array_orientation": "channels_by_samples",
            "segments_treated_as_discontinuous": True,
            "channel_names": list(channel_names) if channel_names is not None else None,
            "channel_identity_validation": (
                "explicit_names_checked" if channel_names is not None else "count_only"
            ),
        },
        "event_detection": {
            "standardization_scope": "per_channel_pooled_across_supplied_segments",
            "standard_deviation_ddof": 0,
            "threshold_type": "two_sided_absolute_z_score",
            "threshold_operator": ">",
            "threshold_z": config.threshold_z,
            "zero_standard_deviation_channels": "raise_error",
        },
        "temporal_binning": {
            "bin_width_samples": config.bin_width_samples,
            "bin_width_seconds": bin_width_seconds,
            "channel_bin_rule": "active_if_any_threshold_crossing_sample",
            "incomplete_final_bin": "discard",
        },
        "avalanche_definition": {
            "active_bin_rule": "at_least_one_active_channel",
            "boundary_rule": "empty_bin_or_segment_boundary",
            "size": "number_of_active_channel_bin_pairs",
            "duration": "number_of_consecutive_active_bins",
            "boundary_touching_avalanches": (
                "discard" if config.discard_boundary_avalanches else "retain_as_censored"
            ),
        },
        "power_law_fit": {
            "algorithm_version": POWER_LAW_ALGORITHM_VERSION,
            "distribution": "unbounded_discrete_power_law",
            "estimator": "exact_mle_hurwitz_zeta",
            "xmin_selection": "minimum_ks_distance_when_not_fixed",
            "size_xmin": config.size_power_law_xmin,
            "duration_xmin": config.duration_power_law_xmin,
            "minimum_tail_observations": config.min_events_for_distribution_fit,
        },
        "kappa": {
            "algorithm_version": KAPPA_ALGORITHM_VERSION,
            "theory_exponent": config.theory_exponent,
            "minimum_avalanches": config.min_events_for_distribution_fit,
            "xmin": config.kappa_xmin,
            "reference_max_size": config.kappa_reference_max_size,
            "reference_max_source": (
                "fixed" if config.kappa_reference_max_size is not None else "observed_maximum"
            ),
            "requested_evaluation_points": config.kappa_evaluation_points,
            "evaluation_points": "log_spaced_xmin_to_reference_max_without_rounding",
            "empirical_cdf": "proportion_of_avalanche_sizes_less_than_or_equal_to_each_point",
            "reference_cdf": "full_integer_support_truncated_discrete_power_law",
            "formula": "1 + mean(reference_cdf - empirical_cdf)",
            "insufficient_data_value": None,
        },
        "branching_ratio": {
            "algorithm_version": BRANCHING_ALGORITHM_VERSION,
            "formula": "sum(descendant_counts_including_terminal_zero) / sum(ancestor_counts)",
            "duration_one_avalanches": "included",
            "terminal_transition": "included_as_zero_descendants",
        },
        "uncertainty": {
            "method": "nonparametric_avalanche_bootstrap",
            "iterations": config.bootstrap_iterations,
            "confidence_level": config.bootstrap_confidence_level,
            "random_seed": config.random_seed,
            "power_law_xmin_during_bootstrap": "fixed_to_full_sample_estimate",
            "kappa_reference_max_during_bootstrap": "fixed_to_full_sample_value",
        },
    }


def _as_segments(data: ArrayLike | Sequence[ArrayLike]) -> list[FloatArray]:
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            raw_segments = [data]
        elif data.ndim == 3:
            raw_segments = [data[index] for index in range(data.shape[0])]
        else:
            raise ValueError("data must be 2D (channels, samples) or 3D (segments, channels, samples)")
    else:
        raw_segments = list(data)

    if not raw_segments:
        raise ValueError("at least one segment is required")

    segments: list[FloatArray] = []
    n_channels: int | None = None
    for index, segment in enumerate(raw_segments):
        array = np.asarray(segment, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"segment {index} is not 2D (channels, samples)")
        if array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError(f"segment {index} is empty")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"segment {index} contains NaN or infinite values")
        if n_channels is None:
            n_channels = array.shape[0]
        elif array.shape[0] != n_channels:
            raise ValueError("all segments must have the same number of channels")
        segments.append(array)
    return segments


def _validated_channel_names(
    channel_names: Sequence[str] | None,
    n_channels: int,
) -> tuple[str, ...] | None:
    if channel_names is None:
        return None
    names = tuple(str(name) for name in channel_names)
    if len(names) != n_channels:
        raise ValueError(
            f"channel_names has {len(names)} entries but data have {n_channels} channels"
        )
    if len(set(names)) != len(names):
        raise ValueError("channel_names must be unique")
    return names


def _bin_events(events: BoolArray, bin_width_samples: int) -> BoolArray:
    if bin_width_samples == 1:
        return events
    n_complete = events.shape[1] // bin_width_samples
    if n_complete == 0:
        return np.empty((events.shape[0], 0), dtype=bool)
    trimmed = events[:, : n_complete * bin_width_samples]
    return trimmed.reshape(events.shape[0], n_complete, bin_width_samples).any(axis=2)


def threshold_events(
    data: ArrayLike | Sequence[ArrayLike],
    threshold_z: float = 2.5,
    bin_width_samples: int = 1,
    channel_names: Sequence[str] | None = None,
) -> list[BoolArray]:
    """Z-score each channel across all segments and return thresholded event bins.

    An event is a positive or negative excursion satisfying ``abs(z) >
    threshold_z``. Pooling channel mean and standard deviation across segments
    matches a condition-level analysis while preserving segment boundaries.
    """
    if not np.isfinite(threshold_z) or threshold_z <= 0:
        raise ValueError("threshold_z must be positive and finite")
    if bin_width_samples < 1:
        raise ValueError("bin_width_samples must be at least 1")

    segments = _as_segments(data)
    names = _validated_channel_names(channel_names, segments[0].shape[0])
    pooled = np.concatenate(segments, axis=1)
    means = pooled.mean(axis=1, keepdims=True)
    standard_deviations = pooled.std(axis=1, keepdims=True)
    flat_indices = np.flatnonzero(standard_deviations[:, 0] == 0)
    if flat_indices.size:
        labels = (
            [names[index] for index in flat_indices]
            if names is not None
            else [int(index) for index in flat_indices]
        )
        raise ValueError(f"zero-standard-deviation channels detected: {labels}")

    return [
        _bin_events(np.abs((segment - means) / standard_deviations) > threshold_z, bin_width_samples)
        for segment in segments
    ]


@dataclass(frozen=True)
class AvalancheDetection:
    """Detected avalanche distributions and branching transitions."""

    sizes: FloatArray
    durations: FloatArray
    ancestor_counts: FloatArray
    descendant_counts: FloatArray
    avalanche_ancestor_totals: FloatArray
    avalanche_descendant_totals: FloatArray
    boundary_avalanches_discarded: int


def detect_avalanches(
    event_segments: Iterable[ArrayLike],
    *,
    discard_boundary_avalanches: bool = True,
) -> AvalancheDetection:
    """Extract avalanches and terminal-inclusive branching transitions.

    Each input is a channels-by-time-bin Boolean array. An avalanche is a run
    of non-empty bins bounded by empty bins or by a segment boundary. Size is
    the number of active channel-bin pairs; duration is the number of bins.
    """
    sizes: list[float] = []
    durations: list[float] = []
    ancestor_counts: list[float] = []
    descendant_counts: list[float] = []
    avalanche_ancestor_totals: list[float] = []
    avalanche_descendant_totals: list[float] = []
    boundary_discarded = 0
    n_channels: int | None = None

    for segment_index, raw_events in enumerate(event_segments):
        events = np.asarray(raw_events, dtype=bool)
        if events.ndim != 2:
            raise ValueError(f"event segment {segment_index} is not 2D")
        if n_channels is None:
            n_channels = events.shape[0]
        elif events.shape[0] != n_channels:
            raise ValueError("all event segments must have the same number of channels")

        bin_counts = events.sum(axis=0)
        active_bins = bin_counts > 0
        index = 0
        while index < active_bins.size:
            if not active_bins[index]:
                index += 1
                continue
            start = index
            while index < active_bins.size and active_bins[index]:
                index += 1
            touches_boundary = start == 0 or index == active_bins.size
            if touches_boundary and discard_boundary_avalanches:
                boundary_discarded += 1
                continue
            counts = bin_counts[start:index].astype(np.float64)
            sizes.append(float(counts.sum()))
            durations.append(float(counts.size))
            ancestor_counts.extend(counts.tolist())
            descendants = np.concatenate((counts[1:], [0.0]))
            descendant_counts.extend(descendants.tolist())
            avalanche_ancestor_totals.append(float(counts.sum()))
            avalanche_descendant_totals.append(float(descendants.sum()))

    return AvalancheDetection(
        sizes=np.asarray(sizes, dtype=np.float64),
        durations=np.asarray(durations, dtype=np.float64),
        ancestor_counts=np.asarray(ancestor_counts, dtype=np.float64),
        descendant_counts=np.asarray(descendant_counts, dtype=np.float64),
        avalanche_ancestor_totals=np.asarray(avalanche_ancestor_totals, dtype=np.float64),
        avalanche_descendant_totals=np.asarray(avalanche_descendant_totals, dtype=np.float64),
        boundary_avalanches_discarded=boundary_discarded,
    )


def _percentile_interval(
    values: Sequence[float],
    confidence_level: float,
) -> list[float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return None
    tail = (1.0 - confidence_level) / 2.0
    return [
        float(np.quantile(finite, tail)),
        float(np.quantile(finite, 1.0 - tail)),
    ]


def _bootstrap_uncertainty(
    detection: AvalancheDetection,
    config: AvalancheConfig,
    *,
    size_xmin: int | None,
    duration_xmin: int | None,
    kappa_reference_max_size: int | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "method": "nonparametric_avalanche_bootstrap",
        "iterations": config.bootstrap_iterations,
        "confidence_level": config.bootstrap_confidence_level,
        "random_seed": config.random_seed,
        "power_law_xmin": "fixed_to_full_sample_estimate",
        "kappa_reference_max_size": "fixed_to_full_sample_value",
        "intervals": None,
    }
    if config.bootstrap_iterations == 0 or detection.sizes.size == 0:
        return metadata

    rng = np.random.default_rng(config.random_seed)
    metrics: dict[str, list[float]] = {
        "mean_size": [],
        "mean_duration_bins": [],
        "size_exponent": [],
        "duration_exponent": [],
        "kappa": [],
        "branching_ratio": [],
    }
    n_avalanches = detection.sizes.size
    for _ in range(config.bootstrap_iterations):
        indices = rng.integers(0, n_avalanches, n_avalanches)
        sizes = detection.sizes[indices]
        durations = detection.durations[indices]
        metrics["mean_size"].append(float(sizes.mean()))
        metrics["mean_duration_bins"].append(float(durations.mean()))
        metrics["size_exponent"].append(
            fit_discrete_power_law(
                sizes,
                min_observations=config.min_events_for_distribution_fit,
                xmin=size_xmin,
            ).exponent
            if size_xmin is not None
            else float("nan")
        )
        metrics["duration_exponent"].append(
            fit_discrete_power_law(
                durations,
                min_observations=config.min_events_for_distribution_fit,
                xmin=duration_xmin,
            ).exponent
            if duration_xmin is not None
            else float("nan")
        )
        metrics["kappa"].append(
            calculate_kappa(
                sizes,
                theory_exponent=config.theory_exponent,
                min_observations=config.min_events_for_distribution_fit,
                evaluation_points=config.kappa_evaluation_points,
                xmin=config.kappa_xmin,
                reference_max_size=kappa_reference_max_size,
            ).value
        )
        ancestors = detection.avalanche_ancestor_totals[indices].sum()
        descendants = detection.avalanche_descendant_totals[indices].sum()
        metrics["branching_ratio"].append(
            float(descendants / ancestors) if ancestors > 0 else float("nan")
        )

    metadata["intervals"] = {
        name: _percentile_interval(values, config.bootstrap_confidence_level)
        for name, values in metrics.items()
    }
    return metadata


def analyze_avalanches(
    data: ArrayLike | Sequence[ArrayLike],
    config: AvalancheConfig,
    *,
    channel_names: Sequence[str] | None = None,
) -> AvalancheResult:
    """Compute neuronal-avalanche metrics from cleaned EEG data."""
    segments = _as_segments(data)
    names = _validated_channel_names(channel_names, segments[0].shape[0])
    events = threshold_events(
        segments,
        threshold_z=config.threshold_z,
        bin_width_samples=config.bin_width_samples,
        channel_names=names,
    )
    detection = detect_avalanches(
        events,
        discard_boundary_avalanches=config.discard_boundary_avalanches,
    )
    sizes = detection.sizes
    durations = detection.durations
    bin_width_seconds = config.bin_width_samples / config.sampling_rate
    size_fit = fit_discrete_power_law(
        sizes,
        min_observations=config.min_events_for_distribution_fit,
        xmin=config.size_power_law_xmin,
    )
    duration_fit = fit_discrete_power_law(
        durations,
        min_observations=config.min_events_for_distribution_fit,
        xmin=config.duration_power_law_xmin,
    )
    kappa_result = calculate_kappa(
        sizes,
        theory_exponent=config.theory_exponent,
        min_observations=config.min_events_for_distribution_fit,
        evaluation_points=config.kappa_evaluation_points,
        xmin=config.kappa_xmin,
        reference_max_size=config.kappa_reference_max_size,
    )
    ancestor_total = float(detection.ancestor_counts.sum())
    branching_ratio = (
        float(detection.descendant_counts.sum() / ancestor_total)
        if ancestor_total > 0
        else float("nan")
    )
    uncertainty = _bootstrap_uncertainty(
        detection,
        config,
        size_xmin=size_fit.xmin,
        duration_xmin=duration_fit.xmin,
        kappa_reference_max_size=kappa_result.reference_max_size,
    )

    return AvalancheResult(
        avalanche_count=int(sizes.size),
        mean_size=float(sizes.mean()) if sizes.size else float("nan"),
        mean_duration_bins=float(durations.mean()) if durations.size else float("nan"),
        mean_duration_seconds=float(durations.mean() * bin_width_seconds) if durations.size else float("nan"),
        size_exponent=size_fit.exponent,
        duration_exponent=duration_fit.exponent,
        size_power_law_fit=size_fit.to_dict(),
        duration_power_law_fit=duration_fit.to_dict(),
        kappa=kappa_result.value,
        kappa_diagnostics=kappa_result.to_dict(),
        branching_ratio=branching_ratio,
        sizes=tuple(float(value) for value in sizes),
        durations_bins=tuple(int(value) for value in durations),
        threshold_z=config.threshold_z,
        bin_width_samples=config.bin_width_samples,
        bin_width_seconds=bin_width_seconds,
        theory_exponent=config.theory_exponent,
        n_channels=segments[0].shape[0],
        n_samples=sum(segment.shape[1] for segment in segments),
        n_segments=len(segments),
        boundary_avalanches_discarded=detection.boundary_avalanches_discarded,
        channel_names=names,
        uncertainty=uncertainty,
        provenance=build_provenance(
            config,
            n_channels=segments[0].shape[0],
            n_samples=sum(segment.shape[1] for segment in segments),
            n_segments=len(segments),
            channel_names=names,
        ),
    )


def validate_batch_compatibility(results: Sequence[AvalancheResult]) -> None:
    """Raise when results should not be compared as one harmonized batch."""
    if not results:
        raise ValueError("at least one result is required")
    first = results[0]
    first_input = first.provenance["input"]
    parameter_sections = (
        "event_detection",
        "temporal_binning",
        "avalanche_definition",
        "power_law_fit",
        "kappa",
        "branching_ratio",
    )
    for index, result in enumerate(results[1:], start=1):
        current_input = result.provenance["input"]
        if result.n_channels != first.n_channels:
            raise ValueError(
                f"result {index} has {result.n_channels} channels; expected {first.n_channels}"
            )
        if (first.channel_names is None) != (result.channel_names is None):
            raise ValueError("channel names are present for only part of the batch")
        if first.channel_names is not None and result.channel_names != first.channel_names:
            raise ValueError(f"result {index} has different channel identities or order")
        if current_input["sampling_rate_hz"] != first_input["sampling_rate_hz"]:
            raise ValueError(f"result {index} has a different sampling rate")
        for section in parameter_sections:
            if result.provenance[section] != first.provenance[section]:
                raise ValueError(f"result {index} has different {section} parameters")
        if (
            result.kappa_diagnostics["reference_max_size"]
            != first.kappa_diagnostics["reference_max_size"]
        ):
            raise ValueError(
                "effective kappa reference maxima differ; set a common "
                "kappa_reference_max_size for harmonized batch comparisons"
            )
