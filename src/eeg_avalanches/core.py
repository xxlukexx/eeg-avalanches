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


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
PROVENANCE_SCHEMA_VERSION = "eeg-avalanches-provenance-1.0"
KAPPA_ALGORITHM_VERSION = "kappa-logpoint-cdf-v1"


@dataclass(frozen=True)
class AvalancheConfig:
    """Analysis parameters."""

    sampling_rate: float
    threshold_z: float = 2.5
    bin_width_samples: int = 1
    theory_exponent: float = 1.5
    min_events_for_distribution_fit: int = 20
    kappa_evaluation_points: int = 10
    kappa_xmin: float = 1.0

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
        if not np.isfinite(self.kappa_xmin) or self.kappa_xmin < 1:
            raise ValueError("kappa_xmin must be finite and at least 1")

    def to_dict(self) -> dict[str, float | int]:
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
    kappa: float
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
) -> dict[str, object]:
    """Build a complete, JSON-serializable record of kappa analysis choices."""
    bin_width_seconds = config.bin_width_samples / config.sampling_rate
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "software": {
            "package": "eeg-avalanches",
            "package_version": _package_version(),
            "algorithm": "sensor-space absolute-threshold neuronal avalanches",
            "algorithm_version": KAPPA_ALGORITHM_VERSION,
        },
        "input": {
            "sampling_rate_hz": config.sampling_rate,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "n_segments": n_segments,
            "array_orientation": "channels_by_samples",
            "segments_treated_as_discontinuous": True,
        },
        "event_detection": {
            "standardization_scope": "per_channel_pooled_across_supplied_segments",
            "standard_deviation_ddof": 0,
            "threshold_type": "two_sided_absolute_z_score",
            "threshold_operator": ">",
            "threshold_z": config.threshold_z,
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
        },
        "kappa": {
            "theory_exponent": config.theory_exponent,
            "minimum_avalanches": config.min_events_for_distribution_fit,
            "xmin": config.kappa_xmin,
            "requested_evaluation_points": config.kappa_evaluation_points,
            "evaluation_points": "log_spaced_xmin_to_observed_max_rounded_to_unique_integers",
            "empirical_cdf": "proportion_of_avalanche_sizes_less_than_or_equal_to_each_point",
            "reference_cdf": "cumulative_normalized_point_mass_x_to_negative_theory_exponent",
            "formula": "1 + mean(empirical_cdf - reference_cdf)",
            "insufficient_data_value": None,
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
    pooled = np.concatenate(segments, axis=1)
    means = pooled.mean(axis=1, keepdims=True)
    standard_deviations = pooled.std(axis=1, keepdims=True)
    safe_standard_deviations = np.where(standard_deviations == 0, 1.0, standard_deviations)

    return [
        _bin_events(np.abs((segment - means) / safe_standard_deviations) > threshold_z, bin_width_samples)
        for segment in segments
    ]


def detect_avalanches(event_segments: Iterable[ArrayLike]) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Extract sizes, durations, and within-avalanche branching ratios.

    Each input is a channels-by-time-bin Boolean array. An avalanche is a run
    of non-empty bins bounded by empty bins or by a segment boundary. Size is
    the number of active channel-bin pairs; duration is the number of bins.
    """
    sizes: list[float] = []
    durations: list[float] = []
    branching_ratios: list[float] = []
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
            counts = bin_counts[start:index].astype(np.float64)
            sizes.append(float(counts.sum()))
            durations.append(float(counts.size))
            if counts.size > 1:
                branching_ratios.extend((counts[1:] / counts[:-1]).tolist())

    return (
        np.asarray(sizes, dtype=np.float64),
        np.asarray(durations, dtype=np.float64),
        np.asarray(branching_ratios, dtype=np.float64),
    )


def power_law_exponent(
    values: ArrayLike,
    min_observations: int = 20,
    xmin: float = 1.0,
) -> float:
    """Estimate a continuous power-law exponent with fixed ``xmin``."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array >= xmin)]
    if array.size < min_observations:
        return float("nan")
    denominator = np.log(array / xmin).sum()
    if denominator <= 0:
        return float("nan")
    return float(1.0 + array.size / denominator)


def kappa_against_theory(
    values: ArrayLike,
    theory_exponent: float = 1.5,
    min_observations: int = 20,
    evaluation_points: int = 10,
    xmin: float = 1.0,
) -> float:
    """Compare the empirical size CDF with a reference power-law CDF.

    This retains the ten-point heuristic used in the originating analysis.
    It is a descriptive index, not a formal goodness-of-fit test.
    """
    if not np.isfinite(theory_exponent) or theory_exponent <= 1:
        raise ValueError("theory_exponent must be greater than 1")
    if min_observations < 2:
        raise ValueError("min_observations must be at least 2")
    if evaluation_points < 2:
        raise ValueError("evaluation_points must be at least 2")
    if not np.isfinite(xmin) or xmin < 1:
        raise ValueError("xmin must be finite and at least 1")

    array = np.asarray(values, dtype=np.float64)
    array = np.sort(array[np.isfinite(array) & (array >= xmin)])
    if array.size < min_observations:
        return float("nan")

    points = np.unique(
        np.round(
            np.logspace(np.log10(xmin), np.log10(array.max()), evaluation_points)
        ).astype(int)
    )
    empirical_cdf = np.asarray([np.mean(array <= point) for point in points], dtype=np.float64)
    reference_mass = points.astype(np.float64) ** (-theory_exponent)
    reference_cdf = np.cumsum(reference_mass / reference_mass.sum())
    return float(1.0 + np.mean(empirical_cdf - reference_cdf))


def analyze_avalanches(
    data: ArrayLike | Sequence[ArrayLike],
    config: AvalancheConfig,
) -> AvalancheResult:
    """Compute neuronal-avalanche metrics from cleaned EEG data."""
    segments = _as_segments(data)
    events = threshold_events(
        segments,
        threshold_z=config.threshold_z,
        bin_width_samples=config.bin_width_samples,
    )
    sizes, durations, branching = detect_avalanches(events)
    bin_width_seconds = config.bin_width_samples / config.sampling_rate

    return AvalancheResult(
        avalanche_count=int(sizes.size),
        mean_size=float(sizes.mean()) if sizes.size else float("nan"),
        mean_duration_bins=float(durations.mean()) if durations.size else float("nan"),
        mean_duration_seconds=float(durations.mean() * bin_width_seconds) if durations.size else float("nan"),
        size_exponent=power_law_exponent(
            sizes,
            min_observations=config.min_events_for_distribution_fit,
        ),
        duration_exponent=power_law_exponent(
            durations,
            min_observations=config.min_events_for_distribution_fit,
        ),
        kappa=kappa_against_theory(
            sizes,
            theory_exponent=config.theory_exponent,
            min_observations=config.min_events_for_distribution_fit,
            evaluation_points=config.kappa_evaluation_points,
            xmin=config.kappa_xmin,
        ),
        branching_ratio=float(branching.mean()) if branching.size else float("nan"),
        sizes=tuple(float(value) for value in sizes),
        durations_bins=tuple(int(value) for value in durations),
        threshold_z=config.threshold_z,
        bin_width_samples=config.bin_width_samples,
        bin_width_seconds=bin_width_seconds,
        theory_exponent=config.theory_exponent,
        n_channels=segments[0].shape[0],
        n_samples=sum(segment.shape[1] for segment in segments),
        n_segments=len(segments),
        provenance=build_provenance(
            config,
            n_channels=segments[0].shape[0],
            n_samples=sum(segment.shape[1] for segment in segments),
            n_segments=len(segments),
        ),
    )
