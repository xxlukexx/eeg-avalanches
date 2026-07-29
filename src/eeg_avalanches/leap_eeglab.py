"""Adapter for LEAP resting-state EEGLAB files.

MNE is an optional dependency. Install the package with ``eeglab`` extras
before using this module.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]

CONDITION_LABELS = {
    "211": "eyes_open",
    "212": "eyes_closed",
}
END_CODE = "215"
VALID_CODE = "213"
INVALID_CODE = "214"


@dataclass(frozen=True)
class RestInterval:
    """One resting-state interval described by EEGLAB annotations."""

    condition_code: str
    condition_label: str
    start_sec: float
    end_sec: float
    validity_code: str | None
    is_valid: bool | None

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class EEGSegment:
    """One continuous EEG block or epoch in channels-by-samples orientation."""

    data_uv: FloatArray
    sampling_rate: float
    channel_names: tuple[str, ...]
    dropped_nonfinite_channels: tuple[str, ...]
    dropped_flat_channels: tuple[str, ...]
    condition_code: str
    condition_label: str
    source_index: int
    start_sec: float | None
    end_sec: float | None
    validity_code: str | None
    is_valid: bool | None

    @property
    def duration_sec(self) -> float:
        return self.data_uv.shape[1] / self.sampling_rate


@dataclass(frozen=True)
class LEAPEEGLABData:
    """In-memory representation of one EEGLAB file."""

    source_path: Path
    source_kind: str
    segments: tuple[EEGSegment, ...]

    def by_condition(self) -> dict[str, list[FloatArray]]:
        """Return arrays grouped by condition, preserving segment boundaries."""
        grouped: dict[str, list[FloatArray]] = {}
        for segment in self.segments:
            grouped.setdefault(segment.condition_label, []).append(segment.data_uv)
        return grouped


def _require_mne():
    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "EEGLAB support requires optional dependencies. "
            "Install with: python -m pip install 'eeg-avalanches[eeglab]'"
        ) from exc
    return mne


def _canonical_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _canonical_channel_name(value: str) -> str:
    return value.strip().upper().replace(" ", "")


def parse_rest_intervals(
    descriptions: Sequence[object],
    onsets_sec: Sequence[float],
) -> list[RestInterval]:
    """Parse LEAP eyes-open/closed intervals from EEGLAB annotations."""
    if len(descriptions) != len(onsets_sec):
        raise ValueError("descriptions and onsets_sec must have equal length")

    annotations = [
        (_canonical_code(description), float(onset))
        for description, onset in zip(descriptions, onsets_sec, strict=True)
    ]
    intervals: list[RestInterval] = []

    index = 0
    while index < len(annotations):
        condition_code, start_sec = annotations[index]
        if condition_code not in CONDITION_LABELS:
            index += 1
            continue

        end_sec: float | None = None
        validity_code: str | None = None
        cursor = index + 1
        next_index: int | None = None
        while cursor < len(annotations):
            next_code, next_onset = annotations[cursor]
            if next_code == END_CODE:
                end_sec = next_onset
                if cursor + 1 < len(annotations):
                    following_code = annotations[cursor + 1][0]
                    if following_code in {VALID_CODE, INVALID_CODE}:
                        validity_code = following_code
                        cursor += 1
                next_index = cursor + 1
                break
            if next_code in {VALID_CODE, INVALID_CODE}:
                end_sec = next_onset
                validity_code = next_code
                next_index = cursor + 1
                break
            if next_code in CONDITION_LABELS:
                end_sec = next_onset
                next_index = cursor
                break
            cursor += 1

        if end_sec is not None and end_sec > start_sec:
            is_valid = None
            if validity_code == VALID_CODE:
                is_valid = True
            elif validity_code == INVALID_CODE:
                is_valid = False
            intervals.append(
                RestInterval(
                    condition_code=condition_code,
                    condition_label=CONDITION_LABELS[condition_code],
                    start_sec=start_sec,
                    end_sec=end_sec,
                    validity_code=validity_code,
                    is_valid=is_valid,
                )
            )
        index = next_index if next_index is not None else cursor + 1

    return intervals


def _select_channels(
    data: np.ndarray,
    channel_names: Sequence[str],
    channels: Iterable[str] | None,
    channel_axis: int = 0,
) -> tuple[np.ndarray, tuple[str, ...]]:
    canonical_names = tuple(_canonical_channel_name(name) for name in channel_names)
    if len(set(canonical_names)) != len(canonical_names):
        raise ValueError("channel names are not unique after canonicalization")
    if channels is None:
        return data, canonical_names

    requested = [_canonical_channel_name(name) for name in channels]
    index_by_name = {name: index for index, name in enumerate(canonical_names)}
    missing = [name for name in requested if name not in index_by_name]
    if missing:
        raise ValueError(f"requested channels not found: {missing}")
    indices = [index_by_name[name] for name in requested]
    return np.take(data, indices, axis=channel_axis), tuple(requested)


def _handle_nonfinite_channels(
    data: np.ndarray,
    channel_names: tuple[str, ...],
    policy: str,
    channel_axis: int = 0,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if policy not in {"drop", "error"}:
        raise ValueError("nonfinite_policy must be 'drop' or 'error'")
    reduction_axes = tuple(axis for axis in range(data.ndim) if axis != channel_axis)
    finite_channels = np.all(np.isfinite(data), axis=reduction_axes)
    if np.all(finite_channels):
        return data, channel_names, ()

    dropped = tuple(name for name, keep in zip(channel_names, finite_channels, strict=True) if not keep)
    if policy == "error":
        raise ValueError(f"non-finite data found in channels: {list(dropped)}")
    retained = tuple(name for name, keep in zip(channel_names, finite_channels, strict=True) if keep)
    if not retained:
        raise ValueError("no finite channels remain")
    indices = np.flatnonzero(finite_channels)
    return np.take(data, indices, axis=channel_axis), retained, dropped


def _handle_flat_channels(
    data: np.ndarray,
    channel_names: tuple[str, ...],
    policy: str,
    channel_axis: int = 0,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if policy not in {"drop", "error"}:
        raise ValueError("flat_channel_policy must be 'drop' or 'error'")
    variability_axes = tuple(axis for axis in range(data.ndim) if axis != channel_axis)
    channel_sd = np.std(data, axis=variability_axes)
    retained_mask = channel_sd > 0
    if np.all(retained_mask):
        return data, channel_names, ()

    dropped = tuple(
        name for name, keep in zip(channel_names, retained_mask, strict=True) if not keep
    )
    if policy == "error":
        raise ValueError(f"zero-standard-deviation channels found: {list(dropped)}")
    retained = tuple(
        name for name, keep in zip(channel_names, retained_mask, strict=True) if keep
    )
    if not retained:
        raise ValueError("no non-flat channels remain")
    indices = np.flatnonzero(retained_mask)
    return np.take(data, indices, axis=channel_axis), retained, dropped


def _resample(data: np.ndarray, source_rate: float, target_rate: float | None) -> tuple[np.ndarray, float]:
    if target_rate is None or np.isclose(source_rate, target_rate):
        return data, float(source_rate)
    if not np.isfinite(target_rate) or target_rate <= 0:
        raise ValueError("target_rate must be positive and finite")

    from scipy.signal import resample_poly

    ratio = Fraction(float(target_rate) / float(source_rate)).limit_denominator(10_000)
    resampled = resample_poly(data.astype(np.float64, copy=False), ratio.numerator, ratio.denominator, axis=-1)
    return resampled, float(target_rate)


def _load_continuous(
    path: Path,
    *,
    valid_only: bool,
    min_duration_sec: float,
    target_rate: float | None,
    channels: Iterable[str] | None,
    nonfinite_policy: str,
    flat_channel_policy: str,
) -> LEAPEEGLABData:
    mne = _require_mne()
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    intervals = parse_rest_intervals(raw.annotations.description, raw.annotations.onset)
    if not intervals:
        raise ValueError("no LEAP resting-state intervals were found")

    data_uv = raw.get_data(picks="eeg") * 1e6
    eeg_channel_names = [raw.ch_names[index] for index in mne.pick_types(raw.info, eeg=True)]
    data_uv, channel_names = _select_channels(data_uv, eeg_channel_names, channels)
    data_uv, channel_names, dropped_channels = _handle_nonfinite_channels(
        data_uv,
        channel_names,
        nonfinite_policy,
    )
    data_uv, channel_names, dropped_flat_channels = _handle_flat_channels(
        data_uv,
        channel_names,
        flat_channel_policy,
    )

    segments: list[EEGSegment] = []
    for source_index, interval in enumerate(intervals, start=1):
        if valid_only and interval.is_valid is False:
            continue
        if interval.duration_sec < min_duration_sec:
            continue
        start = max(0, int(round(interval.start_sec * raw.info["sfreq"])))
        end = min(data_uv.shape[1], int(round(interval.end_sec * raw.info["sfreq"])))
        if end <= start:
            continue
        segment_data, sampling_rate = _resample(
            data_uv[:, start:end],
            float(raw.info["sfreq"]),
            target_rate,
        )
        segments.append(
            EEGSegment(
                data_uv=segment_data.astype(np.float32, copy=False),
                sampling_rate=sampling_rate,
                channel_names=channel_names,
                dropped_nonfinite_channels=dropped_channels,
                dropped_flat_channels=dropped_flat_channels,
                condition_code=interval.condition_code,
                condition_label=interval.condition_label,
                source_index=source_index,
                start_sec=interval.start_sec,
                end_sec=interval.end_sec,
                validity_code=interval.validity_code,
                is_valid=interval.is_valid,
            )
        )
    return LEAPEEGLABData(path, "continuous", tuple(segments))


def _load_epoched(
    path: Path,
    *,
    min_duration_sec: float,
    target_rate: float | None,
    channels: Iterable[str] | None,
    nonfinite_policy: str,
    flat_channel_policy: str,
) -> LEAPEEGLABData:
    mne = _require_mne()
    epochs = mne.io.read_epochs_eeglab(path, verbose="ERROR")
    if target_rate is not None and not np.isclose(epochs.info["sfreq"], target_rate):
        epochs = epochs.copy().resample(target_rate)

    data_uv = epochs.get_data(picks="eeg", copy=True) * 1e6
    eeg_channel_names = [epochs.ch_names[index] for index in mne.pick_types(epochs.info, eeg=True)]
    data_uv, channel_names = _select_channels(
        data_uv,
        eeg_channel_names,
        channels,
        channel_axis=1,
    )
    data_uv, channel_names, dropped_channels = _handle_nonfinite_channels(
        data_uv,
        channel_names,
        nonfinite_policy,
        channel_axis=1,
    )
    data_uv, channel_names, dropped_flat_channels = _handle_flat_channels(
        data_uv,
        channel_names,
        flat_channel_policy,
        channel_axis=1,
    )
    inverse_event_id = {value: _canonical_code(key) for key, value in epochs.event_id.items()}
    segments: list[EEGSegment] = []

    for source_index, (epoch_data, event) in enumerate(zip(data_uv, epochs.events, strict=True), start=1):
        condition_code = inverse_event_id.get(int(event[2]), str(int(event[2])))
        if condition_code not in CONDITION_LABELS:
            continue
        duration_sec = epoch_data.shape[1] / float(epochs.info["sfreq"])
        if duration_sec < min_duration_sec:
            continue
        segments.append(
            EEGSegment(
                data_uv=epoch_data.astype(np.float32, copy=False),
                sampling_rate=float(epochs.info["sfreq"]),
                channel_names=channel_names,
                dropped_nonfinite_channels=dropped_channels,
                dropped_flat_channels=dropped_flat_channels,
                condition_code=condition_code,
                condition_label=CONDITION_LABELS[condition_code],
                source_index=source_index,
                start_sec=float(epochs.tmin),
                end_sec=float(epochs.tmax),
                validity_code=None,
                is_valid=None,
            )
        )
    return LEAPEEGLABData(path, "epoched", tuple(segments))


def load_leap_eeglab(
    path: str | Path,
    *,
    mode: str = "auto",
    valid_only: bool = True,
    min_duration_sec: float = 0.0,
    target_rate: float | None = None,
    channels: Iterable[str] | None = None,
    nonfinite_policy: str = "drop",
    flat_channel_policy: str = "drop",
) -> LEAPEEGLABData:
    """Load and condition-split one LEAP EEGLAB file in memory.

    ``mode`` may be ``"continuous"``, ``"epoched"``, or ``"auto"``.
    Data are returned in microvolts as float32 channels-by-samples arrays.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".set":
        raise ValueError(f"expected an existing EEGLAB .set file: {source}")
    if mode not in {"auto", "continuous", "epoched"}:
        raise ValueError("mode must be 'auto', 'continuous', or 'epoched'")
    if min_duration_sec < 0:
        raise ValueError("min_duration_sec cannot be negative")

    kwargs = {
        "min_duration_sec": min_duration_sec,
        "target_rate": target_rate,
        "channels": channels,
        "nonfinite_policy": nonfinite_policy,
        "flat_channel_policy": flat_channel_policy,
    }
    if mode == "continuous":
        return _load_continuous(source, valid_only=valid_only, **kwargs)
    if mode == "epoched":
        return _load_epoched(source, **kwargs)

    try:
        return _load_continuous(source, valid_only=valid_only, **kwargs)
    except (TypeError, ValueError, RuntimeError) as continuous_error:
        try:
            return _load_epoched(source, **kwargs)
        except (TypeError, ValueError, RuntimeError) as epoched_error:
            raise ValueError(
                f"could not load {source.name} as continuous or epoched EEGLAB data; "
                f"continuous error: {continuous_error}; epoched error: {epoched_error}"
            ) from epoched_error


def save_as_npy(dataset: LEAPEEGLABData, output_dir: str | Path) -> list[dict[str, object]]:
    """Save each segment as `.npy` with a JSON sidecar."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for segment in dataset.segments:
        base_name = (
            f"{dataset.source_path.stem}__segment-{segment.source_index:03d}"
            f"__{segment.condition_label}"
        )
        array_path = destination / f"{base_name}.npy"
        metadata_path = destination / f"{base_name}.json"
        np.save(array_path, segment.data_uv, allow_pickle=False)
        metadata = {
            "source_file": dataset.source_path.name,
            "source_kind": dataset.source_kind,
            "array_file": array_path.name,
            "segment_index": segment.source_index,
            "condition_code": segment.condition_code,
            "condition_label": segment.condition_label,
            "validity_code": segment.validity_code,
            "is_valid": segment.is_valid,
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
            "duration_sec": segment.duration_sec,
            "sampling_rate": segment.sampling_rate,
            "channel_names": list(segment.channel_names),
            "dropped_nonfinite_channels": list(segment.dropped_nonfinite_channels),
            "dropped_flat_channels": list(segment.dropped_flat_channels),
            "n_channels": segment.data_uv.shape[0],
            "n_samples": segment.data_uv.shape[1],
            "units": "microvolts",
            "array_orientation": "channels_by_samples",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        rows.append(
            {
                **metadata,
                "channel_names": "|".join(segment.channel_names),
                "dropped_nonfinite_channels": "|".join(segment.dropped_nonfinite_channels),
                "dropped_flat_channels": "|".join(segment.dropped_flat_channels),
            }
        )
    return rows


def convert_leap_eeglab(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    recursive: bool = False,
    **load_options: object,
) -> list[dict[str, object]]:
    """Convert one `.set` file or a directory of files to `.npy` segments."""
    source = Path(input_path).expanduser().resolve()
    if source.is_file():
        set_files = [source]
    elif source.is_dir():
        pattern = "**/*.set" if recursive else "*.set"
        set_files = sorted(source.glob(pattern))
    else:
        raise ValueError(f"input path does not exist: {source}")
    if not set_files:
        raise ValueError(f"no .set files found under: {source}")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for set_file in set_files:
        try:
            dataset = load_leap_eeglab(set_file, **load_options)
            rows.extend(save_as_npy(dataset, destination))
        except Exception as exc:  # Continue a batch while retaining an explicit error log.
            errors.append({"source_file": set_file.name, "error": f"{type(exc).__name__}: {exc}"})

    manifest_path = destination / "manifest.csv"
    if rows:
        fieldnames = list(rows[0])
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    (destination / "errors.json").write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split LEAP resting-state EEGLAB data into condition-labelled NumPy arrays."
    )
    parser.add_argument("input", type=Path, help="EEGLAB .set file or directory")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--mode", choices=["auto", "continuous", "epoched"], default="auto")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--min-duration", type=float, default=0.0, metavar="SECONDS")
    parser.add_argument("--target-rate", type=float, metavar="HZ")
    parser.add_argument("--channels", help="Comma-separated channel names")
    parser.add_argument(
        "--nonfinite",
        choices=["drop", "error"],
        default="drop",
        help="Drop channels containing non-finite samples, or stop with an error",
    )
    parser.add_argument(
        "--flat",
        choices=["drop", "error"],
        default="drop",
        help="Drop zero-standard-deviation channels, or stop with an error",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    channels = args.channels.split(",") if args.channels else None
    rows = convert_leap_eeglab(
        args.input,
        args.output,
        recursive=args.recursive,
        mode=args.mode,
        valid_only=not args.include_invalid,
        min_duration_sec=args.min_duration,
        target_rate=args.target_rate,
        channels=channels,
        nonfinite_policy=args.nonfinite,
        flat_channel_policy=args.flat,
    )
    print(json.dumps({"saved_segments": len(rows), "output_dir": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
