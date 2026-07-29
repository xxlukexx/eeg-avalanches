from pathlib import Path

import numpy as np

from eeg_avalanches.leap_eeglab import (
    EEGSegment,
    LEAPEEGLABData,
    _handle_flat_channels,
    _handle_nonfinite_channels,
    parse_rest_intervals,
    save_as_npy,
)


def test_parses_validity_markers_as_interval_ends() -> None:
    intervals = parse_rest_intervals(
        ["boundary", "212", "213", "211", "214"],
        [0.0, 10.0, 42.0, 50.0, 82.0],
    )

    assert [interval.condition_label for interval in intervals] == ["eyes_closed", "eyes_open"]
    assert [interval.is_valid for interval in intervals] == [True, False]
    assert [interval.duration_sec for interval in intervals] == [32.0, 32.0]


def test_parses_explicit_end_then_validity_marker() -> None:
    intervals = parse_rest_intervals(
        ["211", "215", "213", "212", "215", "214"],
        [1.0, 31.0, 31.1, 40.0, 70.0, 70.1],
    )

    assert len(intervals) == 2
    assert intervals[0].is_valid is True
    assert intervals[1].is_valid is False


def test_next_condition_can_close_previous_interval() -> None:
    intervals = parse_rest_intervals(
        ["211", "212", "213"],
        [1.0, 31.0, 61.0],
    )

    assert [interval.condition_label for interval in intervals] == ["eyes_open", "eyes_closed"]
    assert intervals[0].is_valid is None
    assert intervals[1].is_valid is True


def test_groups_arrays_without_joining_boundaries() -> None:
    segment = EEGSegment(
        data_uv=np.ones((2, 10), dtype=np.float32),
        sampling_rate=100.0,
        channel_names=("FZ", "CZ"),
        dropped_nonfinite_channels=(),
        dropped_flat_channels=(),
        condition_code="211",
        condition_label="eyes_open",
        source_index=1,
        start_sec=0.0,
        end_sec=0.1,
        validity_code="213",
        is_valid=True,
    )
    dataset = LEAPEEGLABData(Path("recording.set"), "continuous", (segment, segment))

    grouped = dataset.by_condition()

    assert len(grouped["eyes_open"]) == 2
    assert grouped["eyes_open"][0].shape == (2, 10)


def test_drops_nonfinite_channels_consistently_across_epochs() -> None:
    data = np.ones((3, 2, 5))
    data[1, 1, 2] = np.nan

    cleaned, retained, dropped = _handle_nonfinite_channels(
        data,
        ("FZ", "CZ"),
        "drop",
        channel_axis=1,
    )

    assert cleaned.shape == (3, 1, 5)
    assert retained == ("FZ",)
    assert dropped == ("CZ",)


def test_drops_flat_channels_consistently_across_epochs() -> None:
    data = np.ones((3, 2, 5))
    data[:, 0, :] = np.arange(5)

    cleaned, retained, dropped = _handle_flat_channels(
        data,
        ("FZ", "CZ"),
        "drop",
        channel_axis=1,
    )

    assert cleaned.shape == (3, 1, 5)
    assert retained == ("FZ",)
    assert dropped == ("CZ",)


def test_saves_array_and_metadata(tmp_path: Path) -> None:
    segment = EEGSegment(
        data_uv=np.arange(12, dtype=np.float32).reshape(2, 6),
        sampling_rate=200.0,
        channel_names=("FZ", "CZ"),
        dropped_nonfinite_channels=(),
        dropped_flat_channels=(),
        condition_code="212",
        condition_label="eyes_closed",
        source_index=3,
        start_sec=10.0,
        end_sec=10.03,
        validity_code="213",
        is_valid=True,
    )
    dataset = LEAPEEGLABData(Path("recording.set"), "continuous", (segment,))

    rows = save_as_npy(dataset, tmp_path)

    saved = np.load(tmp_path / rows[0]["array_file"], allow_pickle=False)
    np.testing.assert_array_equal(saved, segment.data_uv)
    assert (tmp_path / "recording__segment-003__eyes_closed.json").is_file()
