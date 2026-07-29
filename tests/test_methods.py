import json
from pathlib import Path

import numpy as np

from eeg_avalanches import AvalancheConfig, analyze_avalanches
from eeg_avalanches.core import build_provenance
from eeg_avalanches.methods import load_provenance, render_kappa_methods


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_result_contains_frozen_kappa_provenance() -> None:
    result = analyze_avalanches(
        np.stack(
            [
                np.sin(np.linspace(0, 2 * np.pi * frequency, 100, endpoint=False))
                for frequency in range(1, 5)
            ]
        ),
        AvalancheConfig(
            sampling_rate=200.0,
            threshold_z=2.75,
            bin_width_samples=2,
            theory_exponent=1.6,
            min_events_for_distribution_fit=50,
            kappa_evaluation_points=12,
            kappa_xmin=2.0,
        ),
    )

    provenance = result.provenance
    assert provenance["event_detection"]["threshold_z"] == 2.75
    assert provenance["temporal_binning"]["bin_width_seconds"] == 0.01
    assert provenance["kappa"]["theory_exponent"] == 1.6
    assert provenance["kappa"]["minimum_avalanches"] == 50
    assert provenance["kappa"]["requested_evaluation_points"] == 12
    assert provenance["kappa"]["xmin"] == 2.0
    assert provenance["input"]["n_channels"] == 4
    assert provenance["input"]["n_samples"] == 100


def test_write_json_uses_null_for_undefined_metrics(tmp_path: Path) -> None:
    result = analyze_avalanches(
        np.stack(
            [
                np.sin(np.linspace(0, 2 * np.pi, 20, endpoint=False)),
                np.cos(np.linspace(0, 2 * np.pi, 20, endpoint=False)),
            ]
        ),
        AvalancheConfig(sampling_rate=100.0),
    )
    output = tmp_path / "result.json"

    result.write_json(output, include_distributions=False)
    document = json.loads(output.read_text(encoding="utf-8"))

    assert document["kappa"] is None
    assert document["mean_size"] is None
    assert document["provenance"]["event_detection"]["threshold_z"] == 2.5


def test_default_methods_file_is_generated_from_default_json() -> None:
    parameters_path = REPOSITORY_ROOT / "methods" / "kappa_default_parameters.json"
    expected_path = REPOSITORY_ROOT / "methods" / "kappa_default_methods.md"

    rendered = render_kappa_methods(load_provenance(parameters_path))

    assert rendered == expected_path.read_text(encoding="utf-8")


def test_default_parameter_json_matches_code_defaults() -> None:
    parameters_path = REPOSITORY_ROOT / "methods" / "kappa_default_parameters.json"
    committed = json.loads(parameters_path.read_text(encoding="utf-8"))

    assert committed == build_provenance(AvalancheConfig(sampling_rate=200.0))


def test_renderer_accepts_full_results_json(tmp_path: Path) -> None:
    result = analyze_avalanches(
        np.stack(
            [
                np.sin(np.linspace(0, 2 * np.pi, 20, endpoint=False)),
                np.cos(np.linspace(0, 2 * np.pi, 20, endpoint=False)),
            ]
        ),
        AvalancheConfig(sampling_rate=200.0),
    )
    output = tmp_path / "result.json"
    result.write_json(output)

    rendered = render_kappa_methods(load_provenance(output))

    assert "strictly greater than 2.5" in rendered
    assert "kappa-discrete-full-cdf-v2" in rendered
