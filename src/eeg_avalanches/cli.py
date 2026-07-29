"""Command-line interface for eeg-avalanches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import AvalancheConfig, analyze_avalanches


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute neuronal-avalanche metrics from a cleaned EEG NumPy array."
    )
    parser.add_argument("input", type=Path, help=".npy or .npz input file")
    parser.add_argument("--sampling-rate", type=float, required=True, help="Sampling rate in Hz")
    parser.add_argument("--output", type=Path, help="Output JSON path; defaults to stdout")
    parser.add_argument("--array-key", help="Array key for .npz input")
    parser.add_argument("--threshold-z", type=float, default=2.5)
    parser.add_argument("--bin-width-samples", type=int, default=1)
    parser.add_argument("--theory-exponent", type=float, default=1.5)
    parser.add_argument("--min-fit-events", type=int, default=50)
    parser.add_argument("--kappa-points", type=int, default=10)
    parser.add_argument("--kappa-xmin", type=int, default=1)
    parser.add_argument("--kappa-reference-max-size", type=int)
    parser.add_argument("--size-power-law-xmin", type=int)
    parser.add_argument("--duration-power-law-xmin", type=int)
    parser.add_argument(
        "--keep-boundary-avalanches",
        action="store_true",
        help="Retain potentially censored avalanches touching a segment boundary",
    )
    parser.add_argument(
        "--channel-names",
        help="Comma-separated channel names, used to validate and record channel identity",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=0,
        help="Avalanche-level bootstrap iterations for confidence intervals",
    )
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--parameters-output",
        type=Path,
        help="Also write the frozen analysis provenance to this JSON file",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit size and duration distributions from JSON",
    )
    return parser


def _load_array(path: Path, array_key: str | None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.keys())
            if array_key is None:
                if len(keys) != 1:
                    raise ValueError(f".npz contains {len(keys)} arrays; provide --array-key")
                array_key = keys[0]
            if array_key not in archive:
                raise ValueError(f"array key {array_key!r} not found; available keys: {keys}")
            return archive[array_key]
    raise ValueError("input must be a .npy or .npz file")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    data = _load_array(args.input, args.array_key)
    config = AvalancheConfig(
        sampling_rate=args.sampling_rate,
        threshold_z=args.threshold_z,
        bin_width_samples=args.bin_width_samples,
        theory_exponent=args.theory_exponent,
        min_events_for_distribution_fit=args.min_fit_events,
        kappa_evaluation_points=args.kappa_points,
        kappa_xmin=args.kappa_xmin,
        kappa_reference_max_size=args.kappa_reference_max_size,
        size_power_law_xmin=args.size_power_law_xmin,
        duration_power_law_xmin=args.duration_power_law_xmin,
        discard_boundary_avalanches=not args.keep_boundary_avalanches,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_confidence_level=args.bootstrap_confidence_level,
        random_seed=args.random_seed,
    )
    channel_names = args.channel_names.split(",") if args.channel_names else None
    result = analyze_avalanches(data, config, channel_names=channel_names)
    if args.output:
        result.write_json(
            args.output,
            include_distributions=not args.summary_only,
        )
    else:
        print(result.to_json(include_distributions=not args.summary_only), end="")
    if args.parameters_output:
        args.parameters_output.parent.mkdir(parents=True, exist_ok=True)
        args.parameters_output.write_text(
            json.dumps(result.provenance, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
