"""Render a kappa methods section from frozen analysis provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from .core import PROVENANCE_SCHEMA_VERSION


def load_provenance(path: str | Path) -> dict[str, object]:
    """Load provenance from a full results JSON or a provenance-only JSON."""
    source = Path(path)
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("JSON root must be an object")
    provenance = document.get("provenance", document)
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a JSON object")
    if provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported provenance schema: {provenance.get('schema_version')!r}; "
            f"expected {PROVENANCE_SCHEMA_VERSION!r}"
        )
    return provenance


def _section(document: Mapping[str, object], name: str) -> Mapping[str, object]:
    section = document.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"provenance is missing the {name!r} section")
    return section


def _number(value: object) -> str:
    if not isinstance(value, (int, float)):
        raise ValueError(f"expected a numeric provenance value, got {value!r}")
    return f"{value:g}"


def render_kappa_methods(provenance: Mapping[str, object]) -> str:
    """Render a journal-style kappa methods section from provenance."""
    software = _section(provenance, "software")
    inputs = _section(provenance, "input")
    event = _section(provenance, "event_detection")
    binning = _section(provenance, "temporal_binning")
    avalanche = _section(provenance, "avalanche_definition")
    kappa = _section(provenance, "kappa")

    sampling_rate = _number(inputs.get("sampling_rate_hz"))
    threshold_z = _number(event.get("threshold_z"))
    ddof = _number(event.get("standard_deviation_ddof"))
    bin_width_value = binning.get("bin_width_samples")
    bin_samples = _number(bin_width_value)
    bin_unit = "sample" if bin_width_value == 1 else "samples"
    bin_ms = _number(float(binning["bin_width_seconds"]) * 1000)
    theory_exponent = _number(kappa.get("theory_exponent"))
    minimum_avalanches = _number(kappa.get("minimum_avalanches"))
    xmin = _number(kappa.get("xmin"))
    requested_points = _number(kappa.get("requested_evaluation_points"))
    package = str(software.get("package"))
    package_version = str(software.get("package_version"))
    algorithm_version = str(kappa.get("algorithm_version"))
    reference_max_size = kappa.get("reference_max_size")
    reference_max_text = (
        "the maximum observed avalanche size"
        if reference_max_size is None
        else f"the prespecified maximum size of {_number(reference_max_size)}"
    )

    if event.get("threshold_operator") != ">":
        raise ValueError("the methods renderer currently supports only a strict '>' threshold")
    if avalanche.get("size") != "number_of_active_channel_bin_pairs":
        raise ValueError("unsupported avalanche size definition")
    if kappa.get("formula") != "1 + mean(reference_cdf - empirical_cdf)":
        raise ValueError("unsupported kappa formula")

    return "\n".join(
        [
            "## Neuronal avalanche kappa coefficient",
            "",
            (
                "Neuronal avalanches were quantified separately within each analysis unit "
                f"from cleaned sensor-space EEG sampled at {sampling_rate} Hz. Discontinuous "
                "recording segments were retained as separate arrays. For each channel, the "
                "mean and population standard deviation were estimated across all samples "
                "pooled over the supplied segments, and the signal was standardized within "
                f"channel (standard-deviation degrees of freedom = {ddof}). Positive and "
                f"negative excursions were treated symmetrically: a sample was marked active "
                f"when its absolute z score was strictly greater than {threshold_z}."
            ),
            "",
            (
                f"Thresholded activity was aggregated into non-overlapping bins of {bin_samples} "
                f"{bin_unit} ({bin_ms} ms). A channel was active within a bin when at least one "
                "sample crossed the threshold. Any incomplete final bin within a segment was "
                "discarded. A time bin was considered active when at least one channel was "
                "active. An avalanche was defined as a sequence of consecutive active bins "
                "bounded by an empty bin or by a segment boundary; avalanches were therefore "
                "not permitted to span discontinuous segments. Avalanches touching the first "
                "or last bin of a segment were discarded because their sizes and durations "
                "were potentially censored. Avalanche size was the total number of active "
                "channel-bin pairs in the sequence."
            ),
            "",
            (
                f"The kappa coefficient was calculated when at least {minimum_avalanches} "
                f"avalanches with size greater than or equal to xmin = {xmin} were available. "
                f"A reference probability mass function proportional to x^(-{theory_exponent}) "
                f"was constructed over every integer size from xmin through {reference_max_text} "
                "and normalized over that complete support. Its cumulative sum defined the "
                f"reference CDF. {requested_points} evaluation points were logarithmically "
                "spaced over the same range without rounding. At each point, the empirical cumulative distribution "
                "function (CDF) was the proportion of observed avalanche sizes less than or "
                "equal to that point, and the reference CDF was evaluated from its full integer "
                "support. Kappa was then computed "
                "as 1 plus the mean, across retained evaluation points, of the empirical CDF "
                "subtracted from the reference CDF. Thus, kappa = 1 indicates mean agreement between the "
                "empirical and reference CDFs at the selected points. Kappa was recorded as "
                "missing when the minimum avalanche count was not met."
            ),
            "",
            (
                f"Calculations used {package} version {package_version}, algorithm "
                f"{algorithm_version}. The complete parameter record was saved with each "
                "analysis in machine-readable JSON."
            ),
            "",
            (
                "This kappa value is a descriptive comparison with a specified reference "
                "distribution and was not treated as a formal goodness-of-fit test or, by "
                "itself, as proof of power-law or critical dynamics."
            ),
            "",
        ]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a kappa methods section from eeg-avalanches JSON provenance."
    )
    parser.add_argument("json", type=Path, help="Full results JSON or provenance-only JSON")
    parser.add_argument("--output", type=Path, help="Markdown output path; defaults to stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    methods = render_kappa_methods(load_provenance(args.json))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(methods, encoding="utf-8")
    else:
        print(methods, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
