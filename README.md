# EEG Avalanches

A small Python package for extracting sensor-space neuronal-avalanche metrics
from cleaned EEG arrays.

The package identifies high-amplitude channel events, groups consecutive
non-empty time bins into avalanches, and reports avalanche burden and
criticality-related summaries. It is intended for reproducible research and
sensitivity analysis rather than as a validated measure of excitation/inhibition
balance.

## Method

For each analysis unit (for example, one participant and condition):

1. Pool all supplied segments to estimate each channel's mean and standard
   deviation.
2. Mark positive and negative excursions where `abs(z) > threshold_z`.
3. Optionally combine adjacent samples into temporal bins. A channel is active
   in a bin if it contains at least one threshold-crossing sample.
4. Define an avalanche as consecutive non-empty bins bounded by an empty bin or
   a segment boundary.
5. Discard avalanches touching a segment boundary because they may be censored.
6. Define avalanche size as active channel-bin pairs and duration as bins.

The corrected default parameters are:

- absolute z threshold: `2.5`
- bin width: one sample
- theoretical size exponent for kappa: `1.5`
- minimum avalanches for distribution summaries: `50`
- kappa reference: truncated discrete power law over the complete integer
  support
- exponent fit: exact discrete MLE with `xmin` selected by minimum KS distance
- branching ratio: terminal-inclusive ratio of sums

Reported outputs include count, mean size, mean duration, size and duration
power-law exponents, kappa, branching ratio, and the underlying size and
duration distributions. Versions through `0.3` used biased legacy estimators;
see [`CHANGELOG.md`](CHANGELOG.md) before comparing old and new outputs.

The corrected branching estimator removes the arithmetic bias caused by
omitting avalanche termination, but it does not remove bias from spatial
subsampling or volume conduction. Kappa remains a descriptive CDF comparison,
not a formal goodness-of-fit test or proof of critical dynamics.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[test]"
pytest
```

For EEGLAB input support:

```bash
python -m pip install -e ".[eeglab]"
```

## Python usage

Input may be a two-dimensional `channels x samples` array, a three-dimensional
`segments x channels x samples` array, or a list of two-dimensional arrays.
Discontinuous epochs should be supplied separately so avalanches cannot cross
epoch boundaries.

```python
import numpy as np

from eeg_avalanches import AvalancheConfig, analyze_avalanches

epochs = np.load("clean_eeg_epochs.npy")
config = AvalancheConfig(
    sampling_rate=250.0,
    threshold_z=2.5,
    bin_width_samples=1,
)

result = analyze_avalanches(epochs, config)
print(result.to_dict(include_distributions=False))
result.write_json("avalanche_results.json", include_distributions=False)
print(result.provenance)
```

Pass channel names whenever possible so identity and order are checked and
recorded:

```python
result = analyze_avalanches(
    epochs,
    config,
    channel_names=["F3", "F4", "C3", "C4"],
)
```

## LEAP EEGLAB adapter

The optional adapter reads continuous or epoched EEGLAB `.set` files and
separates the LEAP eyes-open and eyes-closed resting blocks. In memory, each
discontinuous block remains a separate channels-by-samples array:

```python
from eeg_avalanches.leap_eeglab import load_leap_eeglab

dataset = load_leap_eeglab(
    "clean_resting_state.set",
    target_rate=200.0,       # optional
    min_duration_sec=10.0,
)
arrays_by_condition = dataset.by_condition()
```

The grouped lists can be passed directly to `analyze_avalanches`, one condition
at a time:

```python
eyes_open = analyze_avalanches(
    arrays_by_condition["eyes_open"],
    AvalancheConfig(sampling_rate=200.0),
    channel_names=dataset.segments[0].channel_names,
)
```

To convert one file or a directory to `.npy`:

```bash
leap-eeglab-to-npy clean_resting_state.set converted/ \
  --target-rate 200 \
  --min-duration 10
```

The converter writes one `.npy` per resting block, a JSON metadata sidecar,
`manifest.csv`, and `errors.json`. It does not concatenate blocks. Use
`--channels F3,F4,C3,C4` for an explicit channel subset, `--recursive` for
nested folders, or `--include-invalid` to retain blocks marked invalid.
Channels containing non-finite values are dropped consistently across the
recording and listed in the metadata; use `--nonfinite error` to stop instead.
Zero-standard-deviation channels are handled the same way and can be made fatal
with `--flat error`.

## Command line

The command accepts `.npy` and `.npz` files. Arrays must use the orientation
described above.

```bash
eeg-avalanches clean_eeg_epochs.npy \
  --sampling-rate 250 \
  --output avalanche_metrics.json
```

Use `--array-key` when an `.npz` archive contains more than one array. Use
`--summary-only` to omit the raw distributions from JSON.

Use `--bootstrap-iterations 500 --random-seed 0` to request reproducible
avalanche-level confidence intervals. For harmonized group analyses, provide
`--channel-names` and call `validate_batch_compatibility` on the resulting
Python objects before combining them. Kappa follows the original convention of
using each analysis unit's observed maximum by default; for a group comparison,
prespecify a common support with `--kappa-reference-max-size` so the validator
can confirm that all units used the same reference CDF.

## Reproducible kappa methods

Every results JSON now includes a complete `provenance` object with the input
parameters and fixed algorithm choices used for kappa. Save the same record
separately with `--parameters-output frozen_parameters.json`, then render a
methods section from either JSON file:

```bash
eeg-avalanches-methods frozen_parameters.json --output kappa_methods.md
```

See [`methods/README.md`](methods/README.md) for the workflow,
[`methods/kappa_default_parameters.json`](methods/kappa_default_parameters.json)
for the frozen defaults, and
[`methods/kappa_default_methods.md`](methods/kappa_default_methods.md) for the
complete default journal-style output.

## Validation

Deterministic simulations recover a known discrete exponent, kappa reference,
and critical branching ratio:

```bash
python simulations/validate_estimators.py
```

The former calculations remain under `eeg_avalanches.legacy` solely to
reproduce outputs from versions through `0.3`.

## Input expectations

- Data should already be cleaned, channel-aligned, and consistently referenced.
- All segments in one call must contain the same channels in the same order.
- Flat channels raise an error and must be handled by the cleaning pipeline.
- Values must be finite; handle missing samples before analysis.
- Analyze conditions separately unless pooling them is scientifically intended.
- Units do not matter for z-thresholding, but must be consistent within a call.

## License

MIT
