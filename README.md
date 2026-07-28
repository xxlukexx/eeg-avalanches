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
5. Define avalanche size as active channel-bin pairs and duration as bins.

The default parameters reproduce the originating implementation:

- absolute z threshold: `2.5`
- bin width: one sample
- theoretical size exponent for kappa: `1.5`
- minimum avalanches for distribution summaries: `20`

Reported outputs include count, mean size, mean duration, size and duration
power-law exponents, kappa, branching ratio, and the underlying size and
duration distributions.

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
```

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

## Input expectations

- Data should already be cleaned, channel-aligned, and consistently referenced.
- All segments in one call must contain the same channels in the same order.
- Values must be finite; handle missing samples before analysis.
- Analyze conditions separately unless pooling them is scientifically intended.
- Units do not matter for z-thresholding, but must be consistent within a call.

## Sensitivity analysis

Avalanche estimates are highly dependent on threshold, temporal binning,
sampling rate, preprocessing, reference, channel count, and sensor geometry.
Pre-specify a primary configuration and repeat the analysis over a small,
declared grid of plausible thresholds and bin widths. Comparisons across
datasets are most interpretable when these choices and channel coverage are
harmonized.

The included exponent is a fixed-`xmin` continuous maximum-likelihood estimate.
The included kappa is a ten-point descriptive comparison against a reference
power-law distribution. Neither is a formal demonstration of a power law or
critical dynamics. For confirmatory criticality claims, add explicit model
comparison and goodness-of-fit procedures.

## Citation and provenance

If you use this software, cite the repository release and the neuronal-avalanche
methodological literature appropriate to your acquisition and analysis choices.
The implementation is deliberately dataset-agnostic and contains no participant
data or study-specific analysis.

## License

MIT

