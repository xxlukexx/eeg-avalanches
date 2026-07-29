# Reproducible kappa methods

The methods paragraph is generated from the same JSON provenance saved by the
analysis. This freezes configurable parameters and implementation details that
would otherwise be easy to omit from a manuscript.

## Create JSON during analysis

The normal results JSON contains a `provenance` object:

```bash
eeg-avalanches clean_eeg.npy \
  --sampling-rate 200 \
  --threshold-z 2.5 \
  --bin-width-samples 1 \
  --theory-exponent 1.5 \
  --min-fit-events 50 \
  --kappa-points 10 \
  --kappa-xmin 1 \
  --summary-only \
  --output results.json
```

For percentile confidence intervals, add a prespecified iteration count and
seed:

```bash
eeg-avalanches clean_eeg.npy \
  --sampling-rate 200 \
  --bootstrap-iterations 500 \
  --random-seed 0 \
  --output results.json
```

For between-participant comparisons, also prespecify a common upper support for
the kappa reference CDF:

```bash
eeg-avalanches clean_eeg.npy \
  --sampling-rate 200 \
  --kappa-reference-max-size 500 \
  --output results.json
```

Choose this value before outcome analysis, for example from protocol constraints
or a pooled outcome-blind audit. The default observed maximum reproduces the
usual within-recording kappa convention but can vary with recording length,
channel count, and event rate.

The Python API provides the same record:

```python
result = analyze_avalanches(data, config)
result.write_json("results.json", include_distributions=False)
parameters = result.provenance
```

To also save the provenance by itself:

```bash
eeg-avalanches clean_eeg.npy \
  --sampling-rate 200 \
  --output results.json \
  --parameters-output frozen_parameters.json
```

## Render a new methods section

The renderer accepts either the full results JSON or the parameter-only JSON:

```bash
eeg-avalanches-methods results.json --output kappa_methods.md
```

The repository script provides the same operation:

```bash
python scripts/render_kappa_methods.py \
  frozen_parameters.json \
  --output kappa_methods.md
```

From Python:

```python
from eeg_avalanches.methods import load_provenance, render_kappa_methods

provenance = load_provenance("results.json")
methods_text = render_kappa_methods(provenance)
```

## Committed default

`kappa_default_parameters.json` freezes the package defaults for data sampled at
200 Hz. `kappa_default_methods.md` is the complete rendered paragraph. A test
checks that rendering the JSON reproduces the committed Markdown exactly.

Regenerate it after an intentional parameter or template change:

```bash
python scripts/render_kappa_methods.py \
  methods/kappa_default_parameters.json \
  --output methods/kappa_default_methods.md
```
