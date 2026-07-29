# Changelog

## 0.4.0

This is a breaking scientific-method release. Primary outputs from earlier
versions should not be mixed with 0.4 outputs.

- Replaced the continuous power-law MLE applied to integer counts with an exact
  discrete Hurwitz-zeta MLE. `xmin` is selected by minimum KS distance unless
  fixed explicitly, and the default minimum tail count is 50.
- Replaced the kappa point-mass approximation with a truncated discrete
  power-law CDF evaluated over the complete integer support.
- Corrected the kappa subtraction direction to
  `1 + mean(reference_cdf - empirical_cdf)`.
- Replaced mean nonterminal pairwise branching ratios with a
  terminal-inclusive ratio of sums. Duration-one avalanches now contribute.
- Discard avalanches touching segment boundaries by default because they are
  potentially censored.
- Raise on flat channels instead of silently mapping their standard deviation
  to one.
- Added optional channel-name provenance and batch compatibility validation.
- Added optional seeded avalanche-level bootstrap confidence intervals.
- Added deterministic simulations that recover known generating parameters.
- Retained the former exponent, kappa, and branching calculations in
  `eeg_avalanches.legacy` for reproduction only.

## 0.3.0

- Added machine-readable provenance and generated kappa methods text.

## 0.2.0

- Added the optional LEAP EEGLAB adapter.

## 0.1.0

- Initial reusable avalanche extraction package.

