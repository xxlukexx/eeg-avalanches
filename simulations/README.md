# Estimator validation simulations

`validate_estimators.py` runs three deterministic checks:

1. Samples from a discrete power law with exponent 1.5 are fitted using the
   exact discrete likelihood. The corrected estimate must be within 0.03 of the
   generating value, while the legacy continuous estimator demonstrates its
   expected upward bias.
2. Samples from the same truncated discrete distribution used by kappa must
   return kappa within 0.01 of 1.
3. A critical Poisson branching process with true branching ratio 1 is measured
   using terminal-inclusive ratio-of-sums and the legacy mean of nonterminal
   ratios. The corrected estimator must be close to 1 and the legacy estimator
   must exhibit its expected upward bias.

Run from the repository root:

```bash
python simulations/validate_estimators.py
```

The script uses a fixed random seed and exits with an error if any recovery
criterion is not met. Smaller deterministic versions of the power-law and
kappa checks are also part of the unit-test suite.

`validation_results.json` contains the output generated for release `0.4.0`.
