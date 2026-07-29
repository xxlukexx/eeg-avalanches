"""Neuronal avalanche metrics for cleaned EEG arrays."""

from .core import (
    AvalancheConfig,
    AvalancheResult,
    analyze_avalanches,
    detect_avalanches,
    threshold_events,
    validate_batch_compatibility,
)
from .estimators import (
    KappaResult,
    PowerLawFit,
    calculate_kappa,
    fit_discrete_power_law,
    kappa_against_theory,
    power_law_exponent,
)

__all__ = [
    "AvalancheConfig",
    "AvalancheResult",
    "KappaResult",
    "PowerLawFit",
    "analyze_avalanches",
    "calculate_kappa",
    "detect_avalanches",
    "fit_discrete_power_law",
    "kappa_against_theory",
    "power_law_exponent",
    "threshold_events",
    "validate_batch_compatibility",
]

__version__ = "0.4.0"
