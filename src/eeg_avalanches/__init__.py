"""Neuronal avalanche metrics for cleaned EEG arrays."""

from .core import (
    AvalancheConfig,
    AvalancheResult,
    analyze_avalanches,
    detect_avalanches,
    kappa_against_theory,
    power_law_exponent,
    threshold_events,
)

__all__ = [
    "AvalancheConfig",
    "AvalancheResult",
    "analyze_avalanches",
    "detect_avalanches",
    "kappa_against_theory",
    "power_law_exponent",
    "threshold_events",
]

__version__ = "0.2.0"
