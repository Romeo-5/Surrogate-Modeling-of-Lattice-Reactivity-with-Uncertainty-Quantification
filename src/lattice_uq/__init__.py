"""Surrogate modelling of PWR pin-cell reactivity with uncertainty quantification.

Layering, from the bottom up:

    params     what a design point is, and the bounds of the design space
    model      DesignPoint -> openmc.Model
    runner     openmc.Model -> RunResult (the only module that imports openmc
               for execution; everything above it is OpenMC-free)
    store      RunResult -> disk, and back as a dataframe
    campaign   parallel, resumable execution of many design points
    surrogate  dataframe -> fitted surrogate, metrics, reactivity coefficients
    plotting   figures for the report
"""

__version__ = "0.1.0"

from .params import BOUNDS, FIXED, PARAM_NAMES, REFERENCE_POINT, DesignPoint

__all__ = [
    "BOUNDS",
    "FIXED",
    "PARAM_NAMES",
    "REFERENCE_POINT",
    "DesignPoint",
    "__version__",
]
