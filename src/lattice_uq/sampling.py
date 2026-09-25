"""Design of experiments for the parameter sweep.

Three designs are generated, each with a distinct job:

* **train** -- a Latin hypercube over the four-dimensional box.  LHS is used
  rather than a full factorial grid because a 4-D grid fine enough to resolve
  curvature in each direction costs far more runs for the same coverage, and
  because every LHS point contributes information about every dimension.
* **test** -- an *independent* scrambled Sobol sequence.  Holding out a random
  subset of the LHS would leave the test points stratified by the same design
  that produced the training set, which flatters the held-out error.  A
  separately generated sequence does not.
* **replicate** -- a handful of design points repeated with different RNG
  seeds.  These are what make the central claim falsifiable: they measure the
  run-to-run scatter of the transport calculation directly, so the reported
  statistical uncertainty can be checked rather than trusted.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from .params import BOUNDS, PARAM_NAMES, DesignPoint

#: Fixed so the whole campaign is reproducible from the seed alone.
TRAIN_SEED = 20260101
TEST_SEED = 20260102
REPLICATE_SEED = 20260103


def _bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([BOUNDS[n].low for n in PARAM_NAMES], dtype=float)
    hi = np.array([BOUNDS[n].high for n in PARAM_NAMES], dtype=float)
    return lo, hi


def _scale(unit: np.ndarray) -> list[DesignPoint]:
    lo, hi = _bounds_arrays()
    scaled = qmc.scale(unit, lo, hi)
    return [DesignPoint.from_vector(row) for row in scaled]


def latin_hypercube(n: int, seed: int = TRAIN_SEED) -> list[DesignPoint]:
    """Space-filling LHS design.

    `optimization="random-cd"` minimises centred discrepancy, which removes the
    occasional badly clustered draw that plain LHS permits.
    """
    sampler = qmc.LatinHypercube(
        d=len(PARAM_NAMES), seed=seed, optimization="random-cd"
    )
    return _scale(sampler.random(n))


def sobol_design(n: int, seed: int = TEST_SEED) -> list[DesignPoint]:
    """Independent scrambled Sobol design, used for held-out evaluation.

    Sobol's balance guarantees hold for n a power of two, and scipy warns
    otherwise.  The warning is worth heeding: an intermediate n is still a low
    discrepancy point set but no longer a balanced one, so prefer 64, 128, 256.

    The sequence is *extensible* -- the first m points of a length-n design are
    exactly the length-m design under the same seed.  Together with the
    content-addressed store that means a held-out set can be grown later and
    only the genuinely new points get run.
    """
    sampler = qmc.Sobol(d=len(PARAM_NAMES), scramble=True, seed=seed)
    return _scale(sampler.random(n))


def replicate_design(
    n_points: int = 5, n_seeds: int = 8, seed: int = REPLICATE_SEED
) -> list[tuple[DesignPoint, int]]:
    """Design points repeated under independent RNG seeds.

    Returns (point, seed) pairs.  The points are spread over the box by a
    small LHS so the measured scatter is not read off a single corner.
    """
    points = latin_hypercube(n_points, seed=seed)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 2**31 - 1, size=n_seeds)
    return [(p, int(s)) for p in points for s in seeds]


def coverage_report(points: list[DesignPoint]) -> dict[str, dict[str, float]]:
    """Per-dimension min/max/mean of a design -- a sanity check on coverage."""
    arr = np.array([p.vector() for p in points])
    return {
        name: {
            "min": float(arr[:, i].min()),
            "max": float(arr[:, i].max()),
            "mean": float(arr[:, i].mean()),
            "bound_low": BOUNDS[name].low,
            "bound_high": BOUNDS[name].high,
        }
        for i, name in enumerate(PARAM_NAMES)
    }
