"""Parameter definitions for the pin-cell campaign.

A single place that defines what a "design point" is, what its physical
bounds are, and how it is hashed into a stable run identifier.  Every other
module (sampling, campaign, surrogate) imports from here so that the
parameter space is never redefined in two places.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Bound:
    """Inclusive sampling bound for one design variable."""

    low: float
    high: float
    units: str
    rationale: str

    def clip(self, x: float) -> float:
        return min(max(x, self.low), self.high)


#: The design space swept in Phase 2.  Ranges are chosen to stay inside the
#: temperature grid of the ENDF/B HDF5 libraries (250/294/600/900/1200/2500 K)
#: so that the "interpolation" temperature treatment never extrapolates.
BOUNDS: Mapping[str, Bound] = {
    "enrichment": Bound(
        2.0, 5.0, "wt% U235",
        "Primary reactivity control; 5 wt% is the LEU licensing limit.",
    ),
    "fuel_temperature": Bound(
        600.0, 1200.0, "K",
        "Doppler broadening of U238 capture resonances; spans hot-zero-power "
        "to hot-full-power average fuel temperature.",
    ),
    "moderator_density": Bound(
        0.60, 1.00, "g/cm3",
        "Coolant voiding / boiling feedback; 0.74 g/cm3 is nominal PWR hot "
        "full power, 1.0 g/cm3 is cold.",
    ),
    "pitch": Bound(
        1.20, 1.40, "cm",
        "Moderator-to-fuel ratio; brackets the nominal 1.26 cm PWR lattice on "
        "both the under- and over-moderated side.",
    ),
}

#: Held fixed across the campaign.  Documented here so the README and the
#: model build never disagree about what was held constant.
FIXED = {
    "boron_ppm": 975.0,          # soluble boron, beginning-of-cycle PWR
    "fuel_radius": 0.39218,      # cm
    "clad_inner_radius": 0.40005,
    "clad_outer_radius": 0.45720,
    "fuel_density": 10.29769,    # g/cm3, ~94% theoretical
    "clad_temperature": 600.0,   # K
    "moderator_temperature": 600.0,  # K
}

PARAM_NAMES = tuple(BOUNDS.keys())


@dataclass(frozen=True)
class DesignPoint:
    """One point in the four-dimensional design space."""

    enrichment: float
    fuel_temperature: float
    moderator_density: float
    pitch: float

    def as_dict(self) -> dict:
        return asdict(self)

    def vector(self) -> tuple:
        return tuple(getattr(self, name) for name in PARAM_NAMES)

    @classmethod
    def from_vector(cls, values: Iterable[float]) -> "DesignPoint":
        return cls(**dict(zip(PARAM_NAMES, (float(v) for v in values))))

    @property
    def run_id(self) -> str:
        """Stable, content-addressed id.

        Rounding before hashing means a design point recovered from a CSV
        round-trips to the same id as the one that generated it, which is what
        makes the campaign resumable.
        """
        payload = {k: round(float(v), 9) for k, v in self.as_dict().items()}
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    def validate(self) -> None:
        for f in fields(self):
            bound = BOUNDS[f.name]
            value = getattr(self, f.name)
            if not (bound.low - 1e-9 <= value <= bound.high + 1e-9):
                raise ValueError(
                    f"{f.name}={value} outside sampled range "
                    f"[{bound.low}, {bound.high}] {bound.units}"
                )


#: Reference point used for verification runs (Phase 1).  Matches the pin cell
#: distributed with OpenMC's own examples, which is derived from BEAVRS.
REFERENCE_POINT = DesignPoint(
    enrichment=2.4,
    fuel_temperature=600.0,
    moderator_density=0.740582,
    pitch=1.25984,
)
