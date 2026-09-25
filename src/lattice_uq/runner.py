"""Execute one OpenMC transport calculation and reduce it to a record.

The rest of the pipeline never touches statepoint files directly.  This module
is the only place that knows how to turn a `DesignPoint` into a row of data,
which keeps the campaign driver and the surrogate independent of OpenMC.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import openmc

from .model import CELL_IDS, build_model
from .params import DesignPoint


@dataclass
class RunResult:
    """One completed transport calculation."""

    run_id: str
    point: DesignPoint
    keff: float
    keff_sigma: float
    particles: int
    batches: int
    inactive: int
    seed: int | None
    wall_time_s: float
    openmc_version: str
    entropy: list[float] = field(default_factory=list)
    tallies: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict:
        """Flat dict suitable for a dataframe row."""
        row = dict(self.point.as_dict())
        row.update(
            run_id=self.run_id,
            keff=self.keff,
            keff_sigma=self.keff_sigma,
            keff_sigma_pcm=self.keff_sigma * 1e5,
            particles=self.particles,
            batches=self.batches,
            inactive=self.inactive,
            seed=self.seed,
            wall_time_s=self.wall_time_s,
            openmc_version=self.openmc_version,
        )
        row.update({f"tally_{k}": v for k, v in self.tallies.items()
                    if np.isscalar(v)})
        return row


def _scalar_tallies(sp: openmc.StatePoint) -> dict[str, Any]:
    """Pull the handful of integral quantities used for physics checks.

    Reads the tally's mean array directly rather than going through
    `get_pandas_dataframe`, which is both slower and, on some pandas/pyarrow
    combinations, outright broken.  The flattened filter index runs with the
    last filter varying fastest, so with [energy, cell] filters the index is
    `energy_bin * n_cells + cell_bin`.
    """
    out: dict[str, Any] = {}
    try:
        tg = sp.get_tally(name="two_group")
    except (LookupError, KeyError):
        return out

    energy_filter = tg.find_filter(openmc.EnergyFilter)
    cell_filter = tg.find_filter(openmc.CellFilter)
    n_cells = len(cell_filter.bins)

    cell_bins = list(np.asarray(cell_filter.bins).ravel())
    fuel_bin = cell_bins.index(CELL_IDS["fuel"])
    mod_bin = cell_bins.index(CELL_IDS["moderator"])

    # Energy bins are ordered low-to-high, so bin 0 is the thermal group.
    n_energy = len(energy_filter.bins)
    scores = list(tg.scores)
    mean = tg.mean.reshape(n_energy * n_cells, -1, len(scores))

    def _value(score: str, energy_bin: int, cell_bin: int) -> float:
        return float(mean[energy_bin * n_cells + cell_bin, :, scores.index(score)].sum())

    flux_th = _value("flux", 0, fuel_bin)
    flux_fast = _value("flux", 1, fuel_bin)
    total = flux_th + flux_fast
    if total > 0:
        # Fraction of the in-fuel flux below 0.625 eV: the single number that
        # tracks how the spectrum hardens or softens across the design space,
        # and the one that explains most of the movement in k.
        out["thermal_flux_fraction"] = flux_th / total

    mod_flux = _value("flux", 0, mod_bin) + _value("flux", 1, mod_bin)
    if total > 0:
        # Thermal disadvantage factor: moderator flux over fuel flux in the
        # thermal group, which is how the pitch sweep shows up physically.
        out["thermal_disadvantage"] = (
            _value("flux", 0, mod_bin) / flux_th if flux_th > 0 else float("nan")
        )
        out["moderator_to_fuel_flux"] = mod_flux / total
    return out


def _spectrum(sp: openmc.StatePoint) -> dict[str, np.ndarray] | None:
    try:
        tally = sp.get_tally(name="spectrum")
    except LookupError:
        return None
    flux = tally.mean.ravel()
    err = tally.std_dev.ravel()
    edges = tally.filters[0].bins  # (n, 2) array of [low, high] in eV
    return {
        "energy_low": np.asarray(edges)[:, 0],
        "energy_high": np.asarray(edges)[:, 1],
        "flux": flux,
        "flux_std_dev": err,
    }


def run_point(
    point: DesignPoint,
    particles: int = 10_000,
    batches: int = 130,
    inactive: int = 30,
    seed: int | None = None,
    threads: int | None = None,
    with_tallies: bool = False,
    keep_dir: Path | None = None,
) -> tuple[RunResult, dict[str, np.ndarray] | None]:
    """Build, run and reduce a single design point.

    Runs in a scratch directory so that concurrent runs cannot collide on
    OpenMC's fixed input/output filenames.  Returns the reduced result and,
    when `with_tallies` is set, the energy spectrum arrays.
    """
    point.validate()
    model = build_model(
        point,
        particles=particles,
        batches=batches,
        inactive=inactive,
        seed=seed,
        with_tallies=with_tallies,
    )

    workdir = Path(keep_dir) if keep_dir else Path(tempfile.mkdtemp(prefix="omc_"))
    workdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        sp_path = model.run(cwd=str(workdir), threads=threads, output=False)
        wall = time.perf_counter() - t0

        # Reading a statepoint rebuilds its filters with the ids stored in the
        # file, which collide with the ones still live in this process and warn
        # once per run.  Harmless, but across a few hundred runs it buries the
        # campaign's own output.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", openmc.IDWarning)
            with openmc.StatePoint(sp_path) as sp:
                keff = float(sp.keff.nominal_value)
                sigma = float(sp.keff.std_dev)
                entropy = [float(x) for x in np.asarray(sp.entropy).ravel()] \
                    if sp.entropy is not None else []
                tallies = _scalar_tallies(sp) if with_tallies else {}
                spectrum = _spectrum(sp) if with_tallies else None
    finally:
        if keep_dir is None:
            shutil.rmtree(workdir, ignore_errors=True)

    result = RunResult(
        run_id=point.run_id,
        point=point,
        keff=keff,
        keff_sigma=sigma,
        particles=particles,
        batches=batches,
        inactive=inactive,
        seed=seed,
        wall_time_s=wall,
        openmc_version=".".join(str(v) for v in openmc.__version__)
        if isinstance(openmc.__version__, tuple) else str(openmc.__version__),
        entropy=entropy,
        tallies=tallies,
    )
    return result, spectrum


def check_cross_sections() -> str:
    """Fail loudly and early if the data library is not configured.

    Nothing in this project runs without it, and the error OpenMC raises on
    its own is several frames deep and easy to misread.
    """
    path = os.environ.get("OPENMC_CROSS_SECTIONS")
    if not path:
        raise RuntimeError(
            "OPENMC_CROSS_SECTIONS is not set. Point it at the cross_sections.xml "
            "of an ENDF/B HDF5 library (see scripts/fetch_data.sh)."
        )
    if not Path(path).is_file():
        raise RuntimeError(f"OPENMC_CROSS_SECTIONS={path} does not exist.")
    return path
