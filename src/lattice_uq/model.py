"""Parametric PWR pin-cell model.

Geometry and nominal composition follow the pin cell distributed with OpenMC's
own examples (itself derived from the BEAVRS benchmark specification): UO2
fuel, helium gap, Zircaloy-4 cladding, borated light water, square pitch with
reflective boundaries on all six faces, so the eigenvalue computed is
k-infinity for an infinite lattice of identical pins.

Everything that varies during the campaign enters through `DesignPoint`;
everything held fixed lives in `params.FIXED`.
"""

from __future__ import annotations

import numpy as np
import openmc

from .params import FIXED, DesignPoint

# Mass fractions of light water, from standard atomic weights.
_M_H, _M_O = 1.00794, 15.9994
_WF_H = 2 * _M_H / (2 * _M_H + _M_O)
_WF_O = 1.0 - _WF_H

#: Fixed cell ids so tally reduction can name regions instead of guessing.
CELL_IDS = {"fuel": 1, "gap": 2, "clad": 3, "moderator": 4}

#: Thermal / epithermal cutoff (eV), the conventional PWR group boundary.
THERMAL_CUTOFF = 0.625


def _source_class():
    """openmc.IndependentSource on >=0.14, openmc.Source before that."""
    return getattr(openmc, "IndependentSource", None) or openmc.Source


def build_materials(point: DesignPoint) -> openmc.Materials:
    """UO2 / He / Zircaloy-4 / borated water at the requested state."""
    fuel = openmc.Material(name="UO2 fuel")
    fuel.set_density("g/cm3", FIXED["fuel_density"])
    fuel.add_element("U", 1.0, enrichment=point.enrichment)
    fuel.add_element("O", 2.0)
    fuel.temperature = point.fuel_temperature

    helium = openmc.Material(name="Helium gap")
    helium.set_density("g/cm3", 0.001598)
    helium.add_element("He", 1.0)
    helium.temperature = point.fuel_temperature

    clad = openmc.Material(name="Zircaloy-4")
    clad.set_density("g/cm3", 6.55)
    clad.add_element("Zr", 0.98335, "wo")
    clad.add_element("Sn", 0.01400, "wo")
    clad.add_element("Fe", 0.00165, "wo")
    clad.add_element("Cr", 0.00100, "wo")
    clad.temperature = FIXED["clad_temperature"]

    # Soluble boron is specified in ppm by weight of the solution, so the
    # water fractions are scaled down by the boron weight fraction.  Defining
    # the material this way means that when the density is varied the boron
    # number density scales with it, which is the physical behaviour of a
    # fixed-ppm solution that heats up and expands.
    w_b = FIXED["boron_ppm"] * 1e-6
    water = openmc.Material(name="Borated water")
    water.set_density("g/cm3", point.moderator_density)
    water.add_element("H", _WF_H * (1.0 - w_b), "wo")
    water.add_element("O", _WF_O * (1.0 - w_b), "wo")
    water.add_element("B", w_b, "wo")
    # Thermal scattering for bound hydrogen.  Omitting this leaves free-gas
    # scattering in the thermal range and shifts k-infinity by thousands of
    # pcm -- it is not an optional refinement.
    water.add_s_alpha_beta("c_H_in_H2O")
    water.temperature = FIXED["moderator_temperature"]

    return openmc.Materials([fuel, helium, clad, water])


def build_geometry(point: DesignPoint, materials: openmc.Materials) -> openmc.Geometry:
    """Square pin cell with reflective boundaries (infinite lattice)."""
    fuel, helium, clad, water = materials

    fuel_or = openmc.ZCylinder(r=FIXED["fuel_radius"])
    clad_ir = openmc.ZCylinder(r=FIXED["clad_inner_radius"])
    clad_or = openmc.ZCylinder(r=FIXED["clad_outer_radius"])

    # Explicit planes rather than openmc.model.RectangularPrism: the helper's
    # name and call signature changed across OpenMC releases, these have not.
    half = point.pitch / 2.0
    left = openmc.XPlane(-half, boundary_type="reflective")
    right = openmc.XPlane(+half, boundary_type="reflective")
    bottom = openmc.YPlane(-half, boundary_type="reflective")
    top = openmc.YPlane(+half, boundary_type="reflective")
    # Axially infinite: reflective top and bottom one pitch apart.
    down = openmc.ZPlane(-half, boundary_type="reflective")
    up = openmc.ZPlane(+half, boundary_type="reflective")
    slab = +down & -up
    box = +left & -right & +bottom & -top & slab

    # Every cell is clipped to the axial slab.  The cylinders are infinite
    # surfaces: without this the fuel, gap and clad cells would extend past the
    # reflective z-planes, and a particle that got there would belong to a cell
    # with no boundary to reflect off -- OpenMC reports it as a lost particle.
    # Explicit cell ids rather than auto-assigned ones: tally reduction has to
    # tell the fuel apart from the moderator, and relying on creation order to
    # infer which auto-id is which is the kind of assumption that breaks
    # silently and mislabels every reaction rate.
    fuel_cell = openmc.Cell(cell_id=CELL_IDS["fuel"], name="fuel", fill=fuel,
                            region=-fuel_or & slab)
    gap_cell = openmc.Cell(cell_id=CELL_IDS["gap"], name="gap", fill=helium,
                           region=+fuel_or & -clad_ir & slab)
    clad_cell = openmc.Cell(cell_id=CELL_IDS["clad"], name="clad", fill=clad,
                            region=+clad_ir & -clad_or & slab)
    water_cell = openmc.Cell(cell_id=CELL_IDS["moderator"], name="moderator",
                             fill=water, region=+clad_or & box)

    root = openmc.Universe(cells=[fuel_cell, gap_cell, clad_cell, water_cell])
    return openmc.Geometry(root)


def energy_grid(n_groups: int = 500) -> np.ndarray:
    """Log-spaced energy grid in eV, thermal to fast, for spectrum tallies."""
    return np.logspace(-5, np.log10(2.0e7), n_groups + 1)


def build_tallies(geometry: openmc.Geometry, n_groups: int = 500) -> openmc.Tallies:
    """Energy spectrum in the cell plus the reaction rates used for checks."""
    cells = {c.name: c for c in geometry.get_all_cells().values()}

    spectrum = openmc.Tally(name="spectrum")
    spectrum.filters = [openmc.EnergyFilter(energy_grid(n_groups))]
    spectrum.scores = ["flux"]

    rates = openmc.Tally(name="reaction_rates")
    rates.filters = [openmc.CellFilter([cells["fuel"], cells["moderator"]])]
    rates.scores = ["fission", "absorption", "nu-fission", "flux"]

    # Thermal vs fast split at the usual 0.625 eV cutoff -- enough to explain
    # *why* k moves when a parameter moves, not just that it did.
    two_group = openmc.Tally(name="two_group")
    two_group.filters = [
        openmc.EnergyFilter([1e-5, THERMAL_CUTOFF, 2.0e7]),
        openmc.CellFilter([cells["fuel"], cells["moderator"]]),
    ]
    two_group.scores = ["flux", "absorption", "nu-fission"]

    return openmc.Tallies([spectrum, rates, two_group])


def build_settings(
    pitch: float,
    particles: int = 10_000,
    batches: int = 130,
    inactive: int = 30,
    seed: int | None = None,
    entropy: bool = True,
) -> openmc.Settings:
    """Eigenvalue settings.

    `inactive` batches are discarded because the fission source starts from a
    guessed distribution and needs to converge to the fundamental mode before
    the eigenvalue estimate is unbiased.  Shannon entropy of the fission
    source is tallied so that convergence can be demonstrated, not assumed.
    """
    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = int(particles)
    settings.batches = int(batches)
    settings.inactive = int(inactive)
    settings.output = {"tallies": False, "summary": False}

    # Uniform starting source over the fuel pin, clipped to the axial slab.
    half_z = pitch / 2.0
    r = FIXED["fuel_radius"]
    space = openmc.stats.Box((-r, -r, -half_z), (r, r, half_z))
    settings.source = _source_class()(space=space)

    if entropy:
        # The mesh has to cover the geometry it is measuring, so its axial
        # extent follows the cell rather than a hardcoded half-height.
        mesh = openmc.RegularMesh()
        mesh.dimension = [4, 4, 1]
        mesh.lower_left = [-r, -r, -half_z]
        mesh.upper_right = [r, r, half_z]
        settings.entropy_mesh = mesh

    # Cross sections are only tabulated at discrete temperatures; interpolate
    # between them so fuel temperature is a genuinely continuous input rather
    # than snapping to the nearest library point.
    settings.temperature = {"method": "interpolation", "range": (250.0, 2500.0)}

    if seed is not None:
        settings.seed = int(seed)
    return settings


def build_model(
    point: DesignPoint,
    particles: int = 10_000,
    batches: int = 130,
    inactive: int = 30,
    seed: int | None = None,
    with_tallies: bool = False,
    n_groups: int = 500,
) -> openmc.Model:
    """Assemble the full model for one design point."""
    point.validate()
    # Campaign workers build many models in one process; without this, surface
    # and filter ids climb forever and OpenMC warns on every reuse.
    openmc.reset_auto_ids()
    materials = build_materials(point)
    geometry = build_geometry(point, materials)
    settings = build_settings(point.pitch, particles, batches, inactive, seed)
    tallies = build_tallies(geometry, n_groups) if with_tallies else openmc.Tallies()
    return openmc.Model(
        geometry=geometry, materials=materials, settings=settings, tallies=tallies
    )
