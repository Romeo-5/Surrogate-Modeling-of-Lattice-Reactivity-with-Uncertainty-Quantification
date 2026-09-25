"""Tests for the OpenMC model build.

These construct the model but never run transport, so they are fast.  They
cover the modelling mistakes that are silent -- a missing S(alpha,beta) table,
a temperature that never reached the material, boron that does not scale with
water density -- each of which changes k-infinity by hundreds to thousands of
pcm while the run still completes happily.
"""

from __future__ import annotations

import numpy as np
import pytest

openmc = pytest.importorskip("openmc")

from lattice_uq.model import (  # noqa: E402
    build_geometry,
    build_materials,
    build_model,
    build_settings,
    energy_grid,
)
from lattice_uq.params import FIXED, DesignPoint


POINT = DesignPoint(enrichment=3.5, fuel_temperature=900.0,
                    moderator_density=0.72, pitch=1.3)


def test_thermal_scattering_is_applied_to_the_moderator():
    _, _, _, water = build_materials(POINT)
    # Read it off the serialised material rather than a private attribute:
    # the XML is what OpenMC actually consumes.
    sab = [e.get("name") for e in water.to_xml_element().findall("sab")]
    assert any("H_in_H2O" in str(n) for n in sab), (
        "bound-hydrogen scattering missing; free-gas thermal scattering "
        "shifts k-infinity by thousands of pcm"
    )


def test_fuel_temperature_reaches_the_fuel():
    fuel, _, clad, water = build_materials(POINT)
    assert fuel.temperature == pytest.approx(900.0)
    # Only the fuel is swept; clad and moderator stay at their fixed values.
    assert clad.temperature == pytest.approx(FIXED["clad_temperature"])
    assert water.temperature == pytest.approx(FIXED["moderator_temperature"])


def test_enrichment_changes_the_u235_fraction():
    low = build_materials(POINT.__class__(2.0, 900.0, 0.72, 1.3))[0]
    high = build_materials(POINT.__class__(5.0, 900.0, 0.72, 1.3))[0]

    def u235_fraction(mat):
        densities = dict(mat.get_nuclide_atom_densities())
        total = sum(v for k, v in densities.items() if k.startswith("U"))
        return densities.get("U235", 0.0) / total

    assert u235_fraction(high) > u235_fraction(low)
    assert u235_fraction(low) == pytest.approx(0.02, rel=0.05)
    assert u235_fraction(high) == pytest.approx(0.05, rel=0.05)


def test_boron_scales_with_moderator_density():
    """Fixed ppm by weight means boron atoms per cm3 track the water density."""
    dense = build_materials(POINT.__class__(3.5, 900.0, 1.00, 1.3))[3]
    light = build_materials(POINT.__class__(3.5, 900.0, 0.60, 1.3))[3]
    b_dense = sum(v for k, v in dense.get_nuclide_atom_densities().items()
                  if k.startswith("B1"))
    b_light = sum(v for k, v in light.get_nuclide_atom_densities().items()
                  if k.startswith("B1"))
    assert b_dense / b_light == pytest.approx(1.00 / 0.60, rel=1e-3)


def test_geometry_is_reflective_on_all_faces():
    materials = build_materials(POINT)
    geometry = build_geometry(POINT, materials)
    surfaces = geometry.get_all_surfaces().values()
    bounded = [s for s in surfaces if s.boundary_type != "transmission"]
    assert len(bounded) == 6
    assert all(s.boundary_type == "reflective" for s in bounded), (
        "a non-reflective face would make this a leaky cell, not an "
        "infinite lattice, and the eigenvalue would not be k-infinity"
    )


def test_pitch_sets_the_cell_width():
    materials = build_materials(POINT)
    geometry = build_geometry(POINT, materials)
    xs = sorted(s.x0 for s in geometry.get_all_surfaces().values()
                if isinstance(s, openmc.XPlane))
    assert xs[1] - xs[0] == pytest.approx(POINT.pitch)


def test_fuel_radius_is_inside_the_clad():
    assert FIXED["fuel_radius"] < FIXED["clad_inner_radius"]
    assert FIXED["clad_inner_radius"] < FIXED["clad_outer_radius"]
    # The pin must fit inside the smallest pitch that will ever be sampled.
    from lattice_uq.params import BOUNDS
    assert FIXED["clad_outer_radius"] < BOUNDS["pitch"].low / 2.0


def test_all_cells_are_clipped_to_the_axial_slab():
    """Every cell must be bounded in z, or particles leak out of the model.

    The radial surfaces are infinite cylinders, so a cell defined only by them
    extends past the reflective z-planes; a particle that reaches there has no
    boundary to reflect off and OpenMC loses it.
    """
    materials = build_materials(POINT)
    geometry = build_geometry(POINT, materials)
    half = POINT.pitch / 2.0
    for cell in geometry.get_all_cells().values():
        (_, _, zlo), (_, _, zhi) = cell.region.bounding_box
        assert zlo == pytest.approx(-half), f"{cell.name} unbounded below"
        assert zhi == pytest.approx(half), f"{cell.name} unbounded above"


def test_settings_use_temperature_interpolation():
    s = build_settings(POINT.pitch)
    assert s.temperature["method"] == "interpolation"
    assert s.inactive > 0 and s.batches > s.inactive
    assert s.entropy_mesh is not None


def test_energy_grid_spans_thermal_to_fast():
    grid = energy_grid(200)
    assert len(grid) == 201
    assert grid[0] == pytest.approx(1e-5)
    assert grid[-1] == pytest.approx(2.0e7)
    # Uniform in lethargy, which is what makes the spectrum plot readable.
    lethargy = np.diff(np.log(grid))
    assert np.allclose(lethargy, lethargy[0])


def test_model_assembles_with_tallies():
    model = build_model(POINT, particles=100, batches=5, inactive=2,
                        with_tallies=True)
    names = {t.name for t in model.tallies}
    assert {"spectrum", "reaction_rates", "two_group"} <= names
