"""Shared test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_openmc_ids():
    """Give every test a clean OpenMC id counter.

    The model assigns fixed cell ids, so building the geometry in several
    tests within one process re-uses them and OpenMC warns each time.  The
    warning is correct -- those really are duplicate ids -- but it is an
    artefact of the tests sharing a process, not of the model.
    """
    openmc = pytest.importorskip("openmc")
    openmc.reset_auto_ids()
    yield
