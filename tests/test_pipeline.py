"""Tests for everything that does not require running transport.

The expensive part of this project is the campaign, so the parts that surround
it -- the design space, the sampling, the store's resumability, and the
surrogate's metrics -- are tested against synthetic data instead.  A bug in
the store or in the pcm conversion is much cheaper to find here than after a
few hundred OpenMC runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from lattice_uq.params import BOUNDS, PARAM_NAMES, REFERENCE_POINT, DesignPoint
from lattice_uq.sampling import latin_hypercube, replicate_design, sobol_design
from lattice_uq.store import ResultStore, record_key
from lattice_uq.surrogate import (
    GPSurrogate,
    MLPSurrogate,
    coefficient,
    evaluate,
    normalise,
    reactivity,
    uncertainty_comparison,
)


# -- design space -----------------------------------------------------------

def test_run_id_is_stable_and_round_trips():
    p = DesignPoint(3.1, 900.0, 0.72, 1.3)
    assert p.run_id == DesignPoint(3.1, 900.0, 0.72, 1.3).run_id
    # A point recovered from a CSV-style float round-trip must map to the same
    # id, or the campaign would re-run work it has already done.
    recovered = DesignPoint.from_vector([float(f"{v:.9g}") for v in p.vector()])
    assert recovered.run_id == p.run_id


def test_run_id_differs_between_points():
    a = DesignPoint(3.1, 900.0, 0.72, 1.3)
    b = DesignPoint(3.1, 900.0, 0.72, 1.31)
    assert a.run_id != b.run_id


def test_validate_rejects_out_of_range():
    with pytest.raises(ValueError, match="enrichment"):
        DesignPoint(9.0, 900.0, 0.72, 1.3).validate()


def test_reference_point_is_inside_the_design_space():
    REFERENCE_POINT.validate()


# -- sampling ---------------------------------------------------------------

def test_lhs_is_inside_bounds_and_reproducible():
    a = latin_hypercube(40)
    b = latin_hypercube(40)
    assert [p.vector() for p in a] == [p.vector() for p in b]
    for p in a:
        p.validate()


def test_lhs_stratifies_every_dimension():
    n = 64
    pts = np.array([p.vector() for p in latin_hypercube(n)])
    u = normalise(pts)
    # One point per 1/n stratum per dimension is the defining property of LHS.
    for j in range(u.shape[1]):
        strata = np.floor(u[:, j] * n).astype(int)
        assert len(set(strata)) == n


def test_sobol_design_is_independent_of_the_lhs():
    lhs = {p.run_id for p in latin_hypercube(64)}
    sob = {p.run_id for p in sobol_design(64)}
    assert not (lhs & sob)


def test_replicates_share_points_but_not_keys():
    reps = replicate_design(n_points=3, n_seeds=4)
    assert len(reps) == 12
    assert len({p.run_id for p, _ in reps}) == 3
    assert len({record_key(p, s) for p, s in reps}) == 12


# -- store ------------------------------------------------------------------

def _fake_result(point: DesignPoint, keff: float, seed=None):
    from lattice_uq.runner import RunResult

    return RunResult(
        run_id=point.run_id, point=point, keff=keff, keff_sigma=2.0e-4,
        particles=10, batches=3, inactive=1, seed=seed, wall_time_s=0.1,
        openmc_version="test", entropy=[1.0, 1.1], tallies={"thermal_flux_fraction": 0.3},
    )


def test_store_round_trip_and_resume(tmp_path):
    store = ResultStore(tmp_path)
    pts = latin_hypercube(5)
    jobs = [(p, None) for p in pts]
    assert len(store.pending(jobs)) == 5

    store.write(_fake_result(pts[0], 1.23), tag="train")
    assert len(store.pending(jobs)) == 4

    df = store.to_frame()
    assert len(df) == 1
    assert df.keff.iloc[0] == pytest.approx(1.23)
    assert df.keff_sigma_pcm.iloc[0] == pytest.approx(20.0)
    assert df.tally_thermal_flux_fraction.iloc[0] == pytest.approx(0.3)


def test_store_keeps_replicates_separate(tmp_path):
    store = ResultStore(tmp_path)
    p = latin_hypercube(1)[0]
    store.write(_fake_result(p, 1.10, seed=1))
    store.write(_fake_result(p, 1.11, seed=2))
    assert len(store.to_frame()) == 2


# -- surrogate --------------------------------------------------------------

def _toy(n, rng, noise=0.0):
    """A smooth, monotone stand-in for k-infinity over the real bounds."""
    pts = np.array([p.vector() for p in latin_hypercube(n, seed=rng)])
    u = normalise(pts)
    y = (
        1.10
        + 0.09 * u[:, 0]                     # enrichment: up
        - 0.03 * np.sqrt(u[:, 1] + 0.1)      # temperature: down, saturating
        + 0.05 * u[:, 2]                     # moderator density: up
        - 0.04 * (u[:, 3] - 0.4) ** 2        # pitch: peaked
    )
    if noise:
        y = y + rng_normal(len(y), noise)
    return pts, y


def rng_normal(n, scale, seed=3):
    return np.random.default_rng(seed).normal(0.0, scale, n)


def test_metrics_are_reported_in_pcm():
    y = np.array([1.0, 1.0, 1.0])
    # A flat 100 pcm offset must come back as exactly 100 pcm RMSE.
    m = evaluate(y, y + 1e-3)
    assert m.rmse_pcm == pytest.approx(100.0)
    assert m.bias_pcm == pytest.approx(100.0)


def test_uncertainty_comparison_flags_the_claim_correctly():
    m = evaluate(np.array([1.0, 1.0]), np.array([1.0002, 1.0]))  # ~14 pcm RMSE
    big = uncertainty_comparison(m, np.full(2, 5e-4))   # 50 pcm sigma
    small = uncertainty_comparison(m, np.full(2, 1e-5))  # 1 pcm sigma
    assert big["below_mc_sigma"] is True
    assert small["below_mc_sigma"] is False


def test_gp_recovers_a_smooth_response_within_its_own_error_bars():
    x_tr, y_tr = _toy(120, 1)
    x_te, y_te = _toy(40, 2)
    gp = GPSurrogate().fit(x_tr, y_tr, np.full(len(y_tr), 2e-5))
    pred, std = gp.predict(x_te, return_std=True)
    m = evaluate(y_te, pred, std)
    assert m.rmse_pcm < 60.0
    # Calibration matters more than accuracy here: an over-confident GP would
    # make the headline claim untestable.
    assert 0.80 <= m.coverage_95 <= 1.0


def test_mlp_fits_the_same_surface():
    x_tr, y_tr = _toy(200, 1)
    x_te, y_te = _toy(40, 2)
    mlp = MLPSurrogate().fit(x_tr, y_tr)
    assert evaluate(y_te, mlp.predict(x_te)).rmse_pcm < 250.0


def test_reactivity_conversion():
    assert reactivity(1.0) == pytest.approx(0.0)
    assert reactivity(1.25) == pytest.approx(0.2)


def test_coefficient_signs_follow_the_fitted_surface():
    """The toy surface is built to rise with enrichment and fall with T."""
    x_tr, y_tr = _toy(200, 1)
    gp = GPSurrogate().fit(x_tr, y_tr, np.full(len(y_tr), 2e-5))
    centre = np.array([0.5 * (BOUNDS[n].low + BOUNDS[n].high) for n in PARAM_NAMES])

    temp = coefficient(gp, centre, "fuel_temperature")
    enr = coefficient(gp, centre, "enrichment")

    assert temp.value_pcm_per_unit < 0 and temp.consistent
    assert enr.value_pcm_per_unit > 0 and enr.consistent
    # Finite-differencing a GP is far better determined than the two
    # predictions separately; a huge error bar means the covariance term was
    # dropped somewhere -- or that the stencil is too narrow to see past the
    # noise, which is what a 2% step did to the real Doppler coefficient.
    assert temp.sigma_pcm_per_unit < abs(temp.value_pcm_per_unit)


def test_expected_signs_only_covers_unambiguous_parameters():
    """Pitch and moderator density must not carry a fixed sign.

    Both act through the moderator-to-fuel ratio, and the sampled box
    straddles the moderation optimum, so their coefficients change sign inside
    it. Asserting a sign for them would be asserting bad physics.
    """
    from lattice_uq.surrogate import EXPECTED_SIGNS

    assert set(EXPECTED_SIGNS) == {"fuel_temperature", "enrichment"}


def test_moderation_ratio_tracks_both_pitch_and_density():
    """The two moderation knobs are interchangeable at equal Vm/Vf."""
    from lattice_uq.surrogate import moderation_ratio

    # Halving the density halves the ratio.
    assert moderation_ratio(1.3, 0.4) == pytest.approx(
        0.5 * moderation_ratio(1.3, 0.8)
    )
    # Widening the lattice raises it.
    assert moderation_ratio(1.4, 0.8) > moderation_ratio(1.2, 0.8)
    # A nominal PWR pin cell is under-moderated: the optimum is nearer 1.5.
    assert 1.2 < moderation_ratio(1.25984, 0.740582) < 1.7


def test_moderation_optimum_finds_a_peak_from_both_directions():
    """A surface with a genuine moderation peak must be found consistently."""
    from lattice_uq.surrogate import moderation_optimum, moderation_ratio

    # Build a toy response that depends on pitch and density ONLY through the
    # moderation ratio, peaking at 1.6 -- exactly the structure the check is
    # meant to detect.
    pts = np.array([p.vector() for p in latin_hypercube(220, seed=5)])
    ratios = np.array([moderation_ratio(p[3], p[2]) for p in pts])
    y = 1.25 - 0.20 * (ratios - 1.6) ** 2 + 0.02 * (pts[:, 0] - 3.5)

    gp = GPSurrogate().fit(pts, y, np.full(len(y), 2e-5))
    centre = np.array([0.5 * (BOUNDS[n].low + BOUNDS[n].high) for n in PARAM_NAMES])
    out = moderation_optimum(gp, centre)

    assert out["agreement"]["both_peaks_interior"]
    assert out["agreement"]["consistent"]
    # And it should recover roughly the ratio the surface was built around.
    assert out["agreement"]["ratio_from_density_sweep"] == pytest.approx(1.6, abs=0.15)


def test_noise_deconvolution_separates_model_error_from_reference_noise():
    """RMSE measured against noisy references double-counts the noise.

    Build a case where the answer is known: a surrogate whose true error is
    exactly 60 pcm, scored against references carrying 80 pcm of noise, must
    measure sqrt(60^2 + 80^2) = 100 pcm and deconvolve back to 60.
    """
    rng = np.random.default_rng(0)
    n = 4000
    truth = np.full(n, 1.2)
    model_err = rng.normal(0.0, 60e-5, n)
    ref_noise = rng.normal(0.0, 80e-5, n)

    pred = truth + model_err
    reference = truth + ref_noise

    m = evaluate(reference, pred)
    comp = uncertainty_comparison(m, np.full(n, 80e-5))

    assert m.rmse_pcm == pytest.approx(100.0, rel=0.05)
    assert comp["deconvolved_rmse_pcm"] == pytest.approx(60.0, rel=0.12)
    # The measured value is above the reference noise, the true error below it.
    assert comp["below_mc_sigma"] is False
    assert comp["deconvolved_below_mc_sigma"] is True


def test_deconvolution_clamps_at_zero_rather_than_going_imaginary():
    """A surrogate already at the noise floor must not produce a NaN."""
    y = np.full(200, 1.2)
    m = evaluate(y, y + 1e-6)          # ~0.1 pcm error
    comp = uncertainty_comparison(m, np.full(200, 1e-3))  # 100 pcm reference noise
    assert comp["deconvolved_rmse_pcm"] == 0.0
    assert comp["deconvolved_below_mc_sigma"] is True
