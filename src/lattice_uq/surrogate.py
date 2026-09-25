"""Surrogate models for k-infinity, and the uncertainty comparison.

Two surrogates are fit and compared:

* a **Gaussian process**, which is the right default here because it returns a
  predictive variance at every point and because it can be told the noise it
  is fitting through.  Each training label carries a known Monte Carlo
  standard deviation, so `alpha` is set per-point to sigma_i^2 rather than to
  a single fitted noise level -- the GP is then solving the correct
  heteroscedastic regression instead of pretending every run is equally sharp.
* a small **multilayer perceptron**, as a check that the GP's advantage is
  real and not an artefact of the metric.

The headline result is `uncertainty_comparison`: held-out surrogate error
against the stochastic uncertainty of the transport runs themselves.  If the
former is below the latter, the surrogate is as accurate as the calculation it
replaces, and the remaining error is dominated by Monte Carlo noise in the
training labels rather than by the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .params import BOUNDS, PARAM_NAMES

PCM = 1e5  # k-units -> per cent mille


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------

def design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Parameter columns in the canonical order defined by `params`."""
    missing = [c for c in PARAM_NAMES if c not in df.columns]
    if missing:
        raise KeyError(f"results frame is missing parameter columns: {missing}")
    return df[list(PARAM_NAMES)].to_numpy(dtype=float)


def normalise(x: np.ndarray) -> np.ndarray:
    """Map the physical box onto the unit cube.

    Length scales are then comparable across dimensions that differ by three
    orders of magnitude (1.2 cm pitch vs 1200 K), which is what lets a single
    anisotropic kernel be initialised sensibly.
    """
    lo = np.array([BOUNDS[n].low for n in PARAM_NAMES])
    hi = np.array([BOUNDS[n].high for n in PARAM_NAMES])
    return (x - lo) / (hi - lo)


def denormalise(u: np.ndarray) -> np.ndarray:
    lo = np.array([BOUNDS[n].low for n in PARAM_NAMES])
    hi = np.array([BOUNDS[n].high for n in PARAM_NAMES])
    return lo + u * (hi - lo)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

@dataclass
class GPSurrogate:
    """Gaussian process on k-infinity with known per-point observation noise."""

    model: GaussianProcessRegressor | None = None
    y_mean: float = 0.0
    trained_on: int = 0
    noise_floor_pcm: float = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray, y_sigma: np.ndarray | None = None):
        u = normalise(x)
        self.y_mean = float(np.mean(y))
        resid = y - self.y_mean

        if y_sigma is None:
            alpha = 1e-10
        else:
            # A hard floor keeps a freak run with a tiny reported sigma from
            # being treated as an exact constraint and destabilising the fit.
            floor = self.noise_floor_pcm / PCM
            alpha = np.maximum(np.asarray(y_sigma, dtype=float), floor) ** 2

        kernel = (
            ConstantKernel(1.0, (1e-4, 1e4))
            # nu=2.5 -> twice-differentiable sample paths.  k-infinity is
            # smooth in these parameters but not analytic-smooth; the RBF limit
            # over-smooths the Doppler direction.  One length scale per
            # dimension, so the fit can decide which parameters matter -- and
            # the upper bound sits far above the unit cube's diameter so a
            # parameter the response barely depends on can run its length scale
            # off to "irrelevant" instead of piling up against a bound.
            * Matern(
                length_scale=np.ones(len(PARAM_NAMES)),
                length_scale_bounds=(1e-2, 1e3),
                nu=2.5,
            )
            # Absorbs any residual noise the reported MC sigma does not explain.
            + WhiteKernel(1e-8, (1e-14, 1e-4))
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=False,
            n_restarts_optimizer=8,
            random_state=0,
        )
        self.model.fit(u, resid)
        self.trained_on = len(y)
        return self

    def predict(self, x: np.ndarray, return_std: bool = False):
        u = normalise(x)
        if return_std:
            mean, std = self.model.predict(u, return_std=True)
            return mean + self.y_mean, std
        return self.model.predict(u) + self.y_mean

    def predict_cov(self, x: np.ndarray):
        mean, cov = self.model.predict(normalise(x), return_cov=True)
        return mean + self.y_mean, cov

    @property
    def kernel_summary(self) -> dict:
        k = self.model.kernel_
        lengths = dict(
            zip(PARAM_NAMES, np.atleast_1d(k.k1.k2.length_scale).astype(float))
        )
        return {
            "kernel": str(k),
            "length_scales_unit_cube": lengths,
            "log_marginal_likelihood": float(
                self.model.log_marginal_likelihood_value_
            ),
        }


@dataclass
class MLPSurrogate:
    """Small dense network, scaled inputs and centred target."""

    model: Pipeline | None = None
    y_mean: float = 0.0
    y_scale: float = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray, y_sigma: np.ndarray | None = None):
        # y_sigma is accepted for interface parity; a plain MLP has no way to
        # use per-point noise, which is part of what the comparison shows.
        self.y_mean = float(np.mean(y))
        self.y_scale = float(np.std(y)) or 1.0
        self.model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 64),
                        activation="tanh",  # smooth, matching a smooth response
                        solver="lbfgs",     # few hundred points: full-batch wins
                        alpha=1e-4,
                        max_iter=5000,
                        tol=1e-9,
                        random_state=0,
                    ),
                ),
            ]
        )
        self.model.fit(normalise(x), (y - self.y_mean) / self.y_scale)
        return self

    def predict(self, x: np.ndarray, return_std: bool = False):
        pred = self.model.predict(normalise(x)) * self.y_scale + self.y_mean
        if return_std:
            return pred, np.full(len(pred), np.nan)
        return pred


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    n: int
    rmse_pcm: float
    mae_pcm: float
    max_abs_pcm: float
    bias_pcm: float
    r2: float
    coverage_95: float | None = None
    mean_pred_sigma_pcm: float | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_std: np.ndarray | None = None,
    y_true_sigma: np.ndarray | None = None,
) -> Metrics:
    """Held-out accuracy, expressed in pcm.

    Coverage combines the surrogate's predictive sigma with the reference
    run's own MC sigma: the interval has to cover a *noisy* observation, so
    ignoring the label noise would make a well-calibrated GP look
    over-confident.
    """
    err = np.asarray(y_pred) - np.asarray(y_true)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    coverage = None
    mean_sigma = None
    if y_pred_std is not None and np.all(np.isfinite(y_pred_std)):
        total = np.asarray(y_pred_std, dtype=float) ** 2
        if y_true_sigma is not None:
            total = total + np.asarray(y_true_sigma, dtype=float) ** 2
        total = np.sqrt(total)
        coverage = float(np.mean(np.abs(err) <= 1.96 * total))
        mean_sigma = float(np.mean(y_pred_std) * PCM)

    return Metrics(
        n=len(err),
        rmse_pcm=float(np.sqrt(np.mean(err**2)) * PCM),
        mae_pcm=float(np.mean(np.abs(err)) * PCM),
        max_abs_pcm=float(np.max(np.abs(err)) * PCM),
        bias_pcm=float(np.mean(err) * PCM),
        r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        coverage_95=coverage,
        mean_pred_sigma_pcm=mean_sigma,
    )


def uncertainty_comparison(
    metrics: Metrics, mc_sigma: np.ndarray, train_sigma: np.ndarray | None = None
) -> dict:
    """The headline claim, stated so it can be falsified.

    `rms_mc_sigma_pcm` is the RMS reported sigma of the *held-out* runs.  It is
    a genuine floor: the reference values themselves are noisy at that level,
    so even a model that reproduced the true response surface exactly would
    still score a non-zero RMSE against them.  That is the number the headline
    claim is measured against.

    `noise_upper_bound_pcm` adds the training-label noise in quadrature.  It is
    an *upper* bound on the noise contribution, not a second floor: a surrogate
    that pools many neighbouring points averages most of the training noise
    away, so the real contribution is well below this.  It is reported to bound
    how much of the residual could possibly be noise rather than model error.
    """
    mc_sigma = np.asarray(mc_sigma, dtype=float)
    mean_mc = float(np.mean(mc_sigma) * PCM)
    rms_mc = float(np.sqrt(np.mean(mc_sigma**2)) * PCM)

    train_term = 0.0
    if train_sigma is not None and len(train_sigma):
        train_term = float(np.mean(np.asarray(train_sigma, dtype=float) ** 2)) * PCM**2
    upper = float(np.sqrt(rms_mc**2 + train_term))

    # Noise deconvolution.  The measured RMSE is against *noisy* references, and
    # the surrogate's error and the reference's Monte Carlo noise are
    # independent, so they add in quadrature:
    #
    #     E[(pred - reference)^2] = E[(pred - truth)^2] + sigma_ref^2
    #
    # Subtracting recovers the error against the true response surface, which
    # is the quantity anyone using the surrogate actually cares about.  Both
    # numbers get reported: the measured one is what was observed, the
    # deconvolved one is what it means.
    deconvolved = float(np.sqrt(max(metrics.rmse_pcm**2 - rms_mc**2, 0.0)))
    at_or_below_floor = metrics.rmse_pcm <= rms_mc

    return {
        "surrogate_rmse_pcm": metrics.rmse_pcm,
        "mean_mc_sigma_pcm": mean_mc,
        "rms_mc_sigma_pcm": rms_mc,
        "noise_upper_bound_pcm": upper,
        "deconvolved_rmse_pcm": deconvolved,
        "deconvolution_note": (
            "error against the true response surface, with the reference "
            "runs' own Monte Carlo noise removed in quadrature"
            + ("; measured RMSE is already at or below the reference noise, "
               "so this is an upper bound of zero information"
               if at_or_below_floor else "")
        ),
        "ratio_rmse_to_mc_sigma": metrics.rmse_pcm / rms_mc if rms_mc else float("nan"),
        "ratio_deconvolved_to_mc_sigma": (
            deconvolved / rms_mc if rms_mc else float("nan")
        ),
        "below_mc_sigma": bool(metrics.rmse_pcm < rms_mc),
        "deconvolved_below_mc_sigma": bool(deconvolved < rms_mc),
        "below_noise_upper_bound": bool(metrics.rmse_pcm < upper),
    }


# ---------------------------------------------------------------------------
# reactivity coefficients
# ---------------------------------------------------------------------------

def reactivity(k: np.ndarray | float) -> np.ndarray | float:
    """rho = (k - 1) / k, dimensionless."""
    return (np.asarray(k) - 1.0) / np.asarray(k)


@dataclass
class Coefficient:
    """A derivative of reactivity with respect to one parameter."""

    parameter: str
    at: dict
    value_pcm_per_unit: float
    sigma_pcm_per_unit: float
    units: str
    #: None for parameters whose sign is not fixed across the design space --
    #: moderator density and pitch both change sign at the moderation optimum.
    expected_sign: int | None
    consistent: bool | None = field(init=False)

    def __post_init__(self):
        self.consistent = (
            None if self.expected_sign is None
            else bool(np.sign(self.value_pcm_per_unit) == self.expected_sign)
        )

    @property
    def significant(self) -> bool:
        """Is the coefficient distinguishable from zero at 2 sigma?

        Worth asking separately from the sign: near the moderation optimum the
        true coefficient passes through zero, and a sign test on a value that
        is consistent with zero is meaningless either way.
        """
        return abs(self.value_pcm_per_unit) > 2.0 * self.sigma_pcm_per_unit

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["significant"] = self.significant
        return d


#: Physics the surrogate was never told, used as an independent check.
#:
#: Only the two parameters whose sign is unambiguous *everywhere in this design
#: space* are listed.  Moderator density and pitch are deliberately absent:
#: both act on k only through moderation, and the sampled box straddles the
#: moderation optimum, so their coefficients genuinely change sign inside it.
#: Asserting a fixed sign for them would be asserting bad physics -- they get
#: the stronger check in `moderation_optimum` instead.
EXPECTED_SIGNS = {
    # Doppler broadening widens the U238 capture resonances as the fuel heats,
    # so more neutrons are absorbed during slowing down: negative everywhere.
    "fuel_temperature": -1,
    # More U235 per unit volume: positive everywhere, though saturating.
    "enrichment": +1,
}


def moderation_ratio(pitch: float, density: float) -> float:
    """Density-weighted moderator-to-fuel volume ratio.

    The single quantity that both `pitch` and `moderator_density` act through:
    widening the lattice and densifying the coolant add moderator in the same
    way, as far as the neutron spectrum is concerned.
    """
    from .params import FIXED

    fuel_area = np.pi * FIXED["fuel_radius"] ** 2
    moderator_area = pitch**2 - np.pi * FIXED["clad_outer_radius"] ** 2
    return moderator_area / fuel_area * density


def moderation_optimum(gp: "GPSurrogate", base_point: Sequence[float],
                       n: int = 400) -> dict:
    """Locate the moderation peak from two directions and check they agree.

    A far stronger test than a sign check, and one the surrogate can fail.
    k-infinity peaks at some optimum moderation: below it the lattice is
    under-moderated and adding moderator helps; above it, added moderator
    parasitically absorbs more than it thermalises.  Crucially, `pitch` and
    `moderator_density` reach that optimum through the *same* physical
    quantity, the density-weighted moderator-to-fuel ratio.

    So: sweep density at fixed pitch, find the peak, convert to a moderation
    ratio.  Then sweep pitch at fixed density, find that peak, convert.  The
    two numbers come from different directions through a 4-D fitted surface
    and have no reason to agree unless the surrogate has learned the actual
    physics rather than a convenient interpolant.
    """
    base = np.asarray(base_point, dtype=float)
    i_rho = PARAM_NAMES.index("moderator_density")
    i_p = PARAM_NAMES.index("pitch")
    out: dict = {"at": dict(zip(PARAM_NAMES, base.tolist()))}

    for name, idx in (("moderator_density", i_rho), ("pitch", i_p)):
        bound = BOUNDS[name]
        grid = np.linspace(bound.low, bound.high, n)
        pts = np.tile(base, (n, 1))
        pts[:, idx] = grid
        k = gp.predict(pts)
        j = int(np.argmax(k))
        interior = 0 < j < n - 1
        peak_value = float(grid[j])
        ratio = (
            moderation_ratio(base[i_p], peak_value) if name == "moderator_density"
            else moderation_ratio(peak_value, base[i_rho])
        )
        out[name] = {
            "peak_at": peak_value,
            "peak_keff": float(k[j]),
            "peak_is_interior": interior,
            "moderation_ratio_at_peak": float(ratio),
        }

    r1 = out["moderator_density"]["moderation_ratio_at_peak"]
    r2 = out["pitch"]["moderation_ratio_at_peak"]
    both_interior = (out["moderator_density"]["peak_is_interior"]
                     and out["pitch"]["peak_is_interior"])
    out["agreement"] = {
        "ratio_from_density_sweep": r1,
        "ratio_from_pitch_sweep": r2,
        "relative_difference": abs(r1 - r2) / ((r1 + r2) / 2),
        # 5% is loose enough to survive the Monte Carlo noise in the training
        # labels and tight enough that an interpolant which had not learned
        # the physics would fail it.
        "consistent": bool(both_interior and abs(r1 - r2) / ((r1 + r2) / 2) < 0.05),
        "both_peaks_interior": bool(both_interior),
    }
    return out


def coefficient(
    gp: GPSurrogate,
    base_point: Sequence[float],
    parameter: str,
    delta: float | None = None,
) -> Coefficient:
    """Extract d(rho)/d(parameter) from the surrogate by central difference.

    The uncertainty is propagated from the GP's *joint* posterior at the two
    stencil points.  Treating them as independent would badly overstate the
    error: neighbouring predictions from a GP are strongly correlated, and it
    is exactly that correlation that makes a finite difference of a fitted
    surface far better determined than the two values separately.
    """
    idx = PARAM_NAMES.index(parameter)
    bound = BOUNDS[parameter]
    if delta is None:
        # 10% of the range, not 1-2%.  This is *not* a numerical-derivative
        # step on an exact function, where smaller is better; it is a
        # difference of two noisy GP predictions, and the uncertainty on the
        # result scales as 1/h.  A 2% step buried the Doppler coefficient
        # under its own error bar (-1.7 +/- 1.1 pcm/K); 10% resolves it.
        # The response is smooth on this scale, so the truncation error a
        # wider stencil buys is far smaller than the noise it removes.
        delta = 0.10 * (bound.high - bound.low)

    base = np.asarray(base_point, dtype=float)
    lo_pt, hi_pt = base.copy(), base.copy()
    lo_pt[idx] = bound.clip(base[idx] - delta)
    hi_pt[idx] = bound.clip(base[idx] + delta)
    h = hi_pt[idx] - lo_pt[idx]

    mean, cov = gp.predict_cov(np.vstack([lo_pt, hi_pt]))
    k_lo, k_hi = float(mean[0]), float(mean[1])
    var_diff = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    var_diff = max(var_diff, 0.0)

    dk_dp = (k_hi - k_lo) / h
    k_mid = 0.5 * (k_hi + k_lo)
    # rho = 1 - 1/k  =>  drho/dp = (1/k^2) dk/dp
    drho_dp = dk_dp / (k_mid**2)
    sigma = np.sqrt(var_diff) / abs(h) / (k_mid**2)

    return Coefficient(
        parameter=parameter,
        at=dict(zip(PARAM_NAMES, (float(v) for v in base))),
        value_pcm_per_unit=float(drho_dp * PCM),
        sigma_pcm_per_unit=float(sigma * PCM),
        units=f"pcm per {bound.units}",
        expected_sign=EXPECTED_SIGNS.get(parameter),
    )
