#!/usr/bin/env python
"""Phase 3 -- fit the surrogates and test the uncertainty claim.

The claim under test is: *the surrogate's held-out prediction error is smaller
than the stochastic uncertainty of the transport calculation it replaces.*

To make that testable rather than rhetorical, the script does four things:

1. **Calibrates the uncertainty itself** from the replicate runs -- the same
   design point under different RNG seeds.  If the observed scatter does not
   match the reported sigma, then the sigma the claim is measured against is
   not trustworthy and everything after it is suspect.  This is checked first.
2. **Fits a Gaussian process and a neural network** on the LHS training set and
   evaluates both on the independent Sobol set.
3. **Compares** held-out RMSE against the transport runs' own sigma -- a real
   floor, since the reference values themselves are noisy at that level -- and
   against an upper bound that also folds in the training-label noise.
4. **Extracts reactivity coefficients** from the fitted surrogate and checks
   their signs against physics the model was never told.

Usage:
    python scripts/train_surrogate.py --data-dir data --outdir results/surrogate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lattice_uq import BOUNDS, PARAM_NAMES  # noqa: E402
from lattice_uq import plotting  # noqa: E402
from lattice_uq.store import ResultStore  # noqa: E402
from lattice_uq.surrogate import (  # noqa: E402
    EXPECTED_SIGNS,
    GPSurrogate,
    MLPSurrogate,
    coefficient,
    design_matrix,
    evaluate,
    moderation_optimum,
    uncertainty_comparison,
)

PCM = 1e5


# ---------------------------------------------------------------------------

def replicate_analysis(df: pd.DataFrame) -> dict:
    """Is the reported Monte Carlo sigma believable?

    For each repeated design point, compare the sample standard deviation of
    k across seeds with the sigma OpenMC reported.  The ratio should be about
    1.  A ratio well above 1 means the reported sigma understates the true
    spread -- the classic symptom of inter-batch correlation from an
    under-converged fission source -- and would inflate every later claim.
    """
    reps = df[df.tag == "replicate"]
    if reps.empty:
        return {"available": False}

    rows = []
    for run_id, grp in reps.groupby("run_id"):
        if len(grp) < 3:
            continue
        # ddof=1: this is a sample standard deviation over seeds.
        observed = float(grp.keff.std(ddof=1))
        reported = float(np.sqrt(np.mean(grp.keff_sigma**2)))
        rows.append(
            {
                "run_id": run_id,
                "n_seeds": int(len(grp)),
                "mean_keff": float(grp.keff.mean()),
                "observed_std_pcm": observed * PCM,
                "reported_sigma_pcm": reported * PCM,
                "ratio": observed / reported if reported else float("nan"),
                **{p: float(grp[p].iloc[0]) for p in PARAM_NAMES},
            }
        )

    ratios = np.array([r["ratio"] for r in rows], dtype=float)
    return {
        "available": True,
        "points": rows,
        "mean_ratio": float(np.mean(ratios)),
        # With 8 seeds the sample std is itself uncertain at roughly the 25%
        # level, so the band has to be generous to avoid crying wolf.
        "consistent": bool(0.6 < np.mean(ratios) < 1.6),
        "interpretation": (
            "observed scatter / reported sigma; ~1 means the quoted Monte "
            "Carlo uncertainty is honest"
        ),
    }


def fit_and_score(
    name: str,
    surrogate,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    s_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    s_te: np.ndarray,
) -> dict:
    t0 = time.perf_counter()
    surrogate.fit(x_tr, y_tr, s_tr)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred, pred_std = surrogate.predict(x_te, return_std=True)
    predict_time = time.perf_counter() - t0

    metrics = evaluate(y_te, pred, pred_std, s_te)
    comparison = uncertainty_comparison(metrics, s_te, s_tr)
    return {
        "name": name,
        "metrics": metrics.as_dict(),
        "uncertainty_comparison": comparison,
        "fit_time_s": fit_time,
        "predict_time_s_per_point": predict_time / max(len(x_te), 1),
        "_pred": pred,
        "_pred_std": pred_std,
    }


def learning_curve(
    x_tr: np.ndarray, y_tr: np.ndarray, s_tr: np.ndarray,
    x_te: np.ndarray, y_te: np.ndarray, s_te: np.ndarray,
    sizes: list[int], seed: int = 0,
) -> dict:
    """Held-out RMSE as a function of how many transport runs were afforded."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_tr))
    gp_rmse, mlp_rmse, used = [], [], []
    for n in sizes:
        if n > len(x_tr):
            continue
        idx = order[:n]
        gp = GPSurrogate().fit(x_tr[idx], y_tr[idx], s_tr[idx])
        gp_rmse.append(evaluate(y_te, gp.predict(x_te)).rmse_pcm)
        mlp = MLPSurrogate().fit(x_tr[idx], y_tr[idx], s_tr[idx])
        mlp_rmse.append(evaluate(y_te, mlp.predict(x_te)).rmse_pcm)
        used.append(n)
        print(f"[surrogate]   n={n:<5} GP {gp_rmse[-1]:7.1f} pcm   "
              f"MLP {mlp_rmse[-1]:7.1f} pcm")
    return {"sizes": used, "gp_rmse_pcm": gp_rmse, "mlp_rmse_pcm": mlp_rmse}


def coefficients(gp: GPSurrogate, outdir: Path) -> dict:
    """Reactivity coefficients from the surrogate, and the physics checks.

    Two kinds of check, in increasing order of how hard they are to pass:

    * **Signs.**  Doppler broadening must make the fuel temperature
      coefficient negative, and more U235 must raise reactivity.  Both hold
      everywhere in this design space.
    * **The moderation optimum.**  Pitch and moderator density act on k only
      through the moderator-to-fuel ratio, so the peak each produces must sit
      at the same value of that ratio.  Two different directions through a 4-D
      fitted surface have no reason to agree unless the surrogate has learned
      the physics.

    Note what is *not* checked: a fixed sign for moderator density or pitch.
    The sampled box straddles the moderation optimum, so those coefficients
    legitimately change sign inside it.
    """
    # Evaluate at the centre of the design space, holding the other parameters
    # at their mid-range values.
    centre = np.array(
        [0.5 * (BOUNDS[n].low + BOUNDS[n].high) for n in PARAM_NAMES]
    )

    out: dict = {"at_centre": {}, "profiles": {}}
    for param in EXPECTED_SIGNS:
        c = coefficient(gp, centre, param)
        out["at_centre"][param] = c.as_dict()
        sign = "negative" if c.expected_sign < 0 else "positive"
        status = "PASS" if c.consistent else "FAIL"
        print(f"[surrogate]   {status}  d(rho)/d({param}) = "
              f"{c.value_pcm_per_unit:+.1f} +/- {c.sigma_pcm_per_unit:.1f} "
              f"{c.units}  (expected {sign})")

    # Doppler coefficient across the temperature range: the physically
    # interesting profile, since it should also weaken in magnitude as the
    # fuel heats (resonance absorption saturates).
    temps = np.linspace(
        BOUNDS["fuel_temperature"].low + 30,
        BOUNDS["fuel_temperature"].high - 30,
        25,
    )
    vals, sigs = [], []
    for t in temps:
        p = centre.copy()
        p[PARAM_NAMES.index("fuel_temperature")] = t
        c = coefficient(gp, p, "fuel_temperature")
        vals.append(c.value_pcm_per_unit)
        sigs.append(c.sigma_pcm_per_unit)

    fig = plotting.plot_coefficient(
        temps, vals, sigs, outdir / "fig_doppler.png",
        xlabel="Fuel temperature (K)",
        ylabel="d(rho)/dT  (pcm per K)",
        title="Fuel temperature (Doppler) coefficient from the surrogate",
        note=(
            "Must be negative everywhere: Doppler broadening of the U-238 "
            "capture resonances increases absorption as the fuel heats. "
            "The surrogate was never told this."
        ),
    )
    out["profiles"]["fuel_temperature"] = {
        "temperature_K": temps.tolist(),
        "value_pcm_per_K": vals,
        "sigma_pcm_per_K": sigs,
        "negative_everywhere": bool(np.all(np.asarray(vals) < 0)),
        "figure": str(fig),
    }

    # The moderation check.  Both pitch and moderator density act on k only
    # through the moderator-to-fuel ratio, so the peak each of them produces
    # must sit at the same value of that ratio.
    mod = moderation_optimum(gp, centre)
    out["moderation_optimum"] = mod
    agree = mod["agreement"]
    status = "PASS" if agree["consistent"] else "FAIL"
    print(f"[surrogate]   {status}  moderation optimum from density sweep "
          f"Vm/Vf={agree['ratio_from_density_sweep']:.3f}, from pitch sweep "
          f"{agree['ratio_from_pitch_sweep']:.3f} "
          f"({agree['relative_difference'] * 100:.1f}% apart)")

    # Moderator density coefficient across its range.
    dens = np.linspace(
        BOUNDS["moderator_density"].low + 0.02,
        BOUNDS["moderator_density"].high - 0.02,
        25,
    )
    vals_d, sigs_d = [], []
    for d in dens:
        p = centre.copy()
        p[PARAM_NAMES.index("moderator_density")] = d
        c = coefficient(gp, p, "moderator_density")
        vals_d.append(c.value_pcm_per_unit)
        sigs_d.append(c.sigma_pcm_per_unit)
    fig_d = plotting.plot_coefficient(
        dens, vals_d, sigs_d, outdir / "fig_moderator_density.png",
        xlabel="Moderator density (g/cm3)",
        ylabel="d(rho)/d(rho_mod)  (pcm per g/cm3)",
        title="Moderator density coefficient from the surrogate",
        note=(
            "This design space straddles the moderation optimum, so the sign "
            "flips inside it: positive where the lattice is under-moderated "
            "(voiding costs reactivity), negative where extra moderator "
            "absorbs more than it thermalises. The zero crossing locates the "
            "optimum."
        ),
    )
    v_d = np.asarray(vals_d)
    out["profiles"]["moderator_density"] = {
        "density_g_cm3": dens.tolist(),
        "value_pcm_per_g_cm3": vals_d,
        "sigma_pcm_per_g_cm3": sigs_d,
        # Not "positive everywhere" -- the sampled range crosses the moderation
        # optimum, so a sign change here is the expected physics, not a fault.
        "changes_sign": bool(v_d.min() < 0 < v_d.max()),
        "figure": str(fig_d),
    }
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--outdir", default="results/surrogate")
    ap.add_argument("--learning-curve", action="store_true", default=True)
    ap.add_argument("--no-learning-curve", dest="learning_curve",
                    action="store_false")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # The per-run JSON files are the working state and are not committed; the
    # Parquet roll-up is. Falling back to it means the whole analysis can be
    # reproduced from a fresh clone without re-running the campaign.
    df = ResultStore(args.data_dir).to_frame()
    source = "per-run store"
    if df.empty:
        parquet = Path(args.data_dir) / "results.parquet"
        if parquet.is_file():
            df = pd.read_parquet(parquet)
            source = str(parquet)
        else:
            print("[surrogate] no runs found; run scripts/run_campaign.py "
                  "first", file=sys.stderr)
            return 2
    print(f"[surrogate] loaded {len(df)} runs from {source}")

    report: dict = {"n_runs_total": int(len(df))}

    print("[surrogate] checking the reported Monte Carlo uncertainty")
    rep = replicate_analysis(df)
    report["replicate_analysis"] = rep
    if rep.get("available"):
        verdict = "consistent" if rep["consistent"] else "INCONSISTENT"
        print(f"[surrogate]   observed/reported sigma = {rep['mean_ratio']:.2f} "
              f"-> {verdict}")

    train = df[df.tag == "train"]
    test = df[df.tag == "test"]
    if train.empty or test.empty:
        print("[surrogate] need both train and test runs in the store",
              file=sys.stderr)
        return 2

    x_tr, y_tr, s_tr = design_matrix(train), train.keff.to_numpy(), \
        train.keff_sigma.to_numpy()
    x_te, y_te, s_te = design_matrix(test), test.keff.to_numpy(), \
        test.keff_sigma.to_numpy()
    print(f"[surrogate] {len(x_tr)} training / {len(x_te)} held-out points")
    report["n_train"] = int(len(x_tr))
    report["n_test"] = int(len(x_te))
    report["keff_range"] = [float(df.keff.min()), float(df.keff.max())]

    gp = GPSurrogate()
    gp_res = fit_and_score("Gaussian process", gp, x_tr, y_tr, s_tr,
                           x_te, y_te, s_te)
    mlp_res = fit_and_score("Neural network (MLP)", MLPSurrogate(),
                            x_tr, y_tr, s_tr, x_te, y_te, s_te)

    for res in (gp_res, mlp_res):
        m, u = res["metrics"], res["uncertainty_comparison"]
        print(f"\n[surrogate] {res['name']}")
        print(f"   held-out RMSE        {m['rmse_pcm']:8.1f} pcm")
        print(f"   held-out MAE         {m['mae_pcm']:8.1f} pcm")
        print(f"   max |error|          {m['max_abs_pcm']:8.1f} pcm")
        print(f"   bias                 {m['bias_pcm']:+8.1f} pcm")
        print(f"   R^2                  {m['r2']:8.5f}")
        if m["coverage_95"] is not None:
            print(f"   95% interval coverage{m['coverage_95']:8.2f}"
                  f"   (target 0.95)")
        print(f"   RMS Monte Carlo sigma{u['rms_mc_sigma_pcm']:8.1f} pcm")
        print(f"   ratio RMSE / sigma   {u['ratio_rmse_to_mc_sigma']:8.2f}")
        print(f"   below MC sigma:      {u['below_mc_sigma']}")

    report["gaussian_process"] = {
        k: v for k, v in gp_res.items() if not k.startswith("_")
    }
    report["gaussian_process"]["kernel"] = gp.kernel_summary
    report["neural_network"] = {
        k: v for k, v in mlp_res.items() if not k.startswith("_")
    }

    # Cost of a transport run vs the surrogate, from measured wall times.
    # The campaign summary carries the thread count, which the comparison needs
    # to be honest: a wall time is not a cost unless you say how many cores
    # produced it.
    summary_path = Path(args.data_dir) / "campaign_summary.json"
    threads = None
    if summary_path.is_file():
        threads = json.loads(summary_path.read_text()).get(
            "settings", {}
        ).get("threads_per_run")

    mc_time = float(train.wall_time_s.mean())
    surrogate_time = gp_res["predict_time_s_per_point"]
    report["speedup"] = {
        "mean_transport_wall_time_s": mc_time,
        "transport_threads_per_run": threads,
        "mean_transport_core_seconds": mc_time * threads if threads else None,
        "surrogate_predict_time_s": surrogate_time,
        "speedup_factor": mc_time / surrogate_time if surrogate_time else None,
        "note": (
            "One transport run at campaign settings"
            + (f" on {threads} threads" if threads else "")
            + " vs one GP evaluation on one thread. The comparison is of "
            "marginal cost only: it excludes the campaign that trained the "
            "surrogate, which a real application has to amortise over however "
            "many evaluations it needs."
        ),
    }
    print(f"\n[surrogate] transport {mc_time:.1f} s/point vs surrogate "
          f"{surrogate_time * 1e6:.1f} us/point "
          f"({report['speedup']['speedup_factor']:.2e}x)")

    # -- figures ---------------------------------------------------------
    pred, pred_std = gp_res["_pred"], gp_res["_pred_std"]
    abs_err_pcm = np.abs(pred - y_te) * PCM
    figs = {
        "parity": str(plotting.plot_parity(
            y_te, pred, s_te, outdir / "fig_parity.png")),
        "error_vs_sigma": str(plotting.plot_error_vs_sigma(
            abs_err_pcm, s_te * PCM, outdir / "fig_error_vs_sigma.png",
            gp_sigma_pcm=pred_std * PCM,
            deconvolved_rmse_pcm=gp_res["uncertainty_comparison"][
                "deconvolved_rmse_pcm"])),
        "error_by_parameter": str(plotting.plot_error_by_parameter(
            x_te, abs_err_pcm, PARAM_NAMES, outdir / "fig_error_by_param.png",
            mc_sigma_pcm=float(np.mean(s_te) * PCM))),
    }

    # Where the surrogate is weakest, and a hypothesis why.
    worst = int(np.argmax(abs_err_pcm))
    report["worst_point"] = {
        "parameters": dict(zip(PARAM_NAMES, x_te[worst].tolist())),
        "abs_error_pcm": float(abs_err_pcm[worst]),
        "mc_sigma_pcm": float(s_te[worst] * PCM),
        "gp_sigma_pcm": float(pred_std[worst] * PCM),
        "distance_to_nearest_edge": {
            n: float(min(x_te[worst, i] - BOUNDS[n].low,
                         BOUNDS[n].high - x_te[worst, i]))
            for i, n in enumerate(PARAM_NAMES)
        },
    }
    # Correlation of error with distance from the design-space centre tells
    # you whether the weakness is an edge effect (few neighbours) or genuine
    # curvature somewhere in the interior.
    lo = np.array([BOUNDS[n].low for n in PARAM_NAMES])
    hi = np.array([BOUNDS[n].high for n in PARAM_NAMES])
    unit = (x_te - lo) / (hi - lo)
    edge_dist = np.min(np.minimum(unit, 1 - unit), axis=1)
    report["error_vs_edge_distance_corr"] = float(
        np.corrcoef(edge_dist, abs_err_pcm)[0, 1]
    )

    if args.learning_curve:
        print("\n[surrogate] learning curve")
        sizes = [20, 40, 80, 120, 160, 240, 320, len(x_tr)]
        sizes = sorted({s for s in sizes if s <= len(x_tr)})
        lc = learning_curve(x_tr, y_tr, s_tr, x_te, y_te, s_te, sizes)
        report["learning_curve"] = lc
        figs["learning_curve"] = str(plotting.plot_learning_curve(
            lc["sizes"], lc["gp_rmse_pcm"], outdir / "fig_learning_curve.png",
            mc_sigma_pcm=float(np.sqrt(np.mean(s_te**2)) * PCM),
            rmse_pcm_2=lc["mlp_rmse_pcm"],
        ))

    print("\n[surrogate] reactivity coefficients")
    report["reactivity_coefficients"] = coefficients(gp, outdir)
    figs.update({
        "doppler": report["reactivity_coefficients"]["profiles"]
        ["fuel_temperature"]["figure"],
        "moderator_density": report["reactivity_coefficients"]["profiles"]
        ["moderator_density"]["figure"],
    })
    report["figures"] = figs

    path = outdir / "surrogate_report.json"
    path.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[surrogate] report written to {path}")

    headline = gp_res["uncertainty_comparison"]
    print("\n[surrogate] === headline ===")
    print(f"  GP held-out RMSE         {headline['surrogate_rmse_pcm']:7.1f} pcm "
          "(measured against noisy references)")
    print(f"  transport RMS sigma      {headline['rms_mc_sigma_pcm']:7.1f} pcm")
    print(f"  noise upper bound        {headline['noise_upper_bound_pcm']:7.1f} pcm")
    print(f"  deconvolved RMSE         {headline['deconvolved_rmse_pcm']:7.1f} pcm "
          "(vs the true surface)")
    print(f"  measured RMSE below MC sigma:    {headline['below_mc_sigma']}")
    print(f"  deconvolved RMSE below MC sigma: "
          f"{headline['deconvolved_below_mc_sigma']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
