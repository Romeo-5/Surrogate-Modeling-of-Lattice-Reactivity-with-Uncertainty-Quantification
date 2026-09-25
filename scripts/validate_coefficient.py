#!/usr/bin/env python
"""Check the surrogate's reactivity coefficients against direct transport.

Recovering the *sign* of the Doppler coefficient from a surrogate is a
consistency check.  Recovering its *magnitude*, and agreeing with a transport
calculation that was never in the training set, is a validation -- and it is
the one a reactor physicist will ask for.

Method: run dedicated high-precision transport calculations at the two ends of
a parameter interval, holding everything else at the centre of the design
space, and form

    drho/dp  =  [rho(p_hi) - rho(p_lo)] / (p_hi - p_lo)

with the uncertainty propagated from the two runs.  A wide interval is used
deliberately: the reactivity difference across 600-1200 K is of order 1000 pcm
while each run's sigma is a few tens, so the difference is determined to a few
percent.  A narrow interval would be swamped by statistics.

The surrogate is then asked for the same interval-averaged coefficient, and
the two are compared against their combined uncertainty.

A detail worth stating: for fuel temperature the interval ends (600 K, 1200 K)
are *exactly* tabulated library temperatures, so the reference runs involve no
temperature interpolation at all, while nearly every training point did.  That
makes this a stricter test than it looks -- agreement validates OpenMC's
temperature interpolation as used by the campaign, not just the fit.

Usage:
    python scripts/validate_coefficient.py --particles 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lattice_uq import BOUNDS, PARAM_NAMES  # noqa: E402
from lattice_uq.params import DesignPoint  # noqa: E402
from lattice_uq.runner import check_cross_sections, run_point  # noqa: E402
from lattice_uq.store import ResultStore  # noqa: E402
from lattice_uq.surrogate import (  # noqa: E402
    EXPECTED_SIGNS,
    GPSurrogate,
    design_matrix,
    reactivity,
)

PCM = 1e5


def _point_at(centre: np.ndarray, param: str, value: float) -> DesignPoint:
    v = centre.copy()
    v[PARAM_NAMES.index(param)] = value
    return DesignPoint.from_vector(v)


def direct_coefficient(centre: np.ndarray, param: str, lo: float, hi: float,
                       particles: int, batches: int, inactive: int,
                       threads: int | None) -> dict:
    """Interval-averaged d(rho)/d(param) from two dedicated transport runs."""
    out = {}
    for label, value in (("lo", lo), ("hi", hi)):
        point = _point_at(centre, param, value)
        result, _ = run_point(
            point, particles=particles, batches=batches, inactive=inactive,
            threads=threads, seed=20260201 if label == "lo" else 20260202,
        )
        out[label] = {
            "value": value,
            "keff": result.keff,
            "keff_sigma": result.keff_sigma,
            "wall_time_s": result.wall_time_s,
        }
        print(f"[validate]   {param}={value:<8.4g} k={result.keff:.5f} "
              f"+/- {result.keff_sigma * PCM:.0f} pcm "
              f"({result.wall_time_s:.0f} s)")

    k_lo, k_hi = out["lo"]["keff"], out["hi"]["keff"]
    s_lo, s_hi = out["lo"]["keff_sigma"], out["hi"]["keff_sigma"]
    h = hi - lo

    drho = (reactivity(k_hi) - reactivity(k_lo)) / h
    # d(rho)/dk = 1/k^2, so each run's sigma enters scaled by that.
    var = (s_hi / k_hi**2) ** 2 + (s_lo / k_lo**2) ** 2
    out["coefficient_pcm_per_unit"] = float(drho * PCM)
    out["sigma_pcm_per_unit"] = float(np.sqrt(var) / abs(h) * PCM)
    return out


def surrogate_coefficient(gp: GPSurrogate, centre: np.ndarray, param: str,
                          lo: float, hi: float) -> dict:
    """The same interval-averaged coefficient, read off the fitted surface."""
    pts = np.vstack([
        _point_at(centre, param, lo).vector(),
        _point_at(centre, param, hi).vector(),
    ])
    mean, cov = gp.predict_cov(pts)
    k_lo, k_hi = float(mean[0]), float(mean[1])
    h = hi - lo

    drho = (reactivity(k_hi) - reactivity(k_lo)) / h
    # Joint posterior: the covariance term matters, because neighbouring GP
    # predictions are correlated and the difference is far better determined
    # than either value alone.
    g_lo, g_hi = 1.0 / k_lo**2, 1.0 / k_hi**2
    var = (g_hi**2 * cov[1, 1] + g_lo**2 * cov[0, 0]
           - 2.0 * g_hi * g_lo * cov[0, 1])
    return {
        "keff_lo": k_lo,
        "keff_hi": k_hi,
        "coefficient_pcm_per_unit": float(drho * PCM),
        "sigma_pcm_per_unit": float(np.sqrt(max(var, 0.0)) / abs(h) * PCM),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/surrogate/coefficient_validation.json")
    ap.add_argument("--particles", type=int, default=50_000)
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--inactive", type=int, default=40)
    ap.add_argument("--threads", type=int, default=0)
    # Only parameters with an unambiguous sign across the whole range.
    # Moderator density and pitch peak inside the sampled box, so an
    # interval-averaged coefficient across it would average a sign change
    # into a meaningless number.
    ap.add_argument("--params", default="fuel_temperature,enrichment")
    args = ap.parse_args()

    check_cross_sections()
    threads = args.threads or None

    df = ResultStore(args.data_dir).to_frame()
    if df.empty:
        parquet = Path(args.data_dir) / "results.parquet"
        if not parquet.is_file():
            print("[validate] no runs found; run the campaign first",
                  file=sys.stderr)
            return 2
        df = pd.read_parquet(parquet)
    train = df[df.tag == "train"]
    if train.empty:
        print("[validate] no training runs; run the campaign first",
              file=sys.stderr)
        return 2

    gp = GPSurrogate().fit(
        design_matrix(train), train.keff.to_numpy(), train.keff_sigma.to_numpy()
    )
    centre = np.array([0.5 * (BOUNDS[n].low + BOUNDS[n].high) for n in PARAM_NAMES])

    report = {
        "centre": dict(zip(PARAM_NAMES, centre.tolist())),
        "n_train": int(len(train)),
        "particles": args.particles,
        "comparisons": {},
    }

    for param in args.params.split(","):
        bound = BOUNDS[param]
        lo, hi = bound.low, bound.high
        print(f"\n[validate] {param} over [{lo}, {hi}] {bound.units}")

        direct = direct_coefficient(centre, param, lo, hi, args.particles,
                                    args.batches, args.inactive, threads)
        surro = surrogate_coefficient(gp, centre, param, lo, hi)

        d, sd = direct["coefficient_pcm_per_unit"], direct["sigma_pcm_per_unit"]
        g, sg = surro["coefficient_pcm_per_unit"], surro["sigma_pcm_per_unit"]
        combined = float(np.hypot(sd, sg))
        # "Sigmas apart" is the honest way to state agreement: it asks whether
        # the gap is explainable by the uncertainty on both sides at once.
        n_sigma = abs(g - d) / combined if combined else float("inf")

        expected = EXPECTED_SIGNS.get(param)
        report["comparisons"][param] = {
            "units": f"pcm per {bound.units}",
            "interval": [lo, hi],
            "direct": direct,
            "surrogate": surro,
            "difference_pcm_per_unit": g - d,
            "combined_sigma_pcm_per_unit": combined,
            "sigmas_apart": n_sigma,
            "agree_within_2_sigma": bool(n_sigma < 2.0),
            "expected_sign": expected,
            "sign_correct": bool(np.sign(d) == expected) if expected else None,
        }

        print(f"[validate]   direct     {d:+9.2f} +/- {sd:.2f} "
              f"pcm per {bound.units}")
        print(f"[validate]   surrogate  {g:+9.2f} +/- {sg:.2f} "
              f"pcm per {bound.units}")
        print(f"[validate]   difference {g - d:+9.2f} "
              f"({n_sigma:.1f}σ) -> "
              f"{'AGREE' if n_sigma < 2 else 'DISAGREE'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[validate] wrote {out}")

    disagreements = [
        p for p, c in report["comparisons"].items()
        if not c["agree_within_2_sigma"]
    ]
    return 1 if disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())
