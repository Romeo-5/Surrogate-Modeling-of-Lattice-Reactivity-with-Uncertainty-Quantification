#!/usr/bin/env python
"""Phase 1 -- build the pin cell and show that it is right.

Reporting a k-infinity is not verification.  This script produces the four
pieces of evidence that a reactor physicist would actually ask for:

1. **Reference case.**  The pin cell distributed with OpenMC's own examples,
   run at high particle count, reported as k +/- sigma.  The comparison value
   is not hardcoded: it is computed here and written to the report, so the
   claim is reproducible rather than asserted.
2. **Source convergence.**  Shannon entropy of the fission source against
   batch, which is what justifies the number of inactive batches.
3. **Statistical behaviour.**  Reported sigma against particles per batch.  If
   it does not fall as 1/sqrt(N), every uncertainty quoted later in the project
   is meaningless, so this is checked before anything else is built on it.
4. **Physics.**  The energy spectrum shows the thermal peak, the 1/E
   slowing-down plateau and the fast fission region; k-infinity rises
   monotonically with enrichment.

Everything is written to a JSON report plus figures.

Usage:
    python scripts/verify.py --outdir results/verification
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lattice_uq import BOUNDS, FIXED, REFERENCE_POINT, DesignPoint  # noqa: E402
from lattice_uq import plotting  # noqa: E402
from lattice_uq.model import build_geometry, build_materials  # noqa: E402
from lattice_uq.runner import check_cross_sections, run_point  # noqa: E402


def plot_geometry(point: DesignPoint, path: Path) -> str:
    """Render an xy slice of the pin cell.

    Falls back cleanly: the interactive plotter needs openmc.lib, which is not
    available in every build, and a missing geometry picture should not fail a
    verification run.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.colors import to_rgb

    materials = build_materials(point)
    geometry = build_geometry(point, materials)
    fuel, helium, clad, water = materials

    # OpenMC's plotter wants 0-255 RGB triples, not CSS hex strings.
    def rgb(hex_color: str) -> tuple[int, int, int]:
        return tuple(int(round(255 * c)) for c in to_rgb(hex_color))

    hexes = {
        fuel: "#eb6834",     # orange
        helium: "#f0efec",   # near-surface neutral
        clad: "#52514e",     # dark neutral
        water: "#2a78d6",    # blue
    }
    colors = {mat: rgb(h) for mat, h in hexes.items()}
    width = point.pitch * 1.02
    try:
        plotting.apply_style()
        ax = geometry.root_universe.plot(
            basis="xy",
            width=(width, width),
            pixels=(900, 900),
            color_by="material",
            colors=colors,
        )
        ax.set_title(
            f"PWR pin cell, pitch {point.pitch:.5f} cm",
            loc="left", fontsize=10, fontweight="semibold",
        )
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        # The shared style turns the grid on, which here draws lines across the
        # materials it is meant to sit behind.
        ax.grid(False)
        handles = [
            plt.Line2D([], [], marker="s", ls="", ms=9, color=c,
                       markeredgecolor="#fcfcfb")
            for c in hexes.values()
        ]
        ax.legend(
            handles,
            ["UO2 fuel", "He gap", "Zircaloy-4 clad", "borated water"],
            frameon=False, fontsize=8, loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(path, dpi=200, bbox_inches="tight",
                          facecolor=plotting.SURFACE)
        plt.close(ax.figure)
        return str(path)
    except Exception as exc:  # pragma: no cover - build-dependent
        print(f"[verify] geometry plot unavailable ({exc})", file=sys.stderr)
        return ""


def reference_case(outdir: Path, particles: int, batches: int, inactive: int,
                   threads: int) -> dict:
    print(f"[verify] reference case: {particles} particles x {batches} batches")
    result, spectrum = run_point(
        REFERENCE_POINT,
        particles=particles,
        batches=batches,
        inactive=inactive,
        threads=threads,
        with_tallies=True,
        seed=1,
    )
    # Keep the raw arrays next to the figures. Redrawing a figure should never
    # require re-running transport, and the underlying numbers are what someone
    # checking the work actually wants.
    if spectrum is not None:
        np.savez_compressed(outdir / "reference_spectrum.npz", **spectrum)
    if result.entropy:
        np.savez_compressed(
            outdir / "reference_entropy.npz",
            entropy=np.asarray(result.entropy), inactive=np.asarray(inactive),
        )

    figs = {}
    if spectrum is not None:
        figs["spectrum"] = str(
            plotting.plot_spectrum(
                spectrum,
                outdir / "fig_spectrum.png",
                title=(
                    "Neutron energy spectrum, "
                    f"{REFERENCE_POINT.enrichment:.1f} wt% pin cell at "
                    f"{REFERENCE_POINT.fuel_temperature:.0f} K"
                ),
            )
        )
    if result.entropy:
        figs["entropy"] = str(
            plotting.plot_entropy(result.entropy, inactive, outdir / "fig_entropy.png")
        )

    # An unconverged source shows up as drift in the entropy across the active
    # batches, so compare the first and second half rather than eyeballing.
    entropy_check = {}
    if len(result.entropy) > inactive + 10:
        active = np.asarray(result.entropy[inactive:])
        first, second = np.array_split(active, 2)
        drift = float(np.mean(second) - np.mean(first))
        entropy_check = {
            "active_mean": float(np.mean(active)),
            "active_std": float(np.std(active)),
            "half_to_half_drift": drift,
            "drift_within_one_std": bool(abs(drift) < np.std(active)),
        }

    return {
        "point": REFERENCE_POINT.as_dict(),
        "keff": result.keff,
        "keff_sigma": result.keff_sigma,
        "keff_sigma_pcm": result.keff_sigma * 1e5,
        "particles": particles,
        "batches": batches,
        "inactive": inactive,
        "wall_time_s": result.wall_time_s,
        "openmc_version": result.openmc_version,
        "thermal_flux_fraction": result.tallies.get("thermal_flux_fraction"),
        "entropy_check": entropy_check,
        "figures": figs,
    }


def sigma_convergence(outdir: Path, counts: list[int], batches: int,
                      inactive: int, threads: int) -> dict:
    """Check that sigma falls as 1/sqrt(N)."""
    print(f"[verify] sigma convergence over {counts}")
    sigmas, keffs = [], []
    for n in counts:
        result, _ = run_point(
            REFERENCE_POINT, particles=n, batches=batches,
            inactive=inactive, threads=threads, seed=7,
        )
        sigmas.append(result.keff_sigma * 1e5)
        keffs.append(result.keff)
        print(f"         N={n:>7}  k={result.keff:.5f} +/- {sigmas[-1]:5.1f} pcm "
              f"({result.wall_time_s:.1f} s)")

    slope = float(np.polyfit(np.log(counts), np.log(sigmas), 1)[0])
    fig = plotting.plot_sigma_convergence(counts, sigmas, outdir / "fig_sigma_conv.png")

    # A 1/sqrt(N) estimator has slope -0.5; allow +/-0.08 for the scatter in
    # sigma-of-sigma at these batch counts.
    return {
        "particles": counts,
        "keff": keffs,
        "sigma_pcm": sigmas,
        "loglog_slope": slope,
        "expected_slope": -0.5,
        "passes": bool(abs(slope + 0.5) < 0.08),
        "figure": str(fig),
    }


def enrichment_sweep(outdir: Path, particles: int, batches: int, inactive: int,
                     threads: int, n: int = 7) -> dict:
    """k-infinity must rise monotonically with enrichment."""
    lo, hi = BOUNDS["enrichment"].low, BOUNDS["enrichment"].high
    values = np.linspace(lo, hi, n)
    print(f"[verify] enrichment sweep {lo}-{hi} wt% in {n} steps")

    keffs, sigmas = [], []
    for e in values:
        point = DesignPoint(
            enrichment=float(e),
            fuel_temperature=REFERENCE_POINT.fuel_temperature,
            moderator_density=REFERENCE_POINT.moderator_density,
            pitch=REFERENCE_POINT.pitch,
        )
        result, _ = run_point(
            point, particles=particles, batches=batches,
            inactive=inactive, threads=threads, seed=11,
        )
        keffs.append(result.keff)
        sigmas.append(result.keff_sigma)
        print(f"         {e:.2f} wt%  k={result.keff:.5f} "
              f"+/- {result.keff_sigma * 1e5:.0f} pcm")

    diffs = np.diff(keffs)
    # Monotonic *beyond the noise*: each step must rise by more than the
    # combined sigma of its two endpoints, or the ordering is not established.
    step_sigma = np.sqrt(np.asarray(sigmas[:-1]) ** 2 + np.asarray(sigmas[1:]) ** 2)
    fig = plotting.plot_enrichment_sweep(
        values, keffs, sigmas, outdir / "fig_enrichment.png"
    )
    return {
        "enrichment": [float(v) for v in values],
        "keff": keffs,
        "keff_sigma": sigmas,
        "monotonic": bool(np.all(diffs > 0)),
        "monotonic_beyond_noise": bool(np.all(diffs > 2 * step_sigma)),
        "min_step_pcm": float(diffs.min() * 1e5),
        "figure": str(fig),
    }


def replot(outdir: Path) -> int:
    """Redraw every figure from the saved report and arrays.

    Changing how a figure looks should cost seconds, not another transport
    campaign.  Everything needed is already on disk: the scalar results in
    verification.json, the spectrum and entropy as .npz.
    """
    path = outdir / "verification.json"
    if not path.is_file():
        print(f"[verify] {path} not found; run the full verification first",
              file=sys.stderr)
        return 2
    report = json.loads(path.read_text())
    ref = report["reference_case"]
    figs: dict[str, str] = {}

    spec_path = outdir / "reference_spectrum.npz"
    if spec_path.is_file():
        with np.load(spec_path) as z:
            spectrum = {k: z[k] for k in z.files}
        figs["spectrum"] = str(plotting.plot_spectrum(
            spectrum, outdir / "fig_spectrum.png",
            title=(
                f"Neutron energy spectrum, {ref['point']['enrichment']:.1f} wt% "
                f"pin cell at {ref['point']['fuel_temperature']:.0f} K"
            ),
        ))

    ent_path = outdir / "reference_entropy.npz"
    if ent_path.is_file():
        with np.load(ent_path) as z:
            figs["entropy"] = str(plotting.plot_entropy(
                z["entropy"].tolist(), int(z["inactive"]),
                outdir / "fig_entropy.png",
            ))

    conv = report.get("sigma_convergence")
    if conv:
        figs["sigma_convergence"] = str(plotting.plot_sigma_convergence(
            conv["particles"], conv["sigma_pcm"], outdir / "fig_sigma_conv.png"
        ))

    sweep = report.get("enrichment_sweep")
    if sweep:
        figs["enrichment"] = str(plotting.plot_enrichment_sweep(
            sweep["enrichment"], sweep["keff"], sweep["keff_sigma"],
            outdir / "fig_enrichment.png",
        ))

    report["geometry_figure"] = plot_geometry(
        REFERENCE_POINT, outdir / "fig_geometry.png"
    )
    ref["figures"] = {**ref.get("figures", {}), **{
        k: v for k, v in figs.items() if k in ("spectrum", "entropy")
    }}
    if conv and "sigma_convergence" in figs:
        conv["figure"] = figs["sigma_convergence"]
    if sweep and "enrichment" in figs:
        sweep["figure"] = figs["enrichment"]
    path.write_text(json.dumps(report, indent=2))

    for name, fig in {**figs, "geometry": report["geometry_figure"]}.items():
        print(f"[verify] {name:<18} {fig or 'FAILED'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="results/verification")
    ap.add_argument("--particles", type=int, default=50_000,
                    help="particles per batch for the reference case")
    ap.add_argument("--sweep-particles", type=int, default=20_000)
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--inactive", type=int, default=40)
    ap.add_argument("--threads", type=int, default=0,
                    help="OpenMP threads; 0 lets OpenMC decide")
    ap.add_argument("--skip-convergence", action="store_true")
    ap.add_argument("--geometry-only", action="store_true",
                    help="redraw the geometry figure and update an existing "
                         "report, without running any transport")
    ap.add_argument("--replot", action="store_true",
                    help="redraw every figure from the saved report and "
                         "arrays, without running any transport")
    args = ap.parse_args()

    xs = check_cross_sections()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    threads = args.threads or None

    if args.geometry_only:
        path = outdir / "verification.json"
        report = json.loads(path.read_text()) if path.is_file() else {}
        report["geometry_figure"] = plot_geometry(
            REFERENCE_POINT, outdir / "fig_geometry.png"
        )
        path.write_text(json.dumps(report, indent=2))
        print(f"[verify] geometry figure: {report['geometry_figure'] or 'FAILED'}")
        return 0 if report["geometry_figure"] else 1

    if args.replot:
        return replot(outdir)

    report: dict = {
        "cross_sections": xs,
        "fixed_parameters": FIXED,
        "design_space": {k: asdict(v) for k, v in BOUNDS.items()},
    }

    report["geometry_figure"] = plot_geometry(
        REFERENCE_POINT, outdir / "fig_geometry.png"
    )
    report["reference_case"] = reference_case(
        outdir, args.particles, args.batches, args.inactive, threads
    )
    if not args.skip_convergence:
        report["sigma_convergence"] = sigma_convergence(
            outdir, [2_000, 5_000, 10_000, 20_000, 50_000],
            args.batches, args.inactive, threads,
        )
    report["enrichment_sweep"] = enrichment_sweep(
        outdir, args.sweep_particles, args.batches, args.inactive, threads
    )

    path = outdir / "verification.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\n[verify] report written to {path}")

    ref = report["reference_case"]
    print(f"[verify] reference k-inf = {ref['keff']:.5f} "
          f"+/- {ref['keff_sigma_pcm']:.0f} pcm")
    checks = {
        "sigma ~ 1/sqrt(N)": report.get("sigma_convergence", {}).get("passes"),
        "k rises with enrichment": report["enrichment_sweep"]["monotonic_beyond_noise"],
        "source converged": ref["entropy_check"].get("drift_within_one_std"),
    }
    for name, ok in checks.items():
        status = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        print(f"[verify]   {status}  {name}")
    return 0 if all(v is not False for v in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
