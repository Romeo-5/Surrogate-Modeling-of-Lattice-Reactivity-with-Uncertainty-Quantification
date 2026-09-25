#!/usr/bin/env python
"""Render the results section of the README from the JSON reports.

Numbers in a README rot the moment a campaign is re-run with different
settings.  Generating them from `verification.json` and `surrogate_report.json`
means the prose and the data cannot disagree: re-run the pipeline, re-run this,
and the README is correct by construction.

Everything between the RESULTS markers in README.md is replaced.

Usage:
    python scripts/make_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BEGIN = "<!-- BEGIN RESULTS -->"
END = "<!-- END RESULTS -->"


def _fmt(x, spec=".1f", missing="n/a"):
    if x is None:
        return missing
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def _tick(ok) -> str:
    if ok is None:
        return "not run"
    return "yes" if ok else "**no**"


def verification_section(v: dict) -> list[str]:
    ref = v["reference_case"]
    lines = [
        "### Phase 1 — verification",
        "",
        f"Cross sections: `{Path(v['cross_sections']).parent.name}` · "
        f"OpenMC {ref['openmc_version']}",
        "",
        "**Reference pin cell** (the geometry and composition distributed with "
        "OpenMC's own examples, derived from BEAVRS):",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| enrichment | {_fmt(ref['point']['enrichment'], '.2f')} wt% U-235 |",
        f"| fuel temperature | {_fmt(ref['point']['fuel_temperature'], '.0f')} K |",
        f"| moderator density | {_fmt(ref['point']['moderator_density'], '.4f')} g/cm³ |",
        f"| pitch | {_fmt(ref['point']['pitch'], '.5f')} cm |",
        f"| histories | {ref['particles']:,} × "
        f"{ref['batches'] - ref['inactive']} active batches |",
        f"| **k-infinity** | **{_fmt(ref['keff'], '.5f')} ± "
        f"{_fmt(ref['keff_sigma_pcm'], '.0f')} pcm** |",
        f"| thermal flux fraction in fuel | "
        f"{_fmt(ref.get('thermal_flux_fraction'), '.4f')} |",
        f"| wall time | {_fmt(ref['wall_time_s'], '.1f')} s |",
        "",
    ]

    ec = ref.get("entropy_check", {})
    conv = v.get("sigma_convergence", {})
    sweep = v["enrichment_sweep"]
    lines += [
        "**Checks.** Each one is a way the model could be wrong that a single "
        "k value would not reveal.",
        "",
        "| check | what it rules out | result |",
        "|---|---|---|",
        f"| Shannon entropy plateaus before the first active batch | "
        f"tallying on an unconverged fission source | "
        f"{_tick(ec.get('drift_within_one_std'))} "
        f"(half-to-half drift {_fmt(ec.get('half_to_half_drift'), '.2e')}) |",
        f"| σ falls as 1/√N | a reported uncertainty that is not a Monte Carlo "
        f"uncertainty | {_tick(conv.get('passes'))} "
        f"(log–log slope {_fmt(conv.get('loglog_slope'), '+.3f')}, "
        f"expected {-0.5:+.3f}) |",
        f"| k rises monotonically with enrichment, beyond the noise | "
        f"a sign error or a broken material build | "
        f"{_tick(sweep.get('monotonic_beyond_noise'))} "
        f"(smallest step {_fmt(sweep.get('min_step_pcm'), '.0f')} pcm) |",
        "",
        "**What this is not.** No published benchmark k-infinity is reproduced "
        "here. The geometry and composition are the BEAVRS-derived pin cell, "
        "but it is run hot (600 K) with 975 ppm boron, and there is no "
        "published value for *that* state to compare against — quoting one "
        "for a different state would be worse than quoting none. Verification "
        "therefore rests on the checks above, which are properties the answer "
        "must have rather than a number it must match. A formal comparison "
        "against the Mosteller Doppler-defect benchmark, whose pin-cell "
        "specifications come with published MCNP eigenvalues, is the obvious "
        "next step and is listed under limitations.",
        "",
        "Figures: [geometry](results/verification/fig_geometry.png) · "
        "[spectrum](results/verification/fig_spectrum.png) · "
        "[entropy](results/verification/fig_entropy.png) · "
        "[σ convergence](results/verification/fig_sigma_conv.png) · "
        "[enrichment sweep](results/verification/fig_enrichment.png)",
        "",
        "![Neutron energy spectrum](results/verification/fig_spectrum.png)",
        "",
    ]
    return lines


def surrogate_section(s: dict) -> list[str]:
    gp = s["gaussian_process"]
    nn = s["neural_network"]
    u = gp["uncertainty_comparison"]
    rep = s.get("replicate_analysis", {})

    lines = [
        "### Phase 2 — campaign",
        "",
        f"{s['n_runs_total']} transport runs: {s['n_train']} Latin-hypercube "
        f"training points, {s['n_test']} independent Sobol held-out points, "
        "plus replicates of a few points under different RNG seeds.",
        f"k-infinity spans {_fmt(s['keff_range'][0], '.4f')}–"
        f"{_fmt(s['keff_range'][1], '.4f')} across the design space.",
        "",
    ]

    if rep.get("available"):
        lines += [
            "**Is the quoted uncertainty honest?** Repeating a design point "
            "under different seeds measures the run-to-run scatter directly. "
            "The ratio of observed scatter to reported σ is "
            f"**{_fmt(rep['mean_ratio'], '.2f')}** "
            f"(≈1 means yes) — {_tick(rep['consistent'])}. "
            "Everything below is measured against that σ, so this had to be "
            "checked first.",
            "",
        ]

    lines += [
        "### Phase 3 — surrogate and the uncertainty result",
        "",
        "| | Gaussian process | Neural network |",
        "|---|---|---|",
        f"| held-out RMSE | **{_fmt(gp['metrics']['rmse_pcm'])} pcm** | "
        f"{_fmt(nn['metrics']['rmse_pcm'])} pcm |",
        f"| held-out MAE | {_fmt(gp['metrics']['mae_pcm'])} pcm | "
        f"{_fmt(nn['metrics']['mae_pcm'])} pcm |",
        f"| max abs error | {_fmt(gp['metrics']['max_abs_pcm'])} pcm | "
        f"{_fmt(nn['metrics']['max_abs_pcm'])} pcm |",
        f"| bias | {_fmt(gp['metrics']['bias_pcm'], '+.1f')} pcm | "
        f"{_fmt(nn['metrics']['bias_pcm'], '+.1f')} pcm |",
        f"| R² | {_fmt(gp['metrics']['r2'], '.5f')} | "
        f"{_fmt(nn['metrics']['r2'], '.5f')} |",
        f"| 95% interval coverage | "
        f"{_fmt(gp['metrics'].get('coverage_95'), '.2f')} | — |",
        "",
        "**The comparison that matters:**",
        "",
        "| quantity | pcm |",
        "|---|---|",
        f"| surrogate held-out RMSE | {_fmt(u['surrogate_rmse_pcm'])} |",
        f"| RMS Monte Carlo σ of the held-out runs | "
        f"{_fmt(u['rms_mc_sigma_pcm'])} |",
        f"| upper bound on the noise contribution (adds training-label "
        f"noise in quadrature) | {_fmt(u['noise_upper_bound_pcm'])} |",
        f"| **surrogate error vs the true surface** (reference noise "
        f"deconvolved) | **{_fmt(u['deconvolved_rmse_pcm'])}** |",
        f"| ratio measured RMSE / σ | {_fmt(u['ratio_rmse_to_mc_sigma'], '.2f')} |",
        f"| ratio deconvolved / σ | "
        f"{_fmt(u['ratio_deconvolved_to_mc_sigma'], '.2f')} |",
        "",
        "**Reading this table.** The measured RMSE is scored against "
        "references that are themselves noisy, so it double-counts the Monte "
        "Carlo uncertainty — once in the surrogate's error and once in the "
        "value it is compared to. Those are independent, so they add in "
        "quadrature and can be separated: "
        "`E[(pred − ref)²] = E[(pred − truth)²] + σ_ref²`. The "
        "deconvolved figure is the surrogate's error against the true "
        "response surface, which is what anyone using it actually cares "
        "about.",
        "",
        f"Measured RMSE below the transport runs' own σ: "
        f"{_tick(u['below_mc_sigma'])}. "
        f"Deconvolved error below it: {_tick(u['deconvolved_below_mc_sigma'])}. "
        f"Measured RMSE below the noise upper bound: "
        f"{_tick(u['below_noise_upper_bound'])}.",
        "",
        "![Surrogate vs transport on held-out points]"
        "(results/surrogate/fig_parity.png)",
        "",
        "![Surrogate error against transport uncertainty]"
        "(results/surrogate/fig_error_vs_sigma.png)",
        "",
        f"**Cost.** A transport run at campaign settings takes "
        f"{_fmt(s['speedup']['mean_transport_wall_time_s'])} s on "
        f"{s['speedup'].get('transport_threads_per_run', '?')} threads "
        f"({_fmt(s['speedup'].get('mean_transport_core_seconds'))} core-seconds); "
        f"one surrogate evaluation takes "
        f"{_fmt(s['speedup']['surrogate_predict_time_s'] * 1e6)} µs on one "
        f"thread — a factor of {_fmt(s['speedup']['speedup_factor'], '.2e')}. "
        "That is a *marginal* cost comparison: it excludes the campaign that "
        "trained the surrogate, which a real application amortises over "
        "however many evaluations it needs.",
        "",
    ]

    worst = s.get("worst_point", {})
    if worst:
        params = ", ".join(
            f"{k.replace('_', ' ')} {_fmt(v, '.3g')}"
            for k, v in worst["parameters"].items()
        )
        corr = s.get("error_vs_edge_distance_corr")
        # The interesting question is whether the error is an edge effect. A
        # clearly negative correlation says yes; near zero says the weakness is
        # spread uniformly, which points at label noise instead of model bias.
        if corr is None:
            edge_note = ""
        elif corr < -0.2:
            edge_note = (
                f" Across all held-out points the correlation between absolute "
                f"error and distance from the edge of the design space is "
                f"{_fmt(corr, '+.2f')}: the error concentrates near the "
                "boundary, where a point has neighbours on fewer sides and the "
                "GP is extrapolating rather than interpolating."
            )
        else:
            edge_note = (
                f" That single point is not part of a pattern, though. The "
                f"correlation between absolute error and distance from the edge "
                f"of the design space is {_fmt(corr, '+.2f')} — essentially "
                "none, so the error is spread evenly rather than concentrated "
                "at the boundary. Combined with the learning curve flattening, "
                "that points at Monte Carlo noise in the labels as the "
                "limitation rather than any particular region being hard to "
                "fit."
            )
        lines += [
            "**Where it is weakest.** The largest held-out error is "
            f"{_fmt(worst['abs_error_pcm'])} pcm at ({params}), against a "
            f"Monte Carlo σ of {_fmt(worst['mc_sigma_pcm'])} pcm there — "
            f"{_fmt(worst['abs_error_pcm'] / worst['mc_sigma_pcm'], '.1f')}σ, "
            "the sort of outlier a held-out set this size produces on its "
            f"own.{edge_note}",
            "",
            "![Held-out error by parameter]"
            "(results/surrogate/fig_error_by_param.png)",
            "",
        ]

    rc = s.get("reactivity_coefficients", {})
    if rc:
        lines += [
            "### Reactivity coefficients — a physics check the model was never "
            "given",
            "",
            "Reactivity is ρ = (k−1)/k; the coefficients below are derivatives "
            "of the *fitted surface*, with uncertainties propagated from the "
            "GP's joint posterior at the two stencil points.",
            "",
            "| coefficient | value | expected sign | consistent |",
            "|---|---|---|---|",
        ]
        labels = {
            "fuel_temperature": "fuel temperature (Doppler)",
            "moderator_density": "moderator density",
            "enrichment": "enrichment",
        }
        for key, c in rc.get("at_centre", {}).items():
            sign = "negative" if c["expected_sign"] < 0 else "positive"
            lines.append(
                f"| {labels.get(key, key)} | "
                f"{_fmt(c['value_pcm_per_unit'], '+.1f')} ± "
                f"{_fmt(c['sigma_pcm_per_unit'], '.1f')} {c['units']} | "
                f"{sign} | {_tick(c['consistent'])} |"
            )
        prof = rc.get("profiles", {}).get("fuel_temperature", {})
        lines += [
            "",
            "The Doppler coefficient is negative across the whole "
            f"600–1200 K range: {_tick(prof.get('negative_everywhere'))}. "
            "That is the single most important sign in reactor physics — as "
            "the fuel heats, Doppler broadening of the U-238 capture "
            "resonances increases absorption during slowing down, and "
            "reactivity must fall. The surrogate was trained only on "
            "(parameters → k); it was never told this.",
            "",
            "![Doppler coefficient](results/surrogate/fig_doppler.png)",
            "",
        ]

        mod = rc.get("moderation_optimum", {})
        agree = mod.get("agreement", {})
        if agree:
            lines += [
                "**The stronger check: where is the moderation optimum?** "
                "Pitch and moderator density are two different inputs, but "
                "they act on k-infinity through one physical quantity — the "
                "density-weighted moderator-to-fuel volume ratio. So the peak "
                "found by sweeping density at fixed pitch and the peak found "
                "by sweeping pitch at fixed density must land on the *same* "
                "ratio. Two directions through a 4-D fitted surface have no "
                "reason to agree unless the surrogate has learned the physics "
                "rather than a convenient interpolant.",
                "",
                "| swept | peak at | implied Vm/Vf |",
                "|---|---|---|",
                f"| moderator density | "
                f"{_fmt(mod['moderator_density']['peak_at'], '.4f')} g/cm³ | "
                f"{_fmt(agree['ratio_from_density_sweep'], '.3f')} |",
                f"| pitch | {_fmt(mod['pitch']['peak_at'], '.4f')} cm | "
                f"{_fmt(agree['ratio_from_pitch_sweep'], '.3f')} |",
                "",
                f"They differ by "
                f"{_fmt(agree['relative_difference'] * 100, '.2f')}% — "
                f"{_tick(agree['consistent'])}.",
                "",
                "This is also why no fixed sign is asserted for the moderator "
                "density or pitch coefficients: the sampled box **straddles "
                "the optimum**, so both coefficients genuinely change sign "
                "inside it — positive where the lattice is under-moderated, "
                "negative where added moderator absorbs more than it "
                "thermalises. An earlier version of this analysis asserted a "
                "positive sign and was wrong; the design space is wider than "
                "a nominal PWR lattice, and its centre sits on the "
                "over-moderated side.",
                "",
                "![Moderator density coefficient]"
                "(results/surrogate/fig_moderator_density.png)",
                "",
            ]

    lc = s.get("learning_curve")
    if lc:
        lines += [
            "**How many runs were actually needed?**",
            "",
            "![Learning curve](results/surrogate/fig_learning_curve.png)",
            "",
        ]
    return lines


def validation_section(v: dict) -> list[str]:
    """Surrogate coefficients against dedicated transport runs."""
    labels = {
        "fuel_temperature": "fuel temperature (Doppler)",
        "moderator_density": "moderator density",
        "enrichment": "enrichment",
    }
    lines = [
        "### Validation against transport the surrogate never saw",
        "",
        "Signs are a consistency check; magnitudes are a validation. Each "
        "coefficient below is also computed directly from a pair of dedicated "
        f"high-precision transport runs ({v['particles']:,} particles each) at "
        "the ends of the parameter's range, which were never part of the "
        "training set. A wide interval is used deliberately — the reactivity "
        "difference is of order 1000 pcm while each run's σ is a few tens, so "
        "the difference is well determined.",
        "",
        "| coefficient | direct transport | surrogate | agreement |",
        "|---|---|---|---|",
    ]
    for key, c in v.get("comparisons", {}).items():
        d, s = c["direct"], c["surrogate"]
        verdict = "✓" if c["agree_within_2_sigma"] else "**✗**"
        lines.append(
            f"| {labels.get(key, key)} ({c['units']}) | "
            f"{_fmt(d['coefficient_pcm_per_unit'], '+.2f')} ± "
            f"{_fmt(d['sigma_pcm_per_unit'], '.2f')} | "
            f"{_fmt(s['coefficient_pcm_per_unit'], '+.2f')} ± "
            f"{_fmt(s['sigma_pcm_per_unit'], '.2f')} | "
            f"{_fmt(c['sigmas_apart'], '.1f')}σ apart {verdict} |"
        )
    lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--verification", default="results/verification/verification.json")
    ap.add_argument("--surrogate", default="results/surrogate/surrogate_report.json")
    ap.add_argument("--validation",
                    default="results/surrogate/coefficient_validation.json")
    args = ap.parse_args()

    body: list[str] = []
    for path, renderer in ((args.verification, verification_section),
                           (args.surrogate, surrogate_section),
                           (args.validation, validation_section)):
        p = Path(path)
        if p.is_file():
            body += renderer(json.loads(p.read_text()))
        else:
            print(f"[report] {p} not found; skipping that section")

    if not body:
        print("[report] nothing to render")
        return 1

    readme = Path(args.readme)
    text = readme.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print(f"[report] {readme} is missing the {BEGIN} / {END} markers")
        return 1

    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = head + BEGIN + "\n\n" + "\n".join(body).rstrip() + "\n\n" + END + tail
    readme.write_text(new, encoding="utf-8")
    print(f"[report] updated {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
