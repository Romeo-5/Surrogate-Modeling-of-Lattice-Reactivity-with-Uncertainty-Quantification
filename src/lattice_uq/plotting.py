"""Figures for the report.

Light-mode only and print-oriented: these are figures for a written report and
a README, not a themed web dashboard, so the palette is committed to one
surface rather than swapped.  Colours are the first three slots of a
CVD-validated categorical set (blue / orange / aqua), which clear the
all-pairs separation floors; every multi-series figure carries a legend, and
the aqua slot -- which sits below 3:1 against the surface -- is only used where
a legend or direct label makes identity explicit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator  # noqa: E402

# --- palette ---------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
BLUE, ORANGE, AQUA = SERIES
BAND = "#cde2fb"   # blue-100, for uncertainty bands
REF = "#52514e"    # reference / expectation lines


def apply_style() -> None:
    """Recessive axes, thin marks, no chartjunk."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",  # DejaVu has no semibold; asking for it warns
            "axes.labelsize": 9,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "text.color": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _finish(ax, title: str, xlabel: str, ylabel: str, note: str | None = None):
    ax.set_title(title, loc="left", color=INK, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if note:
        ax.text(
            0.0, -0.20, note, transform=ax.transAxes, fontsize=7.5,
            color=MUTED, va="top", ha="left", wrap=True,
        )


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --- Phase 1: verification -------------------------------------------------

def plot_spectrum(spectrum: dict, path: str | Path, title: str | None = None) -> Path:
    """Neutron flux per unit lethargy, with the three physical regions marked.

    Per *lethargy* rather than per eV: on a log energy axis a 1/E slowing-down
    flux is flat in lethargy, so the classic three-region shape -- thermal
    Maxwellian peak, flat epithermal plateau, fast fission bump -- is visible
    by inspection instead of being buried under the 1/E trend.
    """
    apply_style()
    e_lo = np.asarray(spectrum["energy_low"], dtype=float)
    e_hi = np.asarray(spectrum["energy_high"], dtype=float)
    flux = np.asarray(spectrum["flux"], dtype=float)
    err = np.asarray(spectrum.get("flux_std_dev", np.zeros_like(flux)), dtype=float)

    lethargy = np.log(e_hi / e_lo)
    phi = flux / lethargy
    phi_err = err / lethargy
    e_mid = np.sqrt(e_lo * e_hi)
    norm = np.max(phi)
    phi, phi_err = phi / norm, phi_err / norm

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    regions = [
        (1e-5, 0.625, "thermal\n(Maxwellian peak)"),
        (0.625, 1.0e5, "slowing down\n(~1/E, flat in lethargy)"),
        (1.0e5, 2.0e7, "fast\n(fission source)"),
    ]
    for i, (lo, hi, label) in enumerate(regions):
        if i % 2 == 0:
            ax.axvspan(lo, hi, color=GRID, alpha=0.45, lw=0)
        # Inside the axes, not above them: placed above, these run into the
        # title. The headroom in ylim below is what makes the space for them.
        ax.text(
            np.sqrt(lo * hi), 0.97, label, ha="center", va="top",
            fontsize=7.5, color=MUTED, transform=ax.get_xaxis_transform(),
        )

    ax.fill_between(
        e_mid, phi - 2 * phi_err, phi + 2 * phi_err, color=BAND, lw=0,
    )
    ax.plot(e_mid, phi, color=BLUE)

    ax.set_xscale("log")
    ax.set_xlim(1e-3, 2e7)
    ax.set_ylim(0, 1.45)
    ax.xaxis.set_major_locator(LogLocator(numticks=12))
    # One series: the title names it, so a legend box would be noise. The
    # uncertainty band is a property of that series, not a second one.
    peak_sigma = float(np.max(phi_err[phi > 0.1 * phi.max()])) if phi.max() else 0.0
    _finish(
        ax,
        title or "Neutron energy spectrum in the pin cell",
        "Energy (eV)",
        "Flux per unit lethargy (normalised)",
        "Shaded band is ±2σ of the Monte Carlo tally — invisible at this "
        f"scale (largest ≈{peak_sigma * 100:.2f}% of the peak). Thermal / "
        "epithermal cut at 0.625 eV.",
    )
    return _save(fig, path)


def plot_entropy(entropy: Sequence[float], inactive: int, path: str | Path) -> Path:
    """Shannon entropy of the fission source vs batch.

    The point of the figure is to justify the choice of `inactive`: entropy
    must plateau before the first active batch, or the eigenvalue is being
    tallied on an unconverged source.
    """
    apply_style()
    h = np.asarray(entropy, dtype=float)
    batches = np.arange(1, len(h) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(batches, h, color=BAND, lw=1.0, label="per batch")

    # Batch-to-batch entropy is dominated by sampling noise, which hides the
    # thing the figure exists to show. The running mean is what makes a drift
    # -- or the absence of one -- visible.
    window = max(5, len(h) // 20)
    kernel = np.ones(window) / window
    smooth = np.convolve(h, kernel, mode="valid")
    ax.plot(batches[window - 1:], smooth, color=BLUE, lw=1.8,
            label=f"running mean ({window} batches)")

    ax.axvline(inactive, color=ORANGE, lw=1.4, ls="--")
    # At the top: the bottom of the axes is where the legend goes.
    ax.text(
        inactive, 0.97, f"  first active batch ({inactive})",
        transform=ax.get_xaxis_transform(), color=ORANGE,
        fontsize=8, va="top", ha="left",
    )
    converged = float(np.mean(h[inactive:])) if len(h) > inactive else float("nan")
    ax.axhline(converged, color=REF, lw=0.9, ls=":")
    ax.legend(loc="lower right", ncols=2)
    _finish(
        ax,
        "Fission source convergence (Shannon entropy)",
        "Batch",
        "Shannon entropy",
        f"Dotted line: mean entropy over active batches ({converged:.4f}). "
        f"The source must plateau before batch {inactive} for the reported k "
        "to be unbiased; a flat running mean is that plateau.",
    )
    return _save(fig, path)


def plot_sigma_convergence(
    particles: Sequence[int], sigma_pcm: Sequence[float], path: str | Path
) -> Path:
    """Reported sigma against particle count, with the 1/sqrt(N) expectation.

    This is the check that the reported uncertainty behaves like a Monte Carlo
    uncertainty at all.  If sigma does not fall as 1/sqrt(N), something is
    wrong with the run -- correlated batches, or an unconverged source -- and
    every uncertainty quoted downstream is meaningless.
    """
    apply_style()
    n = np.asarray(particles, dtype=float)
    s = np.asarray(sigma_pcm, dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(n, s, "o-", color=BLUE, ms=6, label="reported σ")
    ref = s[0] * np.sqrt(n[0] / n)
    ax.plot(n, ref, ls="--", color=REF, lw=1.2, label=r"$1/\sqrt{N}$ from first point")

    slope = np.polyfit(np.log(n), np.log(s), 1)[0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Label the particle counts actually run. The default log locator puts
    # 2x10^3 and 3x10^3 close enough together that their labels overlap.
    ax.set_xticks(n)
    ax.set_xticklabels([f"{int(v):,}" for v in n])
    ax.set_xticks([], minor=True)
    ax.legend()
    _finish(
        ax,
        "Statistical uncertainty vs particles per batch",
        "Particles per batch",
        "σ(k) (pcm)",
        f"Fitted log–log slope {slope:+.3f}; the Monte Carlo expectation is "
        f"{-0.5:+.3f}.",
    )
    return _save(fig, path)


def plot_enrichment_sweep(
    enrichment: Sequence[float],
    keff: Sequence[float],
    sigma: Sequence[float],
    path: str | Path,
) -> Path:
    """k-infinity vs enrichment, with error bars scaled so they are visible."""
    apply_style()
    e = np.asarray(enrichment, dtype=float)
    k = np.asarray(keff, dtype=float)
    s = np.asarray(sigma, dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(
        e, k, yerr=50 * s, fmt="o-", color=BLUE, ms=6,
        capsize=3, elinewidth=1.2, ecolor=ORANGE,
    )
    for xi, ki in zip(e, k):
        # Offset to the lower right rather than straight up: the exaggerated
        # error bars occupy the space directly above each marker.
        ax.annotate(
            f"{ki:.4f}", (xi, ki), textcoords="offset points",
            xytext=(8, -11), ha="left", fontsize=7.5, color=INK_2,
        )
    monotonic = bool(np.all(np.diff(k) > 0))
    _finish(
        ax,
        "k-infinity vs fuel enrichment",
        "Enrichment (wt% U-235)",
        "k-infinity",
        f"Error bars are ±50σ so they are visible at this scale "
        f"(σ ≈ {np.mean(s) * 1e5:.0f} pcm). "
        f"Monotonically increasing: {monotonic}.",
    )
    return _save(fig, path)


# --- Phase 3: surrogate ----------------------------------------------------

def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_sigma: np.ndarray | None,
    path: str | Path,
    label: str = "Gaussian process",
) -> Path:
    """Predicted vs calculated k-infinity on held-out points."""
    apply_style()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.02 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=REF, lw=1.0, ls="--",
            label="1:1")
    if y_true_sigma is not None:
        ax.errorbar(
            y_true, y_pred, xerr=np.asarray(y_true_sigma) * 1.96,
            fmt="none", ecolor=BAND, elinewidth=1.0,
            label="±1.96σ of the transport run",
        )
    ax.scatter(y_true, y_pred, s=22, color=BLUE, edgecolor=SURFACE, linewidth=0.6,
               zorder=3, label=label)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)) * 1e5)
    _finish(
        ax,
        "Surrogate vs transport, held-out points",
        "OpenMC k-infinity",
        "Surrogate k-infinity",
        f"Held-out RMSE {rmse:.1f} pcm on {len(y_true)} points.",
    )
    return _save(fig, path)


def plot_error_vs_sigma(
    abs_err_pcm: np.ndarray,
    mc_sigma_pcm: np.ndarray,
    path: str | Path,
    gp_sigma_pcm: np.ndarray | None = None,
    deconvolved_rmse_pcm: float | None = None,
) -> Path:
    """The headline figure: surrogate error against Monte Carlo uncertainty.

    Two panels, because one alone under-sells it.  Every run in the campaign
    uses the same particle count, so the left scatter collapses to a narrow
    vertical stripe -- fine for reading off how many points sit below the
    diagonal, useless for seeing the shape of the error.  The right panel is
    the distribution of |error| with the same σ marked on it, which is where
    the comparison is actually legible.
    """
    apply_style()
    err = np.asarray(abs_err_pcm, dtype=float)
    sig = np.asarray(mc_sigma_pcm, dtype=float)
    rms_sigma = float(np.sqrt(np.mean(sig**2)))
    rmse = float(np.sqrt(np.mean(err**2)))
    frac = float(np.mean(err < sig))

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1, 1.15]}
    )

    # -- left: per-point comparison
    top = float(max(err.max(), sig.max()) * 1.15)
    ax.fill_between([0, top], [0, top], [top, top], color=GRID, alpha=0.5, lw=0)
    ax.text(
        0.04 * top, 0.96 * top, "error exceeds MC σ",
        fontsize=8, color=MUTED, va="top",
    )
    ax.plot([0, top], [0, top], color=REF, lw=1.1, ls="--", label="error = MC σ")
    ax.scatter(sig, err, s=24, color=BLUE, edgecolor=SURFACE, linewidth=0.6,
               zorder=3, label="held-out point")
    if gp_sigma_pcm is not None:
        ax.scatter(
            sig, np.asarray(gp_sigma_pcm), s=18, marker="^", color=ORANGE,
            edgecolor=SURFACE, linewidth=0.5, zorder=3,
            label="GP predictive σ",
        )
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.legend(loc="upper right")
    ax.set_title("Per held-out point", loc="left", color=INK, pad=8)
    ax.set_xlabel("Monte Carlo σ of the reference run (pcm)")
    ax.set_ylabel("|surrogate − transport| (pcm)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # -- right: the distribution, which is what the claim is really about
    bins = np.linspace(0, top, 26)
    ax2.hist(err, bins=bins, color=BLUE, alpha=0.85, edgecolor=SURFACE,
             linewidth=0.8)

    # Three markers, at staggered heights. The measured RMSE and the reference
    # noise land within a few pcm of each other, so labelling them at the same
    # height would overlap them into illegibility.
    # Headroom above the tallest bar, so the marker labels sit in empty space
    # rather than on top of the distribution they are annotating.
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.55)

    markers = [
        (rms_sigma, REF, "--", 0.98,
         f"RMS Monte Carlo σ\n{rms_sigma:.0f} pcm"),
        (rmse, ORANGE, "-", 0.78,
         f"measured RMSE\n{rmse:.0f} pcm"),
    ]
    if deconvolved_rmse_pcm is not None:
        markers.append((
            float(deconvolved_rmse_pcm), AQUA, "-", 0.98,
            f"surrogate error vs\nthe true surface\n{deconvolved_rmse_pcm:.0f} pcm",
        ))

    for value, colour, style, y, label in markers:
        ax2.axvline(value, color=colour, lw=1.6, ls=style)
        # Labels go on whichever side of the line has room.
        right_side = value < 0.55 * top
        ax2.text(
            value, y, ("    " + label) if right_side else (label + "    "),
            transform=ax2.get_xaxis_transform(), color=colour, fontsize=8,
            va="top", ha="left" if right_side else "right",
        )
    ax2.set_xlim(0, top)
    ax2.set_title("Distribution of held-out error", loc="left", color=INK, pad=8)
    ax2.set_xlabel("|surrogate − transport| (pcm)")
    ax2.set_ylabel("Held-out points")
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.suptitle(
        "Surrogate error against transport uncertainty",
        x=0.005, ha="left", fontsize=11, fontweight="bold", color=INK,
    )
    # Hard-wrapped rather than left as one long line: savefig(bbox_inches=
    # "tight") grows the canvas to fit the caption, so a single long line
    # stretches the whole figure and squashes both panels.
    tail = (
        "\nThe aqua line removes the reference runs' own noise, leaving the "
        "surrogate's error against the true response surface."
        if deconvolved_rmse_pcm is not None else ""
    )
    fig.text(
        0.005, -0.06,
        f"{frac:.0%} of held-out points fall below their reference run's own "
        "statistical noise.\nThe measured RMSE is scored against noisy "
        "references, so it counts the Monte Carlo uncertainty twice." + tail,
        fontsize=7.5, color=MUTED, ha="left", va="top", linespacing=1.6,
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_error_by_parameter(
    x: np.ndarray,
    abs_err_pcm: np.ndarray,
    param_names: Sequence[str],
    path: str | Path,
    mc_sigma_pcm: float | None = None,
) -> Path:
    """Where in the design space the surrogate is weakest.

    Small multiples rather than a single coloured scatter: the question is
    per-dimension structure in the residual, and one panel per parameter
    answers it without needing a colour scale at all.
    """
    apply_style()
    x = np.asarray(x, dtype=float)
    err = np.asarray(abs_err_pcm, dtype=float)
    n = len(param_names)

    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.scatter(x[:, i], err, s=18, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
        if mc_sigma_pcm is not None:
            ax.axhline(mc_sigma_pcm, color=REF, lw=1.0, ls="--")
        ax.set_xlabel(name.replace("_", " "))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("|surrogate − transport| (pcm)")
    if mc_sigma_pcm is not None:
        axes[-1].text(
            0.98, mc_sigma_pcm, "  mean MC σ", transform=axes[-1].get_yaxis_transform(),
            color=REF, fontsize=7.5, va="bottom", ha="right",
        )
    fig.suptitle(
        "Held-out error by parameter", x=0.005, ha="left", fontsize=10,
        fontweight="bold", color=INK,
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_coefficient(
    x: Sequence[float],
    value_pcm: Sequence[float],
    sigma_pcm: Sequence[float],
    path: str | Path,
    xlabel: str,
    title: str,
    note: str = "",
    ylabel: str = "Reactivity coefficient (pcm per unit)",
) -> Path:
    """A reactivity coefficient extracted from the surrogate, with its band."""
    apply_style()
    x = np.asarray(x, dtype=float)
    v = np.asarray(value_pcm, dtype=float)
    s = np.asarray(sigma_pcm, dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.fill_between(x, v - 1.96 * s, v + 1.96 * s, color=BAND, lw=0,
                    label="±1.96σ from the GP posterior")
    ax.plot(x, v, color=BLUE, label="surrogate")
    ax.axhline(0.0, color=REF, lw=1.0, ls="--")

    # Headroom below the band, and the legend parked in it: the zero line is
    # the thing the reader is asked to compare against, so the legend must not
    # sit on top of it.
    lo, hi = float((v - 1.96 * s).min()), float(max((v + 1.96 * s).max(), 0.0))
    pad = 0.28 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 0.08 * (hi - lo))
    ax.legend(loc="lower right", ncols=2)

    _finish(ax, title, xlabel, ylabel, note)
    return _save(fig, path)


def plot_learning_curve(
    sizes: Sequence[int],
    rmse_pcm: Sequence[float],
    path: str | Path,
    mc_sigma_pcm: float | None = None,
    rmse_pcm_2: Sequence[float] | None = None,
    label_1: str = "Gaussian process",
    label_2: str = "Neural network",
) -> Path:
    """Held-out RMSE vs training-set size.

    Tells you whether the campaign was long enough: a curve still falling at
    the right-hand edge means more transport runs would still buy accuracy,
    while a curve flattening onto the MC-σ line means the surrogate has
    reached the noise floor of its own training labels.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(sizes, rmse_pcm, "o-", color=BLUE, ms=5, label=label_1)
    if rmse_pcm_2 is not None:
        ax.plot(sizes, rmse_pcm_2, "s-", color=ORANGE, ms=5, label=label_2)
    if mc_sigma_pcm is not None:
        ax.axhline(mc_sigma_pcm, color=REF, lw=1.1, ls="--")
        # Left end: that is where the curves are still high and the space just
        # above the line is empty. On the right they have converged onto it.
        ax.text(
            0.01, mc_sigma_pcm, "  Monte Carlo σ of a single run",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=8, color=REF,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Label the training-set sizes actually run, rather than powers of ten.
    ax.set_xticks(list(sizes))
    ax.set_xticklabels([str(int(v)) for v in sizes])
    ax.set_xticks([], minor=True)
    ax.legend()
    _finish(
        ax,
        "Held-out accuracy vs number of transport runs",
        "Training points",
        "Held-out RMSE (pcm)",
        "Where the curve meets the dashed line, the surrogate is as accurate "
        "as the calculation it replaces.",
    )
    return _save(fig, path)
