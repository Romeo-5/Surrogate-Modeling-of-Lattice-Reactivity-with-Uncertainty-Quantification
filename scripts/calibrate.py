#!/usr/bin/env python
"""Measure how this machine actually runs the campaign, before committing to it.

Monte Carlo transport is memory-bandwidth bound: most of the work is random
access into cross-section tables far larger than cache.  That makes the obvious
parallel choice -- many independent single-threaded processes -- not obviously
right, because every process holds *its own copy* of those tables, while OpenMP
threads inside one process share one copy.  Which wins is an empirical question
about a specific machine, and guessing it wrong costs hours.

It also matters that campaign points sit at temperatures *between* the tabulated
library values, so every lookup interpolates between two datasets.  Calibrating
on the reference point (exactly 600 K, no interpolation) would flatter the
estimate, so this measures at an interpolated temperature.

Reports histories/second for each configuration, which is the number that
predicts campaign wall time.

Usage:
    python scripts/calibrate.py --particles 4000 --batches 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lattice_uq.params import DesignPoint  # noqa: E402
from lattice_uq.runner import check_cross_sections, run_point  # noqa: E402

# Deliberately between library temperatures (600/900/1200 K) so the cost of
# temperature interpolation is included, as it will be for most campaign points.
CALIBRATION_POINT = DesignPoint(
    enrichment=3.5, fuel_temperature=847.0, moderator_density=0.72, pitch=1.3
)


def _one(args) -> float:
    particles, batches, inactive, threads, seed = args
    result, _ = run_point(
        CALIBRATION_POINT, particles=particles, batches=batches,
        inactive=inactive, threads=threads, seed=seed,
    )
    return result.wall_time_s


def measure(particles: int, batches: int, inactive: int,
            processes: int, threads: int) -> dict:
    """Aggregate throughput for `processes` concurrent runs of `threads` each."""
    total_histories = particles * batches * processes
    payload = [(particles, batches, inactive, threads, 1 + i)
               for i in range(processes)]

    t0 = time.perf_counter()
    if processes == 1:
        walls = [_one(payload[0])]
    else:
        with ProcessPoolExecutor(max_workers=processes) as pool:
            walls = list(pool.map(_one, payload))
    elapsed = time.perf_counter() - t0

    return {
        "processes": processes,
        "threads_per_process": threads,
        "cores_used": processes * threads,
        "wall_s": elapsed,
        "mean_run_s": sum(walls) / len(walls),
        # The only number that matters for planning: total histories the
        # machine retires per second in this configuration.
        "histories_per_s": total_histories / elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--particles", type=int, default=4000)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--inactive", type=int, default=10)
    ap.add_argument("--configs", default="1x1,1x8,1x32,4x8,8x4,14x1,20x1",
                    help="comma-separated processes x threads")
    ap.add_argument("--out", default="results/calibration.json")
    args = ap.parse_args()

    check_cross_sections()
    print(f"[calibrate] {args.particles} particles x {args.batches} batches "
          f"per run, at {CALIBRATION_POINT.fuel_temperature:.0f} K "
          "(interpolated)\n")
    print(f"{'config':>10} {'cores':>6} {'wall s':>9} {'run s':>9} "
          f"{'histories/s':>13}")

    results = []
    for spec in args.configs.split(","):
        p, t = (int(v) for v in spec.lower().split("x"))
        r = measure(args.particles, args.batches, args.inactive, p, t)
        results.append(r)
        print(f"{spec:>10} {r['cores_used']:>6} {r['wall_s']:>9.1f} "
              f"{r['mean_run_s']:>9.1f} {r['histories_per_s']:>13,.0f}")

    best = max(results, key=lambda r: r["histories_per_s"])
    print(f"\n[calibrate] best: {best['processes']} process(es) x "
          f"{best['threads_per_process']} thread(s) -> "
          f"{best['histories_per_s']:,.0f} histories/s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {
            "point": CALIBRATION_POINT.as_dict(),
            "particles": args.particles,
            "batches": args.batches,
            "inactive": args.inactive,
            "configurations": results,
            "best": best,
        },
        indent=2,
    ))
    print(f"[calibrate] wrote {out}")

    # Campaign sizing, so the choice of settings is arithmetic rather than hope.
    print("\n[calibrate] estimated campaign wall time at the best config:")
    for particles in (5_000, 10_000, 20_000):
        for n_runs in (300, 440):
            hist = particles * 130 * n_runs
            hours = hist / best["histories_per_s"] / 3600
            print(f"   {n_runs:>3} runs x {particles:>6,} particles "
                  f"({particles * 100:,} active histories each): "
                  f"{hours:5.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
