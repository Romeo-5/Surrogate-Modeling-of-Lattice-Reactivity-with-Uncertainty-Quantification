#!/usr/bin/env python
"""Phase 2 -- run the parameter sweep.

Three designs are executed into one store:

    train       Latin hypercube, the surrogate's training data
    test        independent Sobol sequence, held out for evaluation
    replicate   a few points repeated under different RNG seeds, to measure
                the transport calculation's run-to-run scatter directly

The driver is resumable: it skips anything already in the store, so it can be
interrupted and restarted, run in stages, or extended with more points later
without recomputing anything.

Usage:
    python scripts/run_campaign.py --n-train 320 --n-test 128 --workers 28
    python scripts/run_campaign.py --n-train 320 --n-test 128 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lattice_uq.campaign import CampaignConfig, run_campaign  # noqa: E402
from lattice_uq.sampling import (  # noqa: E402
    coverage_report,
    latin_hypercube,
    replicate_design,
    sobol_design,
)
from lattice_uq.store import ResultStore, record_key  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-train", type=int, default=320)
    # Power of two: Sobol's balance properties require it.
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--replicate-points", type=int, default=5)
    ap.add_argument("--replicate-seeds", type=int, default=8)
    ap.add_argument("--particles", type=int, default=10_000)
    ap.add_argument("--batches", type=int, default=130)
    ap.add_argument("--inactive", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    # Hybrid, not pure multiprocessing: see scripts/calibrate.py. Each process
    # holds its own copy of the cross-section tables, so past a handful of
    # workers the machine runs out of memory bandwidth before it runs out of
    # cores, and threads inside a process (which share one copy) do better.
    ap.add_argument("--threads-per-run", type=int, default=4)
    ap.add_argument("--with-tallies", action="store_true",
                    help="also store the energy spectrum for every run (large)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and report the designs without running OpenMC")
    ap.add_argument("--only", choices=["train", "test", "replicate"], default=None)
    args = ap.parse_args()

    store = ResultStore(args.data_dir)

    train = [(p, None) for p in latin_hypercube(args.n_train)]
    test = [(p, None) for p in sobol_design(args.n_test)]
    reps = replicate_design(args.replicate_points, args.replicate_seeds)

    designs = {"train": train, "test": test, "replicate": reps}
    if args.only:
        designs = {args.only: designs[args.only]}

    # Sobol and LHS are independent, so a test point can land on a training
    # point.  Keys are content-addressed, so the collision would silently make
    # a "held-out" point one the surrogate trained on.  Drop them.
    train_keys = {record_key(p, s) for p, s in train}
    if "test" in designs:
        before = len(designs["test"])
        designs["test"] = [
            (p, s) for p, s in designs["test"]
            if record_key(p, s) not in train_keys
        ]
        dropped = before - len(designs["test"])
        if dropped:
            print(f"[campaign] dropped {dropped} test point(s) colliding with train")

    summary = {
        "designs": {k: len(v) for k, v in designs.items()},
        "coverage": {
            k: coverage_report([p for p, _ in v]) for k, v in designs.items()
        },
        "settings": {
            "particles": args.particles,
            "batches": args.batches,
            "inactive": args.inactive,
            "active_batches": args.batches - args.inactive,
            "histories_per_run": args.particles * (args.batches - args.inactive),
            "workers": args.workers,
            "threads_per_run": args.threads_per_run,
        },
    }
    print(json.dumps(summary["designs"], indent=2))
    print(f"[campaign] {summary['settings']['histories_per_run']:,} active "
          f"histories per run")

    if args.dry_run:
        print("[campaign] dry run; nothing executed")
        for tag, cov in summary["coverage"].items():
            print(f"\n-- {tag} coverage")
            for name, c in cov.items():
                print(f"   {name:<20} [{c['min']:.4g}, {c['max']:.4g}] "
                      f"within [{c['bound_low']:.4g}, {c['bound_high']:.4g}]")
        return 0

    totals = {}
    t0 = time.perf_counter()
    for tag, jobs in designs.items():
        cfg = CampaignConfig(
            particles=args.particles,
            batches=args.batches,
            inactive=args.inactive,
            workers=args.workers,
            threads_per_run=args.threads_per_run,
            with_tallies=args.with_tallies,
            tag=tag,
        )
        print(f"\n[campaign] === {tag} ({len(jobs)} points) ===")
        totals[tag] = run_campaign(jobs, store, cfg,
                                   log=Path(args.data_dir) / "failures.log")
    elapsed = time.perf_counter() - t0

    summary["totals"] = totals
    summary["wall_time_s"] = elapsed
    Path(args.data_dir, "campaign_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    df = store.to_frame()
    print(f"\n[campaign] {len(df)} runs in store, {elapsed / 60:.1f} min wall time")
    if not df.empty:
        parquet = store.export_parquet()
        print(f"[campaign] k range {df.keff.min():.5f} - {df.keff.max():.5f}")
        print(f"[campaign] mean sigma {df.keff_sigma_pcm.mean():.0f} pcm")
        print(f"[campaign] wrote {parquet}")

    failed = sum(t["failed"] for t in totals.values())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
