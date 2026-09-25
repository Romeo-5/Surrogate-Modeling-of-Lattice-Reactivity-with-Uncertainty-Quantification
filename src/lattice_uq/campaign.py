"""Resumable, parallel driver for the transport campaign.

Each design point is an independent eigenvalue calculation, so the campaign is
embarrassingly parallel in principle.  In practice it is not: Monte Carlo
transport is memory-bandwidth bound, and the right shape of the parallelism is
an empirical question, not an obvious one.

`scripts/calibrate.py` measures it.  On the machine this was developed on (32
cores), neither extreme wins:

    1 process  x  1 thread    12,777 histories/s
    1 process  x 32 threads   61,710
    8 processes x 4 threads   94,382   <- best
    14 processes x 1 thread   81,558
    20 processes x 1 thread   81,238

Threads inside one process share a single copy of the cross-section tables;
separate processes each hold their own, so past a handful of workers the
machine runs out of memory bandwidth before it runs out of cores -- 20 cores in
20 processes beat one core by only 6.4x.  The default is therefore a *hybrid*:
several processes, a few threads each.  Run the calibration before committing
to a long campaign on unfamiliar hardware; the difference here was a factor of
1.5 in total wall time.

Resumption is by presence in the store, not by a checkpoint file, so killing
the driver at any moment and restarting it is always safe.
"""

from __future__ import annotations

import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .params import DesignPoint
from .runner import RunResult, check_cross_sections, run_point
from .store import ResultStore, record_key

Job = tuple[DesignPoint, int | None]


@dataclass
class CampaignConfig:
    particles: int = 10_000
    batches: int = 130
    inactive: int = 30
    workers: int = 8
    threads_per_run: int = 1
    with_tallies: bool = False
    tag: str = "train"


def _worker(args) -> tuple[str, RunResult | None, dict | None, str | None]:
    """Run one point in a subprocess; never raise across the pool boundary.

    A single bad design point (an unreadable temperature, a data gap) should
    cost one row, not the whole campaign, so failures come back as a message
    to be logged by the parent.
    """
    point, seed, cfg = args
    key = record_key(point, seed)
    # One OpenMP thread per process: the parallelism is across runs.
    os.environ["OMP_NUM_THREADS"] = str(cfg.threads_per_run)
    try:
        result, spectrum = run_point(
            point,
            particles=cfg.particles,
            batches=cfg.batches,
            inactive=cfg.inactive,
            seed=seed,
            threads=cfg.threads_per_run,
            with_tallies=cfg.with_tallies,
        )
        return key, result, spectrum, None
    except Exception:
        return key, None, None, traceback.format_exc(limit=6)


def run_campaign(
    jobs: list[Job],
    store: ResultStore,
    cfg: CampaignConfig,
    log: Path | None = None,
) -> dict[str, int]:
    """Execute every job not already in the store.

    Results are written as each run finishes, so progress survives an
    interrupt without any explicit checkpointing.
    """
    check_cross_sections()

    todo = store.pending(jobs)
    skipped = len(jobs) - len(todo)
    print(
        f"[campaign] {len(jobs)} jobs, {skipped} already complete, "
        f"{len(todo)} to run on {cfg.workers} workers",
        flush=True,
    )
    if not todo:
        return {"submitted": 0, "completed": 0, "failed": 0, "skipped": skipped}

    payloads = [(p, s, cfg) for p, s in todo]
    completed = failed = 0
    log_fh = open(log, "a") if log else None

    try:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures = {pool.submit(_worker, pl): pl[0] for pl in payloads}
            for i, fut in enumerate(as_completed(futures), start=1):
                key, result, spectrum, error = fut.result()
                if error is not None:
                    failed += 1
                    msg = f"[campaign] FAILED {key}\n{error}"
                    print(msg, file=sys.stderr, flush=True)
                    if log_fh:
                        log_fh.write(msg + "\n")
                        log_fh.flush()
                    continue
                store.write(result, tag=cfg.tag, spectrum=spectrum)
                completed += 1
                print(
                    f"[campaign] {i}/{len(payloads)} {key} "
                    f"k={result.keff:.5f} +/- {result.keff_sigma * 1e5:.0f} pcm "
                    f"({result.wall_time_s:.1f} s)",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n[campaign] interrupted; completed runs are saved. "
              "Re-run the same command to resume.", file=sys.stderr)
    finally:
        if log_fh:
            log_fh.close()

    return {
        "submitted": len(payloads),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
    }
