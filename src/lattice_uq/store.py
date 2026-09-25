"""Results store for the transport campaign.

One JSON file per completed run, written atomically, plus a Parquet roll-up
for analysis.  The per-run files are what make the campaign resumable and
crash-tolerant: a run is either fully on disk or absent, never half-written,
and restarting the driver simply skips the keys it already finds.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from .params import DesignPoint
from .runner import RunResult

RUNS_SUBDIR = "runs"
SPECTRA_SUBDIR = "spectra"


def record_key(point: DesignPoint, seed: int | None = None) -> str:
    """Storage key for a run.

    Replicates share a design point but not a seed, so the seed has to be part
    of the key or the second replicate would silently overwrite the first.
    """
    return point.run_id if seed is None else f"{point.run_id}-s{seed}"


class ResultStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.runs_dir = self.root / RUNS_SUBDIR
        self.spectra_dir = self.root / SPECTRA_SUBDIR
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # -- membership ----------------------------------------------------

    def path_for(self, key: str) -> Path:
        return self.runs_dir / f"{key}.json"

    def has(self, key: str) -> bool:
        return self.path_for(key).is_file()

    def completed_keys(self) -> set[str]:
        return {p.stem for p in self.runs_dir.glob("*.json")}

    def pending(
        self, jobs: Iterable[tuple[DesignPoint, int | None]]
    ) -> list[tuple[DesignPoint, int | None]]:
        """Filter a job list down to what has not been run yet."""
        done = self.completed_keys()
        return [(p, s) for p, s in jobs if record_key(p, s) not in done]

    # -- writing -------------------------------------------------------

    def write(
        self,
        result: RunResult,
        tag: str = "train",
        spectrum: dict[str, np.ndarray] | None = None,
    ) -> Path:
        key = record_key(result.point, result.seed)
        payload = {
            "key": key,
            "tag": tag,
            **asdict(result.point),
            "run_id": result.run_id,
            "keff": result.keff,
            "keff_sigma": result.keff_sigma,
            "particles": result.particles,
            "batches": result.batches,
            "inactive": result.inactive,
            "seed": result.seed,
            "wall_time_s": result.wall_time_s,
            "openmc_version": result.openmc_version,
            "entropy": result.entropy,
            "tallies": result.tallies,
        }
        path = self.path_for(key)
        _atomic_write_json(path, payload)

        if spectrum is not None:
            self.spectra_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.spectra_dir / f"{key}.npz", **spectrum)
        return path

    # -- reading -------------------------------------------------------

    def iter_records(self) -> Iterator[dict]:
        for path in sorted(self.runs_dir.glob("*.json")):
            with open(path) as fh:
                yield json.load(fh)

    def to_frame(self) -> pd.DataFrame:
        """All completed runs as one dataframe, uncertainties included."""
        rows = []
        for rec in self.iter_records():
            row = {k: v for k, v in rec.items() if k not in ("entropy", "tallies")}
            row["keff_sigma_pcm"] = row["keff_sigma"] * 1e5
            row["n_entropy"] = len(rec.get("entropy", []))
            for k, v in (rec.get("tallies") or {}).items():
                if np.isscalar(v):
                    row[f"tally_{k}"] = v
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("key").reset_index(drop=True)
        # Only replicate runs carry an explicit seed, so the column mixes ints
        # and None and lands as object dtype -- which Parquet cannot type. A
        # nullable integer says "int, sometimes absent", which is the truth.
        if "seed" in df.columns:
            df["seed"] = df["seed"].astype("Int64")
        return df

    def entropy_for(self, key: str) -> list[float]:
        with open(self.path_for(key)) as fh:
            return json.load(fh).get("entropy", [])

    def spectrum_for(self, key: str) -> dict[str, np.ndarray]:
        with np.load(self.spectra_dir / f"{key}.npz") as z:
            return {k: z[k] for k in z.files}

    def export_parquet(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.root / "results.parquet"
        df = self.to_frame()
        if df.empty:
            raise RuntimeError(f"No completed runs under {self.runs_dir}")
        df.to_parquet(path, index=False)
        return path


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write to a sibling temp file then rename.

    os.replace is atomic on both POSIX and Windows, so a crash mid-write
    leaves the previous state intact rather than a truncated JSON file that
    would poison every later read.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
