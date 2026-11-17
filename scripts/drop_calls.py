"""Remove some calls from a run's log so that `scv elicit` runs them again.

    uv run python scripts/drop_calls.py <run_id> --model mistral_7b --mode pairwise

The whole log is first copied to ``results/calls_backup/<run_id>-<time>/``. Then every
shard is rewritten without the matching rows (shards are kept, even if empty, so shard
numbering is undisturbed). Run `scv elicit --config ...` afterwards: it skips everything
still in the log and redoes only what was removed.
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


def drop(run_id: str, model: str, mode: str, results: str = "results") -> int:
    log = Path(results) / "calls" / run_id
    shards = sorted(log.glob("part-*.parquet"))
    if not shards:
        raise SystemExit(f"no call log at {log}")
    backup = Path(results) / "calls_backup" / f"{run_id}-{datetime.now():%Y%m%d-%H%M%S}"
    shutil.copytree(log, backup)
    removed = 0
    for shard in shards:
        table = pq.read_table(shard)
        hit = pc.and_(pc.equal(table["model_key"], model), pc.equal(table["mode"], mode))
        n = pc.sum(hit).as_py() or 0
        if n:
            tmp = shard.with_suffix(".parquet.tmp")
            pq.write_table(table.filter(pc.invert(hit)), tmp)
            os.replace(tmp, shard)
            removed += n
    print(f"{run_id}: removed {removed} {mode} calls of {model}; backup in {backup}")
    return removed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_id", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["thermometer", "ranking", "pairwise"])
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    for run in args.run_id:
        drop(run, args.model, args.mode, args.results)
