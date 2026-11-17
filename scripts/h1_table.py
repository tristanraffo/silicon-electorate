"""Line up the H1 rows of ANALYSIS_PLAN.md §3 across analysis variants.

    uv run python scripts/h1_table.py anes2024_ts_full_v4 anes2016_ts_full_v4 ...

For each run, model and outcome (``sp_deletion_share_minus_human``, the primary one, and
``pc1_share_minus_human``), thermometer ballots, weighted: the paired difference and its
95% interval in the default analysis and in the ``__ties_break_random`` and
``__ties_drop_flat`` variants (DECISIONS G21), whether each interval excludes zero, and
whether all available variants agree in sign (``same_sign``), and whether every variant's
interval excludes zero in that same direction (``robust``, the rule proposed in
ANALYSIS_PLAN §6). It reports; it does not decide.

Note: ``break_random`` leaves ``pc1_share`` unchanged by construction (breaking a tie
adds almost no variance), so for that outcome only ``drop_flat`` is a real check. Missing
variants are shown as blank. ``--weighting unweighted`` and ``--source ranking`` give
the robustness views.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = {"default": "", "break_random": "__ties_break_random", "drop_flat": "__ties_drop_flat"}
OUTCOMES = ("sp_deletion_share_minus_human", "pc1_share_minus_human")


def h1_table(
    run_ids: list[str],
    results: str = "results",
    weighting: str = "weighted",
    source: str = "thermometer",
) -> pd.DataFrame:
    rows = []
    for run in run_ids:
        for variant, suffix in VARIANTS.items():
            path = Path(results) / "metrics" / f"{run}{suffix}" / "summary.csv"
            if not path.exists():
                continue
            s = pd.read_csv(path)
            s = s[
                s["metric"].isin(OUTCOMES)
                & (s["weighting"] == weighting)
                & (s["source"] == source)
                & (s["kind"] == "homogeneous")
            ]
            for r in s.itertuples():
                rows.append(
                    {
                        "run": run,
                        "outcome": r.metric.removesuffix("_minus_human"),
                        "model": r.electorate,
                        "variant": variant,
                        "diff": r.value,
                        "ci": f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]",
                        "excl0": bool(r.ci_low > 0 or r.ci_high < 0),
                        "n": int(r.n_voters),
                    }
                )
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    wide = long.pivot_table(
        index=["run", "outcome", "model"],
        columns="variant",
        values=["diff", "ci", "excl0"],
        aggfunc="first",
    )
    signs = np.sign(
        long.pivot_table(index=["run", "outcome", "model"], columns="variant", values="diff")
    )
    excl = long.pivot_table(
        index=["run", "outcome", "model"], columns="variant", values="excl0", aggfunc="first"
    )
    same = signs.apply(lambda r: r.dropna().nunique() == 1, axis=1)
    wide[("same_sign", "all")] = same
    # the plan's robustness rule: every variant's interval excludes zero, same direction
    wide[("robust", "all")] = same & excl.fillna(False).astype(bool).all(axis=1)
    order = [v for v in VARIANTS if v in long["variant"].unique()]
    cols = [(k, v) for v in order for k in ("diff", "ci", "excl0")]
    cols += [("same_sign", "all"), ("robust", "all")]
    return wide[cols]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_id", nargs="+")
    ap.add_argument("--results", default="results")
    ap.add_argument("--weighting", default="weighted", choices=["weighted", "unweighted"])
    ap.add_argument("--source", default="thermometer", choices=["thermometer", "ranking"])
    args = ap.parse_args()
    table = h1_table(args.run_id, args.results, args.weighting, args.source)
    with pd.option_context(
        "display.width", 250, "display.max_columns", 30, "display.float_format", "{:+.3f}".format
    ):
        print(f"weighting={args.weighting} source={args.source}; diff = model minus human")
        print(table.to_string())
