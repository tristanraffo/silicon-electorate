"""Compare ranking prompts after a GPU trial.

    uv run python scripts/ranking_report.py anes2024_pilot_v3 rank_trial_strict ...

For each run and model: parse rate of the greedy ranking (the ballot the analysis uses)
and of the sampled ones, and three position-bias measures on the parsed greedy rankings.
The order shown is random per call, so with no position bias:

- ``same_as_shown`` (ranking identical to the order shown) is about n_ok / n!;
- ``first_top`` (share that put the first-listed candidate first) is about 1/n;
- ``pos_corr`` (mean Spearman correlation between list position and rank) is about 0,
  and ``pos_z`` (that mean over its standard error) is within about +-2.

A positive ``pos_corr`` means earlier-listed candidates are ranked higher; negative means
the reverse. Then the share of rankings with a tie, and a few failures.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scv.elicit import GREEDY, ranking_table, unreadable_pairwise
from scv.runner import read_calls


def report(run_id: str, results: str = "results", examples: int = 4) -> None:
    calls = read_calls(results, run_id)
    table = ranking_table(calls)
    if table.empty:
        print(f"\n== {run_id}: no ranking calls")
        return
    table["greedy"] = table["sample_idx"] == GREEDY
    rows = []
    for model, g in table.groupby("model_key", sort=True):
        greedy, ok = g[g["greedy"]], g[g["greedy"] & g["parse_ok"]]
        groups = [[list(x) for x in b.groups] for b in ok["ballot"]]
        shown = [[[c] for c in s] for s in ok["candidates_shown"]]
        corr, first = [], []
        for ballot, order in zip(ok["ballot"], ok["candidates_shown"], strict=True):
            rank = ballot.rank_of()
            ranks = [rank[c] for c in order]
            first.append(ranks[0] == min(ranks))
            if len(set(ranks)) > 1:
                corr.append(spearmanr(range(len(order)), ranks).statistic)
        corr = np.asarray(corr)
        se = corr.std(ddof=1) / np.sqrt(len(corr)) if len(corr) > 1 else float("nan")
        rows.append(
            {
                "model": model,
                "greedy_ok": greedy["parse_ok"].mean(),
                "sampled_ok": g[~g["greedy"]]["parse_ok"].mean(),
                "n_ok": len(ok),
                "same_as_shown": sum(a == b for a, b in zip(groups, shown, strict=True)),
                "first_top": float(np.mean(first)) if first else float("nan"),
                "pos_corr": float(corr.mean()) if len(corr) else float("nan"),
                "pos_z": float(corr.mean() / se) if len(corr) > 1 and se > 0 else float("nan"),
                "with_tie": sum(any(len(x) > 1 for x in b) for b in groups),
            }
        )
    n = len(table["candidates_shown"].iloc[0])
    print(
        f"\n== {run_id}  ({n} candidates: chance first_top {1 / n:.3f}, "
        f"same_as_shown 1 in {math.factorial(n)})"
    )
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    pair = calls[calls["mode"] == "pairwise"]
    if len(pair):
        # Failed calls, plus (in older logs) answers recorded as exactly 50/50; see G19.
        share = unreadable_pairwise(pair).groupby(pair["model_key"]).mean().round(3)
        print("  pairwise answers unreadable (left out of H4; should be small):", share.to_dict())
    failed = calls[(calls["mode"] == "ranking") & (~calls["parse_ok"])]
    for model, g in failed.groupby("model_key", sort=True):
        for idx, text in g[["sample_idx", "raw_text"]].head(examples).itertuples(index=False):
            print(f"  {model} {'greedy' if idx == GREEDY else 'sampled'} | {text!r}")


if __name__ == "__main__":
    for run in sys.argv[1:] or ["anes2024_pilot_v3"]:
        report(run)
