"""Metrics, bootstrap, n-curves, matched comparisons and figures (SPEC §12).

``analyze`` turns a :class:`scv.electorates.BallotTable` into tidy tables;
``write_outputs`` saves them under ``results/metrics/<run_id>/`` together with one
``summary.csv``; the ``fig_*`` functions draw the three figures from those tables. The
notebook only reads the files and calls the ``fig_*`` functions.

Weighted metrics use the ANES survey weights and are the headline; the unweighted
versions are reported alongside (SPEC §14.3).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from scv import domains, rules
from scv.electorates import (
    SOURCES,
    BallotTable,
    Electorate,
    heterogeneous,
    homogeneous,
    human_electorate,
    null_ic,
    null_permutation,
)
from scv.elicit import derive_seed

logger = logging.getLogger(__name__)

N_SIZES = (11, 25, 51, 101, 251, 501)
WEIGHTINGS = ("weighted", "unweighted")


@dataclass(frozen=True)
class AnalysisOptions:
    bootstrap: int = 1000  # voter bootstrap replicates (B)
    draws: int = 500  # sub-electorates per size for the n-curves
    replicates: int = 50  # heterogeneous assignments / null draws (R)
    sizes: tuple[int, ...] = N_SIZES
    seed: int = 0
    curve_replicates: int = 10  # how many replicated electorates the n-curves cycle through


# --------------------------------------------------------------------------------------
# Fast batch statistics: many weightings of one electorate at once
# --------------------------------------------------------------------------------------


def _preference_matrix(ranks: np.ndarray) -> np.ndarray:
    """(n, m*m) float32: entry [i, a*m+b] is 1 if voter i strictly prefers a to b."""
    n, m = ranks.shape
    return (ranks[:, :, None] < ranks[:, None, :]).reshape(n, m * m).astype(np.float32)


def batch_majority(prefs: np.ndarray, counts: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """For each row of ``counts`` (a weight per voter), whether the weighted majority
    relation has a cycle and whether a Condorcet winner exists. Same definitions and
    tolerance as :mod:`scv.rules`."""
    totals = (counts.astype(np.float64) @ prefs.astype(np.float64)).reshape(len(counts), m, m)
    margin = totals - totals.transpose(0, 2, 1)
    tol = rules.RTOL * np.maximum(1.0, counts.sum(axis=1))[:, None, None]
    beats = margin > tol
    winner = (beats | np.eye(m, dtype=bool)).all(axis=2).any(axis=1)
    reach = beats.copy()
    for k in range(m):
        reach |= reach[:, :, [k]] & reach[:, [k], :]
    cycle = np.diagonal(reach, axis1=1, axis2=2).any(axis=1)
    return cycle, winner


def batch_sp_deletion(consistency: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """SP deletion share for each row of ``counts``, from a (n, n_axes) consistency matrix."""
    shares = counts.astype(np.float64) @ consistency.astype(np.float64)
    best = shares.max(axis=1) / counts.sum(axis=1)
    return np.where(best >= 1 - 1e-12, 0.0, 1.0 - best)


# --------------------------------------------------------------------------------------
# Point metrics for one electorate
# --------------------------------------------------------------------------------------


def point_metrics(
    e: Electorate,
    candidates: Sequence[str],
    weighted: bool = True,
    *,
    full: bool = True,
    seed: int = 0,
    consistency: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
    """All SPEC §5-§6 numbers for one electorate, plus each rule's winners.

    ``full=False`` skips the slow extras (plateaued single-peakedness, the IRV
    monotonicity probe); it is used for the replicated electorates.
    """
    profile = e.profile(candidates, weighted)
    w = profile.weights
    if consistency is None:
        consistency = domains.sp_consistency(e.profile(candidates, weighted=False))
    cycles = rules.majority_cycles(profile)
    results = rules.run_all_rules(profile, seed=seed)
    cw = rules.condorcet_winner(profile)
    vr = domains.value_restriction(profile)
    div = domains.diversity(profile)
    ranked = ~np.isnan(e.ranks)
    strict = [len(set(row[ok])) == ok.sum() for row, ok in zip(e.ranks, ranked, strict=True)]

    out: dict[str, float] = {
        "has_cycle": float(cycles.has_cycle),
        "condorcet_winner_exists": float(cw is not None),
        "condorcet_loser_exists": float(rules.condorcet_loser(profile) is not None),
        "majority_transitive": float(rules.condorcet_order(profile) is not None),
        "smith_size": float(cycles.smith_size),
        "n_three_cycles": float(len(cycles.three_cycles)),
        "rule_disagreement_share": rules.rule_disagreement(results).share,
        "borda_differs_from_condorcet": (
            math.nan if cw is None else float(results["borda"].winners != (cw,))
        ),
        "sp_deletion_share": float(batch_sp_deletion(consistency, w[None, :])[0]),
        "vr_share": vr.share,
        "vr_mean_min_violation": vr.mean_min_violation,
        "distinct_share": div.distinct_share,
        "ballot_entropy": div.entropy,
        "mean_kendall_tau": div.mean_kendall_tau,
        "share_strict_ballots": float(np.average(strict, weights=w)),
        "share_complete_ballots": float(np.average(ranked.all(axis=1), weights=w)),
        "pc1_share": math.nan,
        "pc2_share": math.nan,
    }
    if e.scores is not None:
        dim = domains.dimensionality(e.scores, w)
        out["pc1_share"], out["pc2_share"] = dim.pc1_share, dim.pc2_share
    if full:
        out["sp_plateau_deletion_share"] = domains.sp_deletion_share(profile, "plateaued")
        out["irv_monotonicity_failure_fraction"] = rules.irv_monotonicity_probe(
            profile, seed=seed
        ).failure_fraction
    return out, {rule: res.winners for rule, res in results.items()}


# --------------------------------------------------------------------------------------
# Bootstrap and n-curves
# --------------------------------------------------------------------------------------


def bootstrap(
    e: Electorate, weighted: bool, b: int, rng: np.random.Generator, consistency: np.ndarray
) -> dict[str, np.ndarray]:
    """Voter bootstrap: resample voters with replacement, keeping each voter's weight."""
    n, m = e.ranks.shape
    w = e.weights if weighted else np.ones(n)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=b) * w[None, :]
    cycle, winner = batch_majority(_preference_matrix(e.ranks), counts, m)
    out = {
        "has_cycle": cycle.astype(float),
        "condorcet_winner_exists": winner.astype(float),
        "sp_deletion_share": batch_sp_deletion(consistency, counts),
    }
    if e.scores is not None:
        out["pc1_share"] = np.array([domains.dimensionality(e.scores, c).pc1_share for c in counts])
    return out


def _sample_counts(
    weights: np.ndarray, size: int, draws: int, weighted: bool, rng: np.random.Generator
) -> np.ndarray:
    """(draws, n) 0/1 matrix: each row picks ``size`` distinct voters, with probability
    proportional to weight if ``weighted``, uniformly otherwise. Inside a sub-electorate
    every voter counts once."""
    n = len(weights)
    p = weights / weights.sum() if weighted else None
    counts = np.zeros((draws, n), dtype=np.float32)
    for d in range(draws):
        counts[d, rng.choice(n, size=size, replace=False, p=p)] = 1.0
    return counts


def n_curve(
    electorates: Sequence[Electorate],
    consistencies: Sequence[np.ndarray],
    weighted: bool,
    opts: AnalysisOptions,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Cycle frequency, Condorcet-winner frequency and SP deletion share for random
    sub-electorates of each size (SPEC §12). With several electorates (replicates of a
    heterogeneous or null electorate) the draws are split evenly between them. Sizes
    larger than the electorate are skipped."""
    rows = []
    m = electorates[0].ranks.shape[1]
    prefs = [_preference_matrix(e.ranks) for e in electorates]
    for size in opts.sizes:
        if size > electorates[0].n:
            continue
        cycle, winner, deletion = [], [], []
        for k, e in enumerate(electorates):
            share = len(range(k, opts.draws, len(electorates)))
            if share == 0:
                continue
            counts = _sample_counts(e.weights, size, share, weighted, rng)
            c, w = batch_majority(prefs[k], counts, m)
            cycle.append(c)
            winner.append(w)
            deletion.append(batch_sp_deletion(consistencies[k], counts))
        cycle, winner, deletion = map(np.concatenate, (cycle, winner, deletion))
        rows.append(
            {
                "n": size,
                "draws": len(cycle),
                "cycle_freq": float(cycle.mean()),
                "condorcet_winner_freq": float(winner.mean()),
                "sp_deletion_mean": float(deletion.mean()),
                "sp_deletion_sd": float(deletion.std(ddof=0)),
            }
        )
    return rows


# --------------------------------------------------------------------------------------
# Matched human-LLM comparisons
# --------------------------------------------------------------------------------------


def _tau_b(x: np.ndarray, y: np.ndarray) -> float:
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 2 or len(set(x[ok])) < 2 or len(set(y[ok])) < 2:
        return math.nan
    return float(kendalltau(x[ok], y[ok], variant="b").statistic)


def _top_agree(x: np.ndarray, y: np.ndarray) -> float:
    """1 if the two ballots share a top-ranked candidate (ties count), NaN if either is empty."""
    if np.isnan(x).all() or np.isnan(y).all():
        return math.nan
    return float(bool(np.any((x == np.nanmax(x)) & (y == np.nanmax(y)))))


def matched_comparisons(table: BallotTable) -> pd.DataFrame:
    """Per respondent and model: Kendall tau-b between the human and LLM ballots,
    top-choice agreement and mean absolute thermometer error (SPEC §12).

    ``*_thermometer`` compares the human thermometers with the model's thermometers,
    ``*_ranking`` with the model's greedy ranking. Candidates either side left unrated
    are skipped; tau-b is NaN when either ballot is constant on the shared candidates.
    """
    frames = []
    for model in table.models:
        scores = table.llm_scores.get(model)
        ranks = table.llm_ranks.get(model)
        rows: dict[str, list[float]] = {
            k: []
            for k in (
                "tau_b_thermometer",
                "top_agree_thermometer",
                "abs_error",
                "tau_b_ranking",
                "top_agree_ranking",
            )
        }
        for i in range(table.n):
            h = table.human[i]
            if scores is not None:
                rows["tau_b_thermometer"].append(_tau_b(h, scores[i]))
                rows["top_agree_thermometer"].append(_top_agree(h, scores[i]))
                both = ~np.isnan(h) & ~np.isnan(scores[i])
                rows["abs_error"].append(
                    float(np.abs(h[both] - scores[i][both]).mean()) if both.any() else math.nan
                )
            if ranks is not None:
                rows["tau_b_ranking"].append(_tau_b(h, -ranks[i]))
                rows["top_agree_ranking"].append(_top_agree(h, -ranks[i]))
        frame = pd.DataFrame({k: v for k, v in rows.items() if v})
        frame.insert(0, "model", model)
        frame.insert(1, "respondent_id", table.respondent_ids)
        frame.insert(2, "weight", table.weights)
        for var in ("party_id_7", "ideology_7"):
            frame[var] = table.personas[var].to_numpy() if var in table.personas else None
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


MATCHED_METRICS = (
    "tau_b_thermometer",
    "top_agree_thermometer",
    "abs_error",
    "tau_b_ranking",
    "top_agree_ranking",
)


def matched_groups(matched: pd.DataFrame) -> pd.DataFrame:
    """Weighted and unweighted means of the matched comparisons, overall and broken
    down by party ID and ideology. Long format: one row per model, group and metric."""
    rows = []
    if matched.empty:
        return pd.DataFrame()
    for model, g in matched.groupby("model", sort=False):
        splits = [("all", "all", g)]
        for var in ("party_id_7", "ideology_7"):
            labelled = g.assign(**{var: g[var].fillna("(missing)")})
            splits += [(var, str(level), sub) for level, sub in labelled.groupby(var, sort=True)]
        for group_var, group, sub in splits:
            for metric in MATCHED_METRICS:
                if metric not in sub:
                    continue
                ok = sub[metric].notna()
                if not ok.any():
                    continue
                values, weights = sub.loc[ok, metric], sub.loc[ok, "weight"]
                rows.append(
                    {
                        "model": model,
                        "group_var": group_var,
                        "group": group,
                        "metric": metric,
                        "weighted_mean": float(np.average(values, weights=weights)),
                        "unweighted_mean": float(values.mean()),
                        "n": int(ok.sum()),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# The whole analysis
# --------------------------------------------------------------------------------------

BOOTSTRAPPED = ("sp_deletion_share", "pc1_share")
BOOT_FREQ = {
    "has_cycle": "has_cycle_boot_freq",
    "condorcet_winner_exists": "condorcet_winner_boot_freq",
}


def _rng(opts: AnalysisOptions, *parts: Any) -> np.random.Generator:
    return np.random.default_rng(derive_seed("analysis", opts.seed, *parts))


def analyze(
    table: BallotTable,
    run_id: str,
    pool: Sequence[str] | None = None,
    opts: AnalysisOptions | None = None,
) -> dict[str, pd.DataFrame]:
    """Every metric for every electorate of one run.

    Returns tables ``metrics`` (tidy: one row per electorate x source x weighting x
    metric), ``ncurves``, ``winners``, ``matched`` and ``matched_groups``.

    - human and homogeneous electorates: point value plus a voter-bootstrap interval;
    - heterogeneous and null electorates: mean over ``replicates`` draws, with the
      spread across draws (``sd``, and the 2.5-97.5 percentile range as the interval).

    ``pool`` is the set of models mixed in the heterogeneous electorate (default: all
    models in the table); it needs at least two.
    """
    opts = opts or AnalysisOptions()
    cands = table.candidates
    pool = [m for m in (pool if pool is not None else table.models) if m in table.models]
    base = {"run_id": run_id, "election": table.election}
    metrics: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []

    def ident(e: Electorate, weighting: str) -> dict[str, Any]:
        return {
            **base,
            "electorate": e.name,
            "kind": e.kind,
            "model": e.model,
            "source": e.source,
            "weighting": weighting,
            "n_voters": e.n,
            "n_candidates": len(cands),
        }

    # -- fixed electorates: human, and each model on its own ---------------------------
    fixed = [human_electorate(table)]
    fixed += [homogeneous(table, m, "thermometer") for m in table.llm_scores]
    fixed += [homogeneous(table, m, "ranking") for m in table.llm_ranks]
    for e in fixed:
        consistency = domains.sp_consistency(e.profile(cands, weighted=False))
        for weighting in WEIGHTINGS:
            weighted = weighting == "weighted"
            values, rule_winners = point_metrics(
                e, cands, weighted, seed=opts.seed, consistency=consistency
            )
            boot = (
                bootstrap(
                    e,
                    weighted,
                    opts.bootstrap,
                    _rng(opts, "boot", e.name, e.source, weighting),
                    consistency,
                )
                if opts.bootstrap > 0
                else {}
            )
            for metric, value in values.items():
                row = {**ident(e, weighting), "metric": metric, "value": value}
                if metric in BOOTSTRAPPED and metric in boot:
                    lo, hi = np.nanpercentile(boot[metric], [2.5, 97.5])
                    row.update(
                        ci_low=float(lo), ci_high=float(hi), sd=float(np.nanstd(boot[metric])),
                        interval="bootstrap",
                    )  # fmt: skip
                metrics.append(row)
            for metric, name in BOOT_FREQ.items():
                if metric in boot:
                    freq = float(boot[metric].mean())
                    metrics.append({**ident(e, weighting), "metric": name, "value": freq})
            for rule, names in rule_winners.items():
                winners.append({**ident(e, weighting), "rule": rule, "winners": "|".join(names)})
            for row in n_curve(
                [e], [consistency], weighted, opts, _rng(opts, "curve", e.name, e.source, weighting)
            ):
                curves.append({**ident(e, weighting), **row})

    # -- replicated electorates: heterogeneous mixes and the two nulls ------------------
    makers: list[tuple[str, str, Any]] = []
    if len(pool) >= 2:
        for source in SOURCES:
            have = table.llm_scores if source == "thermometer" else table.llm_ranks
            if all(m in have for m in pool):
                makers.append(
                    (
                        "heterogeneous",
                        source,
                        lambda rng, r, s=source: heterogeneous(table, pool, s, rng, r),
                    )
                )
    else:
        logger.warning("heterogeneous electorate skipped: it needs at least two pool models")
    makers.append(("null_ic", "thermometer", lambda rng, r: null_ic(table, rng, r)))
    makers.append(
        ("null_permutation", "thermometer", lambda rng, r: null_permutation(table, rng, r))
    )

    for name, source, make in makers:
        if opts.replicates <= 0:
            continue
        per_weighting: dict[str, list[dict[str, float]]] = {w: [] for w in WEIGHTINGS}
        kept: list[tuple[Electorate, np.ndarray]] = []
        for r in range(opts.replicates):
            e = make(_rng(opts, "replicate", name, source, r), r)
            consistency = domains.sp_consistency(e.profile(cands, weighted=False))
            if r < opts.curve_replicates:
                kept.append((e, consistency))
            for weighting in WEIGHTINGS:
                values, _ = point_metrics(
                    e, cands, weighting == "weighted", full=False, seed=opts.seed,
                    consistency=consistency,
                )  # fmt: skip
                per_weighting[weighting].append(values)
        example = kept[0][0]
        for weighting in WEIGHTINGS:
            frame = pd.DataFrame(per_weighting[weighting])
            for metric in frame.columns:
                col = frame[metric].to_numpy(dtype=float)
                if np.isnan(col).all():
                    stats = dict(value=math.nan)
                else:
                    lo, hi = np.nanpercentile(col, [2.5, 97.5])
                    stats = dict(
                        value=float(np.nanmean(col)), ci_low=float(lo), ci_high=float(hi),
                        sd=float(np.nanstd(col)), interval="replicates",
                    )  # fmt: skip
                metrics.append({**ident(example, weighting), "metric": metric, **stats})
            for row in n_curve(
                [e for e, _ in kept],
                [c for _, c in kept],
                weighting == "weighted",
                opts,
                _rng(opts, "curve", name, source, weighting),
            ):
                curves.append({**ident(example, weighting), **row})

    # -- matched comparisons ------------------------------------------------------------
    matched = matched_comparisons(table)
    groups = matched_groups(matched)
    if len(groups):
        for row in groups[groups["group_var"] == "all"].itertuples():
            source = "ranking" if row.metric.endswith("_ranking") else "thermometer"
            for weighting, value in (
                ("weighted", row.weighted_mean),
                ("unweighted", row.unweighted_mean),
            ):
                metrics.append(
                    {
                        **base,
                        "electorate": row.model,
                        "kind": "homogeneous",
                        "model": row.model,
                        "source": source,
                        "weighting": weighting,
                        "n_voters": row.n,
                        "n_candidates": len(cands),
                        "metric": f"matched_{row.metric}",
                        "value": value,
                    }
                )

    columns = [
        "run_id", "election", "electorate", "kind", "model", "source", "weighting", "metric",
        "value", "ci_low", "ci_high", "sd", "interval", "n_voters", "n_candidates",
    ]  # fmt: skip
    return {
        "metrics": pd.DataFrame(metrics).reindex(columns=columns),
        "ncurves": pd.DataFrame(curves),
        "winners": pd.DataFrame(winners),
        "matched": matched,
        "matched_groups": groups,
    }


H4_METRICS = (
    "wst_violation_share",
    "any_wst_violation",
    "strict_cycle_share",
    "any_strict_cycle",
    "ranking_agreement",
    "abs_position_bias",
)


def voter_coherence(calls: pd.DataFrame, candidates: Sequence[str]) -> pd.DataFrame:
    """H4, per model and respondent (SPEC §10.3): share of candidate triples violating
    weak stochastic transitivity, share forming a strict 3-cycle, agreement between the
    pairwise choices and the voter's own greedy ranking, and mean absolute position bias.
    Empty if the run has no pairwise calls."""
    from scv.elicit import GREEDY, pairwise_coherence, pairwise_table, ranking_table

    if not (calls["mode"] == "pairwise").any():
        return pd.DataFrame()
    pairs = pairwise_table(calls)
    ranks = ranking_table(calls)
    ranks = ranks[ranks["sample_idx"] == GREEDY]
    ballots = {(r.model_key, r.respondent_id): r.ballot for r in ranks.itertuples()}
    rows = []
    for (model, rid), g in pairs.groupby(["model_key", "respondent_id"], sort=False):
        p = {(a, b): v for a, b, v in zip(g["a"], g["b"], g["p_a_over_b"], strict=True)}
        present = [c for c in candidates if any(c in pair for pair in p)]
        out = pairwise_coherence(p, present, ballots.get((model, rid)))
        rows.append(
            {
                "model": model,
                "respondent_id": rid,
                "wst_violation_share": out["wst_violation_share"],
                "any_wst_violation": float(out["wst_violations"] > 0),
                "strict_cycle_share": out["strict_cycle_share"],
                "any_strict_cycle": float(out["strict_cycles"] > 0),
                "ranking_agreement": out["ranking_agreement"],
                "abs_position_bias": float(g["position_bias"].abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def coherence_metrics(
    coherence: pd.DataFrame, table: BallotTable, run_id: str
) -> list[dict[str, Any]]:
    """Per-model means of the H4 measures, weighted and unweighted, as summary rows
    (``source = "pairwise"``, metric names prefixed ``h4_``)."""
    rows: list[dict[str, Any]] = []
    if coherence.empty:
        return rows
    weight = dict(zip(table.respondent_ids, table.weights, strict=True))
    for model, g in coherence.groupby("model", sort=False):
        g = g[g["respondent_id"].isin(weight)]
        w = g["respondent_id"].map(weight).to_numpy(dtype=float)
        for metric in H4_METRICS:
            ok = g[metric].notna().to_numpy()
            if not ok.any():
                continue
            values = g[metric].to_numpy(dtype=float)[ok]
            for weighting, value in (
                ("weighted", float(np.average(values, weights=w[ok]))),
                ("unweighted", float(values.mean())),
            ):
                rows.append(
                    {
                        "run_id": run_id,
                        "election": table.election,
                        "electorate": model,
                        "kind": "homogeneous",
                        "model": model,
                        "source": "pairwise",
                        "weighting": weighting,
                        "metric": f"h4_{metric}",
                        "value": value,
                        "n_voters": int(ok.sum()),
                        "n_candidates": len(table.candidates),
                    }
                )
    return rows


def metrics_dir(results_dir: str | Path, run_id: str) -> Path:
    return Path(results_dir) / "metrics" / run_id


def write_outputs(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> Path:
    """Write every table as Parquet, plus the single tidy ``summary.csv`` (the metrics
    table). Returns the directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    tables["metrics"].to_csv(out / "summary.csv", index=False)
    return out


# --------------------------------------------------------------------------------------
# Figures (matplotlib). Colour follows the electorate, never its position in a chart.
# --------------------------------------------------------------------------------------

_SURFACE, _INK, _INK2, _MUTED, _GRID, _AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)  # fmt: skip
_HUMAN, _HETERO, _NULL = "#2a78d6", "#6250d6", "#898781"
_MODEL_SLOTS = ("#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#e34948")
_LABELS = {
    "human": "Human (ANES)",
    "heterogeneous": "Mixed models",
    "null_ic": "Null: impartial culture",
    "null_permutation": "Null: permuted human",
}


def electorate_styles(names: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Fixed colour, line style and label per electorate. Models take the categorical
    slots in alphabetical order; a seventh model would repeat a colour, so facet instead."""
    models = sorted(n for n in set(names) if n not in _LABELS)
    styles: dict[str, dict[str, Any]] = {
        "human": dict(color=_HUMAN, linestyle="-", label=_LABELS["human"]),
        "heterogeneous": dict(color=_HETERO, linestyle="-", label=_LABELS["heterogeneous"]),
        "null_ic": dict(color=_NULL, linestyle=":", label=_LABELS["null_ic"]),
        "null_permutation": dict(color=_NULL, linestyle="--", label=_LABELS["null_permutation"]),
    }
    for i, m in enumerate(models):
        styles[m] = dict(color=_MODEL_SLOTS[i % len(_MODEL_SLOTS)], linestyle="-", label=m)
    return styles


def _order(names: Sequence[str]) -> list[str]:
    present = set(names)
    models = sorted(n for n in present if n not in _LABELS)
    ordered = ["human", *models, "heterogeneous", "null_permutation", "null_ic"]
    return [n for n in ordered if n in present]


def _axes(ncols: int = 1, width: float = 7.0, height: float = 4.2):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, ncols, figsize=(width * max(1, ncols) * (1 if ncols == 1 else 0.7), height),
        sharey=True, squeeze=False,
    )  # fmt: skip
    fig.patch.set_facecolor(_SURFACE)
    for ax in axes[0]:
        ax.set_facecolor(_SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(_AXIS)
        ax.tick_params(colors=_MUTED, length=0, labelsize=9)
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    return fig, list(axes[0])


def _select(frame: pd.DataFrame, weighting: str, source: str) -> pd.DataFrame:
    """Rows for one weighting, with human and nulls (thermometer-only) kept for either source."""
    keep = (frame["weighting"] == weighting) & (
        (frame["source"] == source) | frame["kind"].isin(["human", "null_ic", "null_permutation"])
    )
    return frame[keep]


def fig_cycle_ncurves(
    ncurves: pd.DataFrame, weighting: str = "weighted", source: str = "thermometer"
):
    """Figure 1: majority-cycle frequency against sub-electorate size, by electorate."""
    data = _select(ncurves, weighting, source)
    elections = list(dict.fromkeys(data["election"]))
    fig, axes = _axes(len(elections))
    styles = electorate_styles(data["electorate"])
    for ax, election in zip(axes, elections, strict=True):
        sub = data[data["election"] == election]
        for name in _order(sub["electorate"]):
            line = sub[sub["electorate"] == name].sort_values("n")
            ax.plot(
                line["n"], line["cycle_freq"], marker="o", markersize=5, linewidth=2,
                markeredgecolor=_SURFACE, markeredgewidth=1.2, **styles[name],
            )  # fmt: skip
        ax.set_xscale("log")
        sizes = sorted(sub["n"].unique())
        ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.minorticks_off()
        ax.set_xlabel("Voters in the sub-electorate (n)", color=_INK2, fontsize=9)
        ax.set_title(election, loc="left", color=_INK, fontsize=10)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Share of draws with a majority cycle", color=_INK2, fontsize=9)
    axes[-1].legend(
        frameon=False, fontsize=8, labelcolor=_INK2, loc="upper left", bbox_to_anchor=(1.01, 1.0)
    )
    fig.suptitle(
        f"Majority cycles by electorate size ({weighting}, {source} ballots)",
        x=0.01, ha="left", color=_INK, fontsize=11,
    )  # fmt: skip
    fig.tight_layout()
    return fig


def _bars(metrics: pd.DataFrame, metric: str, weighting: str, source: str, ylabel: str, title: str):
    data = _select(metrics[metrics["metric"] == metric], weighting, source).dropna(subset=["value"])
    elections = list(dict.fromkeys(data["election"]))
    fig, axes = _axes(len(elections))
    styles = electorate_styles(data["electorate"])
    for ax, election in zip(axes, elections, strict=True):
        sub = data[data["election"] == election].set_index("electorate")
        names = _order(sub.index)
        x = np.arange(len(names))
        values = sub.loc[names, "value"].to_numpy(dtype=float)
        ax.bar(x, values, width=0.62, color=[styles[n]["color"] for n in names], linewidth=0)
        lo = values - sub.loc[names, "ci_low"].to_numpy(dtype=float)
        hi = sub.loc[names, "ci_high"].to_numpy(dtype=float) - values
        has = ~np.isnan(lo)
        if has.any():
            ax.errorbar(
                x[has], values[has], yerr=[lo[has].clip(0), hi[has].clip(0)], fmt="none",
                ecolor=_INK2, elinewidth=1.2, capsize=3,
            )  # fmt: skip
        tops = np.where(has, values + np.nan_to_num(hi), values)  # label clears the whisker
        for xi, v, top in zip(x, values, tops, strict=True):
            ax.annotate(
                f"{v:.2f}", (xi, top), xytext=(0, 4), textcoords="offset points", ha="center",
                color=_INK, fontsize=8,
            )  # fmt: skip
        ax.set_xticks(x, [styles[n]["label"] for n in names], rotation=30, ha="right")
        ax.tick_params(axis="x", labelcolor=_INK2)
        ax.set_title(election, loc="left", color=_INK, fontsize=10)
        ax.set_ylim(0, max(1.0, float(np.nanmax(values)) * 1.15) if len(values) else 1.0)
    axes[0].set_ylabel(ylabel, color=_INK2, fontsize=9)
    fig.suptitle(
        f"{title} ({weighting}, {source} ballots)", x=0.01, ha="left", color=_INK, fontsize=11
    )
    fig.tight_layout()
    return fig


def fig_sp_deletion(
    metrics: pd.DataFrame, weighting: str = "weighted", source: str = "thermometer"
):
    """Figure 2: SP deletion share by electorate, one panel per election."""
    return _bars(
        metrics, "sp_deletion_share", weighting, source,
        "Share of voters to remove for single-peakedness",
        "Distance from single-peakedness by electorate",
    )  # fmt: skip


def fig_pc1(metrics: pd.DataFrame, weighting: str = "weighted"):
    """Figure 3: PC1 explained-variance share of thermometer scores, human vs. LLM."""
    return _bars(
        metrics, "pc1_share", weighting, "thermometer",
        "Variance share of the first principal component",
        "Preference dimensionality: human vs. LLM electorates",
    )  # fmt: skip


FIGURES = {
    "fig1_cycle_ncurves.png": lambda t: fig_cycle_ncurves(t["ncurves"]),
    "fig2_sp_deletion.png": lambda t: fig_sp_deletion(t["metrics"]),
    "fig3_pc1_share.png": lambda t: fig_pc1(t["metrics"]),
}


def write_figures(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> list[Path]:
    """Draw the three SPEC §13 figures into ``<out_dir>/figures/``."""
    import matplotlib

    matplotlib.use("Agg")  # files only; the notebook keeps its own inline backend
    import matplotlib.pyplot as plt

    directory = Path(out_dir) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, draw in FIGURES.items():
        fig = draw(tables)
        fig.savefig(directory / name, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths.append(directory / name)
    return paths


def read_outputs(out_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the tables written by :func:`write_outputs` (what the notebook does)."""
    out = Path(out_dir)
    return {p.stem: pd.read_parquet(p) for p in sorted(out.glob("*.parquet"))}


# --------------------------------------------------------------------------------------
# From a run config
# --------------------------------------------------------------------------------------


def analyze_run(cfg: Any, opts: AnalysisOptions | None = None, figures: bool = True) -> Path:
    """Analyse one finished (or partly finished) run and write
    ``results/metrics/<run_id>/``: the Parquet tables, ``summary.csv`` and the figures.

    The heterogeneous electorate mixes the run's models that are in the ``pool`` section
    of models.yaml (SPEC §14.4 keeps the training-stage checkpoints out); if the run has
    none of those, it mixes all of the run's chat models.
    """
    from scv import runner
    from scv.electorates import build_ballot_table

    election = runner.resolve_election(cfg)
    calls = runner.read_calls(cfg.results_dir, cfg.run_id)
    if calls.empty:
        raise ValueError(
            f"run {cfg.run_id!r} has no call log in {cfg.results_dir}; run `scv elicit`"
        )
    table = build_ballot_table(
        election.id, election.candidates, election.human_scores, calls, election.personas
    )
    registry = runner.load_models(cfg.models_file)
    in_run = [m for m in cfg.models if m in table.models]
    pool = [m for m in in_run if registry.get(m, {}).get("section") == "pool"]
    if len(pool) < 2:
        pool = [m for m in in_run if not registry.get(m, {}).get("base", False)]
    tables = analyze(table, cfg.run_id, pool, opts)
    coherence = voter_coherence(calls[calls["election"] == election.id], election.candidates)
    extra = pd.DataFrame(coherence_metrics(coherence, table, cfg.run_id))
    if len(extra):
        tables["metrics"] = pd.concat(
            [tables["metrics"], extra.reindex(columns=tables["metrics"].columns)],
            ignore_index=True,
        )
    tables["coherence"] = coherence
    out = write_outputs(tables, metrics_dir(cfg.results_dir, cfg.run_id))
    if figures:
        write_figures(tables, out)
    return out
