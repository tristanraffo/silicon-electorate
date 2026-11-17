"""SPEC §12: electorates, metrics, bootstrap, n-curves, matched comparisons, figures.

The main fixture is a mock-generated run on the synthetic ANES-shaped file: the mock
answers with each respondent's own human thermometer scores plus noise, so the human and
"LLM" electorates are matched and their relationship is known.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from helpers import profile_from, random_weak_profile
from scv import analysis, anes, domains, rules
from scv.analysis import AnalysisOptions
from scv.cli import app
from scv.electorates import (
    BallotTable,
    assign_models,
    build_ballot_table,
    heterogeneous,
    homogeneous,
    human_electorate,
    null_ic,
    null_permutation,
)
from scv.nulls import spatial_1d
from scv.rules import Profile
from scv.runner import RunConfig, read_calls, resolve_election, run_from_config

YAML = Path(__file__).parent / "fixtures" / "anes_synthetic.yaml"
MODELS = ["llama31_8b", "qwen25_7b", "mistral_7b"]
OPTS = AnalysisOptions(bootstrap=60, draws=60, replicates=6, sizes=(11, 25, 51, 501), seed=3)


def make_run(root: Path, run_id: str, noise: float, models=MODELS, limit=150) -> RunConfig:
    if not (root / "processed" / "anes_synthetic").exists():
        anes.build(anes.load_election(YAML), root / "processed")
    cfg = RunConfig.model_validate(
        {
            "run_id": run_id,
            "seed": 5,
            "backend": "mock",
            "models": models,
            "election": {
                "source": "anes",
                "id": "anes_synthetic",
                "config": str(YAML),
                "processed_dir": str(root / "processed"),
            },
            "limit_respondents": limit,
            "results_dir": str(root / "results"),
            "backend_options": {"noise": noise},
        }
    )
    run_from_config(cfg)
    return cfg


def table_for(cfg: RunConfig) -> BallotTable:
    election = resolve_election(cfg)
    calls = read_calls(cfg.results_dir, cfg.run_id)
    return build_ballot_table(
        election.id, election.candidates, election.human_scores, calls, election.personas
    )


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("analysis")


@pytest.fixture(scope="module")
def noisy(root) -> RunConfig:
    return make_run(root, "noisy", noise=10.0)


@pytest.fixture(scope="module")
def exact(root) -> RunConfig:
    return make_run(root, "exact", noise=0.0, models=["tiny"], limit=120)


@pytest.fixture(scope="module")
def table(noisy) -> BallotTable:
    return table_for(noisy)


@pytest.fixture(scope="module")
def tables(table) -> dict[str, pd.DataFrame]:
    return analysis.analyze(table, "noisy", MODELS, OPTS)


def metric(tables, electorate, name, weighting="weighted", source="thermometer") -> pd.Series:
    m = tables["metrics"]
    rows = m[
        (m["electorate"] == electorate)
        & (m["metric"] == name)
        & (m["weighting"] == weighting)
        & (m["source"] == source)
    ]
    assert len(rows) == 1, (electorate, name, weighting, source, len(rows))
    return rows.iloc[0]


# --------------------------------------------------------------------------------------
# Ballot table and electorates
# --------------------------------------------------------------------------------------


def test_ballot_table_lines_humans_up_with_every_model(noisy, table):
    election = resolve_election(noisy)
    human = election.human_scores.set_index("respondent_id")
    assert table.n == 150 and table.candidates == election.candidates
    assert set(table.models) == set(MODELS)
    assert np.array_equal(
        table.human,
        human.loc[table.respondent_ids, list(table.candidates)].to_numpy(),
        equal_nan=True,
    )
    assert np.array_equal(table.weights, human.loc[table.respondent_ids, "weight"].to_numpy())
    personas = election.personas.set_index("respondent_id")
    expected = personas.loc[table.respondent_ids, "party_id_7"].fillna("").to_numpy()
    assert list(table.strata) == list(expected)
    for m in MODELS:
        assert table.llm_scores[m].shape == table.llm_ranks[m].shape == (150, 5)
        assert not np.isnan(table.llm_scores[m]).any()
    # different mock "models" are different noisy copies
    assert not np.array_equal(table.llm_scores[MODELS[0]], table.llm_scores[MODELS[1]])


def test_noise_free_mock_reproduces_the_human_electorate(exact):
    t = table_for(exact)
    rated = ~np.isnan(t.human)
    assert np.array_equal(t.llm_scores["tiny"][rated], t.human[rated])
    assert (t.llm_scores["tiny"][~rated] == 50).all()  # the mock's stand-in for "not rated"
    complete = rated.all(axis=1)
    h, llm, rk = human_electorate(t), homogeneous(t, "tiny"), homogeneous(t, "tiny", "ranking")
    assert complete.sum() > 60
    assert np.array_equal(h.ranks[complete], llm.ranks[complete])
    assert np.array_equal(h.ranks[complete], rk.ranks[complete])  # ranking mode agrees too
    assert h.scores is not None and rk.scores is None
    assert llm.kind == "homogeneous" and llm.model == "tiny" and rk.source == "ranking"


def test_assign_models_is_stratified_seeded_and_varies():
    strata = np.array(["D"] * 31 + ["R"] * 20 + [""] * 9, dtype=object)
    a = assign_models(strata, 4, rng=1)
    assert set(a) == {0, 1, 2, 3}
    for s in ("D", "R", ""):
        counts = np.bincount(a[strata == s], minlength=4)
        assert counts.max() - counts.min() <= 1, s  # every model gets the same persona mix
    assert np.array_equal(a, assign_models(strata, 4, rng=1))
    assert not np.array_equal(a, assign_models(strata, 4, rng=2))
    # across many draws each persona visits every model about equally often
    visits = np.stack([assign_models(strata, 4, rng=s) for s in range(400)])
    share = (visits == 0).mean(axis=0)
    assert np.all(np.abs(share - 0.25) < 0.1)


def test_heterogeneous_takes_each_ballot_from_its_assigned_model():
    n, cands = 40, ("A", "B", "C")
    base = np.tile([3.0, 2.0, 1.0], (n, 1))
    t = BallotTable(
        election="e",
        candidates=cands,
        respondent_ids=np.array([f"r{i}" for i in range(n)]),
        weights=np.linspace(0.5, 2.0, n),
        strata=np.array(["x", "y"] * (n // 2), dtype=object),
        human=base,
        llm_scores={"m0": base, "m1": base[:, ::-1].copy()},  # m1 reverses the order
        llm_ranks={"m0": np.tile([0.0, 1.0, 2.0], (n, 1)), "m1": np.tile([2.0, 1.0, 0.0], (n, 1))},
    )
    assignment = assign_models(t.strata, 2, rng=9)
    for source in ("thermometer", "ranking"):
        e = heterogeneous(t, ["m0", "m1"], source, rng=9)
        assert np.array_equal(e.ranks[:, 0] == 0, assignment == 0)  # A first iff model 0
        assert np.array_equal(e.weights, t.weights) and e.kind == "heterogeneous"
    assert 15 <= (assignment == 0).sum() <= 25


def test_null_electorates(table):
    ic = null_ic(table, rng=0)
    assert ic.ranks.shape == table.human.shape and ic.scores is None
    assert np.array_equal(np.sort(ic.ranks, axis=1), np.tile(np.arange(5.0), (table.n, 1)))
    assert np.array_equal(ic.weights, table.weights)
    perm = null_permutation(table, rng=0)
    for j in range(5):
        assert np.array_equal(
            np.sort(perm.scores[:, j]), np.sort(table.human[:, j]), equal_nan=True
        )
    assert not np.array_equal(perm.scores, table.human, equal_nan=True)


# --------------------------------------------------------------------------------------
# Fast batch statistics agree with the engine
# --------------------------------------------------------------------------------------


def test_batch_statistics_match_rules_and_domains():
    rng = np.random.default_rng(0)
    seen = {True: 0, False: 0}
    for _ in range(60):
        ballots, cands = random_weak_profile(rng, int(rng.integers(3, 25)), int(rng.integers(3, 6)))
        base = Profile(ballots, cands)
        prefs = analysis._preference_matrix(base.ranks)
        consistency = domains.sp_consistency(base)
        counts = rng.integers(0, 4, size=(8, base.n)).astype(float)
        counts[counts.sum(axis=1) == 0, 0] = 1
        cycle, winner = analysis.batch_majority(prefs, counts, base.m)
        deletion = analysis.batch_sp_deletion(consistency, counts)
        for k, w in enumerate(counts):
            p = Profile(ballots, cands, w)
            assert cycle[k] == rules.has_majority_cycle(p)
            assert winner[k] == (rules.condorcet_winner(p) is not None)
            assert deletion[k] == pytest.approx(domains.sp_deletion_share(p))
            seen[bool(cycle[k])] += 1
    assert seen[True] > 5 and seen[False] > 100


def test_point_metrics_on_the_condorcet_paradox():
    p = profile_from([(1, "A>B>C"), (1, "B>C>A"), (1, "C>A>B")], "ABC")
    e = analysis.Electorate("paradox", "human", "thermometer", np.array(p.ranks), np.ones(3))
    values, winners = analysis.point_metrics(e, p.candidates)
    assert values["has_cycle"] == 1.0 and values["condorcet_winner_exists"] == 0.0
    assert values["majority_transitive"] == 0.0 and values["smith_size"] == 3.0
    assert values["n_three_cycles"] == 1.0 and values["vr_share"] == 0.0
    assert values["sp_deletion_share"] == pytest.approx(1 / 3)
    assert values["sp_plateau_deletion_share"] == pytest.approx(1 / 3)
    assert values["share_strict_ballots"] == 1.0 and values["share_complete_ballots"] == 1.0
    assert math.isnan(values["borda_differs_from_condorcet"]) and math.isnan(values["pc1_share"])
    assert set(winners) == set(rules.RULES) and winners["copeland"] == ("A", "B", "C")
    quick, _ = analysis.point_metrics(e, p.candidates, full=False)
    assert "irv_monotonicity_failure_fraction" not in quick


# --------------------------------------------------------------------------------------
# The full analysis of a mock-generated run
# --------------------------------------------------------------------------------------


def test_summary_is_tidy_and_complete(tables):
    m = tables["metrics"]
    assert list(m.columns) == [
        "run_id", "election", "electorate", "kind", "model", "source", "weighting", "metric",
        "value", "ci_low", "ci_high", "sd", "interval", "n_voters", "n_candidates",
    ]  # fmt: skip
    assert set(m["election"]) == {"anes_synthetic"} and set(m["weighting"]) == {
        "weighted",
        "unweighted",
    }
    assert set(m["electorate"]) == {
        "human",
        *MODELS,
        "heterogeneous",
        "null_ic",
        "null_permutation",
    }
    assert set(m["source"]) == {"thermometer", "ranking"}
    assert not m.duplicated(["electorate", "source", "weighting", "metric"]).any()
    kinds = m.drop_duplicates(["electorate", "source"]).groupby("kind").size().to_dict()
    assert kinds == {
        "human": 1,
        "homogeneous": 6,
        "heterogeneous": 2,
        "null_ic": 1,
        "null_permutation": 1,
    }
    for name in (
        "has_cycle", "condorcet_winner_exists", "smith_size", "n_three_cycles",
        "rule_disagreement_share", "irv_monotonicity_failure_fraction", "sp_deletion_share",
        "sp_plateau_deletion_share", "vr_share", "vr_mean_min_violation", "pc1_share", "pc2_share",
        "distinct_share", "ballot_entropy", "mean_kendall_tau", "has_cycle_boot_freq",
        "condorcet_winner_boot_freq", "matched_tau_b_thermometer", "matched_abs_error",
    ):  # fmt: skip
        assert name in set(m.loc[m["electorate"] == MODELS[0], "metric"]), name
    assert (m["n_candidates"] == 5).all()


def test_headline_numbers_equal_direct_computation(table, tables):
    cands = table.candidates
    for weighting, w in (("weighted", table.weights), ("unweighted", np.ones(table.n))):
        human = Profile.from_scores(table.human, cands, w)
        assert metric(tables, "human", "sp_deletion_share", weighting)["value"] == pytest.approx(
            domains.sp_deletion_share(human)
        )
        assert metric(tables, "human", "has_cycle", weighting)["value"] == float(
            rules.has_majority_cycle(human)
        )
        assert metric(tables, "human", "pc1_share", weighting)["value"] == pytest.approx(
            domains.dimensionality(table.human, w).pc1_share
        )
        assert metric(tables, "human", "mean_kendall_tau", weighting)["value"] == pytest.approx(
            domains.diversity(human).mean_kendall_tau
        )
        model = Profile.from_scores(table.llm_scores["qwen25_7b"], cands, w)
        assert metric(tables, "qwen25_7b", "vr_mean_min_violation", weighting)[
            "value"
        ] == pytest.approx(domains.value_restriction(model).mean_min_violation)
        ranking = Profile.from_ranks(table.llm_ranks["qwen25_7b"], cands, w)
        assert metric(tables, "qwen25_7b", "sp_deletion_share", weighting, "ranking")[
            "value"
        ] == pytest.approx(domains.sp_deletion_share(ranking))
    # weights matter: the two versions are not the same number
    a = metric(tables, "human", "sp_deletion_share", "weighted")["value"]
    b = metric(tables, "human", "sp_deletion_share", "unweighted")["value"]
    assert a != b
    assert math.isnan(metric(tables, "qwen25_7b", "pc1_share", source="ranking")["value"])


def test_bootstrap_intervals_and_replicate_spreads(tables):
    for name in ("human", *MODELS):
        row = metric(tables, name, "sp_deletion_share")
        assert row["interval"] == "bootstrap" and row["sd"] > 0
        assert row["ci_low"] < row["ci_high"]
        assert row["ci_low"] - 0.05 <= row["value"] <= row["ci_high"] + 0.05
        pc1 = metric(tables, name, "pc1_share")
        assert pc1["ci_low"] <= pc1["value"] <= pc1["ci_high"]
        freq = metric(tables, name, "has_cycle_boot_freq")["value"]
        assert 0.0 <= freq <= 1.0
    for name in ("heterogeneous", "null_ic", "null_permutation"):
        row = metric(tables, name, "sp_deletion_share")
        assert row["interval"] == "replicates" and row["ci_low"] <= row["value"] <= row["ci_high"]
    assert metric(tables, "null_ic", "sp_deletion_share")["sd"] > 0
    # mean over R assignments of a 0/1 outcome is a frequency
    assert 0.0 <= metric(tables, "null_ic", "has_cycle")["value"] <= 1.0


def test_structure_of_the_known_mock_world(tables):
    sp = {
        n: metric(tables, n, "sp_deletion_share")["value"]
        for n in ("human", *MODELS, "heterogeneous", "null_permutation", "null_ic")
    }
    # the noisy copies are further from single-peaked than the humans they copy,
    # and both nulls are further still
    assert all(sp["human"] < sp[m] < sp["null_permutation"] < sp["null_ic"] for m in MODELS)
    assert (
        min(sp[m] for m in MODELS) - 0.08 < sp["heterogeneous"] < max(sp[m] for m in MODELS) + 0.08
    )
    pc1 = {
        n: metric(tables, n, "pc1_share")["value"] for n in ("human", MODELS[0], "null_permutation")
    }
    assert pc1["human"] > pc1[MODELS[0]] > pc1["null_permutation"]  # noise, then no correlation
    assert metric(tables, "null_ic", "mean_kendall_tau")["value"] == pytest.approx(0.5, abs=0.03)
    assert (
        metric(tables, "null_ic", "has_cycle")["value"]
        > metric(tables, "human", "has_cycle")["value"] - 1e-9
    )


def test_n_curves(table, tables):
    c = tables["ncurves"]
    assert set(c["n"]) == {11, 25, 51}  # 501 > 150 voters: skipped
    assert (c["draws"] == 60).all()
    for col in ("cycle_freq", "condorcet_winner_freq", "sp_deletion_mean"):
        assert c[col].between(0, 1).all()
    assert not c.duplicated(["electorate", "source", "weighting", "n"]).any()
    assert set(c["electorate"]) == {
        "human",
        *MODELS,
        "heterogeneous",
        "null_ic",
        "null_permutation",
    }
    w = c[(c["weighting"] == "weighted") & (c["source"] == "thermometer")].set_index(
        ["electorate", "n"]
    )
    assert w.loc[("null_ic", 51), "cycle_freq"] > w.loc[("human", 51), "cycle_freq"] + 0.1
    assert w.loc[("null_ic", 51), "sp_deletion_mean"] > w.loc[("human", 51), "sp_deletion_mean"]


def test_n_curve_of_a_single_peaked_electorate_never_cycles():
    world = spatial_1d(300, 5, rng=0)
    e = analysis.Electorate(
        "sp", "human", "thermometer", np.array(world.profile.ranks), np.ones(300)
    )
    consistency = domains.sp_consistency(world.profile)
    opts = AnalysisOptions(draws=80, sizes=(11, 51, 101))
    rows = analysis.n_curve([e], [consistency], False, opts, np.random.default_rng(0))
    assert [r["n"] for r in rows] == [11, 51, 101]
    for r in rows:  # Black's theorem, at every odd size
        assert r["cycle_freq"] == 0.0 and r["condorcet_winner_freq"] == 1.0
        assert r["sp_deletion_mean"] == 0.0 and r["draws"] == 80


def test_analysis_is_seeded(table, tables):
    again = analysis.analyze(table, "noisy", MODELS, OPTS)
    pd.testing.assert_frame_equal(again["metrics"], tables["metrics"])
    pd.testing.assert_frame_equal(again["ncurves"], tables["ncurves"])
    other = analysis.analyze(
        table, "noisy", MODELS, AnalysisOptions(**{**OPTS.__dict__, "seed": 4})
    )
    assert not other["ncurves"]["cycle_freq"].equals(tables["ncurves"]["cycle_freq"])


def test_winners_table(table, tables):
    w = tables["winners"]
    assert set(w["rule"]) == set(rules.RULES)
    row = w[(w["electorate"] == "human") & (w["weighting"] == "weighted") & (w["rule"] == "borda")]
    direct = rules.borda(Profile.from_scores(table.human, table.candidates, table.weights))
    assert row["winners"].iloc[0] == "|".join(direct.winners)


# --------------------------------------------------------------------------------------
# Matched comparisons
# --------------------------------------------------------------------------------------


def test_matched_comparisons_are_perfect_without_noise(exact):
    t = table_for(exact)
    m = analysis.matched_comparisons(t)
    assert len(m) == t.n and set(m["model"]) == {"tiny"}
    complete = ~np.isnan(t.human).any(axis=1)
    full = m[complete]
    assert full["abs_error"].eq(0).all()
    for column in ("tau_b_thermometer", "tau_b_ranking"):
        assert len(full[column].dropna()) > 50
        assert np.allclose(full[column].dropna(), 1.0)
    assert full["top_agree_thermometer"].eq(1.0).all() and full["top_agree_ranking"].eq(1.0).all()
    assert m["abs_error"].eq(0).all()  # unrated candidates are skipped, not scored against 50
    assert list(m["respondent_id"]) == list(t.respondent_ids)


def test_matched_comparisons_degrade_with_noise_and_break_down_by_group(table, tables):
    m, g = tables["matched"], tables["matched_groups"]
    assert len(m) == 3 * table.n
    assert 3 < m["abs_error"].mean() < 15  # noise sd 10, median of 5 samples
    assert 0.4 < m["tau_b_thermometer"].mean() < 0.98
    assert m["top_agree_thermometer"].mean() < 1.0
    assert set(g["group_var"]) == {"all", "party_id_7", "ideology_7"}
    assert "(missing)" in set(g.loc[g["group_var"] == "ideology_7", "group"])

    one = m[(m["model"] == "qwen25_7b") & (m["party_id_7"] == "a strong Democrat")].dropna(
        subset=["abs_error"]
    )
    row = g[
        (g["model"] == "qwen25_7b")
        & (g["group"] == "a strong Democrat")
        & (g["metric"] == "abs_error")
    ].iloc[0]
    assert row["n"] == len(one) > 3
    assert row["weighted_mean"] == pytest.approx(
        np.average(one["abs_error"], weights=one["weight"])
    )
    assert row["unweighted_mean"] == pytest.approx(one["abs_error"].mean())
    overall = g[
        (g["model"] == "qwen25_7b") & (g["group_var"] == "all") & (g["metric"] == "abs_error")
    ]
    assert metric(tables, "qwen25_7b", "matched_abs_error")["value"] == pytest.approx(
        overall["weighted_mean"].iloc[0]
    )


def test_matched_helpers():
    nan = math.nan
    assert analysis._tau_b(np.array([1.0, 2, 3]), np.array([10.0, 20, 30])) == pytest.approx(1.0)
    assert analysis._tau_b(np.array([1.0, 2, 3]), np.array([30.0, 20, 10])) == pytest.approx(-1.0)
    assert math.isnan(analysis._tau_b(np.array([1.0, 1, 1]), np.array([1.0, 2, 3])))  # constant
    assert math.isnan(analysis._tau_b(np.array([1.0, nan, nan]), np.array([1.0, 2, 3])))
    assert analysis._tau_b(np.array([1.0, 2, nan, 4]), np.array([5.0, 6, 9, 7])) == pytest.approx(
        1.0
    )
    assert (
        analysis._top_agree(np.array([9.0, 1, 1]), np.array([5.0, 5, 0])) == 1.0
    )  # tie includes it
    assert analysis._top_agree(np.array([9.0, 1, 1]), np.array([0.0, 5, 5])) == 0.0
    assert math.isnan(analysis._top_agree(np.array([nan, nan]), np.array([1.0, 2])))


# --------------------------------------------------------------------------------------
# M7 acceptance: `scv analyze` on a mock-generated run
# --------------------------------------------------------------------------------------


def write_config(cfg: RunConfig, path: Path) -> Path:
    import yaml

    path.write_text(yaml.safe_dump(cfg.model_dump()))
    return path


def test_scv_analyze_writes_summary_and_three_figures(noisy, root):
    """SPEC §13 M7."""
    path = write_config(noisy, root / "noisy.yaml")
    out = CliRunner().invoke(
        app,
        [
            "analyze",
            "--config",
            str(path),
            "--bootstrap",
            "40",
            "--draws",
            "40",
            "--replicates",
            "4",
        ],
    )
    assert out.exit_code == 0, out.output
    d = Path(noisy.results_dir) / "metrics" / "noisy"
    assert "summary.csv" in out.output

    summary = pd.read_csv(d / "summary.csv")
    assert {"electorate", "source", "weighting", "metric", "value"} <= set(summary.columns)
    assert len(summary) > 300 and summary["value"].notna().mean() > 0.9
    pd.testing.assert_frame_equal(
        summary[["electorate", "metric", "value"]],
        pd.read_parquet(d / "metrics.parquet")[["electorate", "metric", "value"]],
    )
    for name in ("ncurves", "winners", "matched", "matched_groups", "coherence"):
        assert len(pd.read_parquet(d / f"{name}.parquet")) > 0
    h4 = summary[summary["source"] == "pairwise"]
    assert set(h4["metric"]) == {f"h4_{m}" for m in analysis.H4_METRICS}
    assert set(h4["electorate"]) == set(MODELS) and h4["value"].between(0, 1).all()
    figures = sorted(p.name for p in (d / "figures").glob("*.png"))
    assert figures == ["fig1_cycle_ncurves.png", "fig2_sp_deletion.png", "fig3_pc1_share.png"]
    for name in figures:
        data = (d / "figures" / name).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 15_000
    assert set(analysis.read_outputs(d)) == {
        "metrics", "ncurves", "winners", "matched", "matched_groups", "coherence",
    }  # fmt: skip


def test_scv_analyze_without_a_call_log_fails_clearly(noisy, root):
    empty = noisy.model_copy(update={"run_id": "never_ran"})
    out = CliRunner().invoke(
        app, ["analyze", "--config", str(write_config(empty, root / "e.yaml"))]
    )
    assert out.exit_code == 1 and "no call log" in out.output


def test_single_model_and_synthetic_runs_are_analysed_too(exact, tmp_path):
    out = analysis.analyze_run(
        exact, AnalysisOptions(bootstrap=10, draws=10, replicates=2, sizes=(11,)), figures=False
    )
    m = pd.read_csv(out / "summary.csv")
    assert "heterogeneous" not in set(m["electorate"])  # one model: nothing to mix
    assert {"human", "tiny", "null_ic", "null_permutation"} == set(m["electorate"])
    assert not (out / "figures").exists()
    # without noise the mock electorate is as single-peaked as the humans on complete ballots
    assert (
        m[(m["electorate"] == "tiny") & (m["metric"] == "matched_abs_error")]["value"].eq(0).all()
    )

    cfg = RunConfig.model_validate(
        {
            "run_id": "syn",
            "backend": "mock",
            "models": ["tiny", "qwen25_7b"],
            "election": {
                "id": "demo",
                "candidates": ["Ann Lee", "Bo Chen", "Cy Diaz"],
                "question_text": "Rate this person.",
                "n_respondents": 30,
            },
            "results_dir": str(tmp_path),
            "backend_options": {"noise": 5.0},
        }
    )
    run_from_config(cfg)
    out = analysis.analyze_run(
        cfg, AnalysisOptions(bootstrap=10, draws=10, replicates=2, sizes=(11,))
    )
    m = pd.read_csv(out / "summary.csv")
    assert (
        "heterogeneous" in set(m["electorate"]) and len(list((out / "figures").glob("*.png"))) == 3
    )


# --------------------------------------------------------------------------------------
# Figures and notebook
# --------------------------------------------------------------------------------------


def test_figures_show_every_electorate_once_with_stable_colours(tables):
    import matplotlib.pyplot as plt

    fig = analysis.fig_cycle_ncurves(tables["ncurves"])
    (ax,) = fig.axes
    labels = [line.get_label() for line in ax.get_lines()]
    assert labels == [
        "Human (ANES)",
        *sorted(MODELS),
        "Mixed models",
        "Null: permuted human",
        "Null: impartial culture",
    ]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == labels
    assert ax.get_xscale() == "log" and ax.get_ylim()[0] == 0

    fig2 = analysis.fig_sp_deletion(tables["metrics"])
    bars = fig2.axes[0].patches
    assert len(bars) == 7 and fig2.axes[0].get_ylim() == (0, 1.0)
    heights = [b.get_height() for b in bars]
    assert heights[0] == pytest.approx(metric(tables, "human", "sp_deletion_share")["value"])

    fig3 = analysis.fig_pc1(tables["metrics"])
    assert len(fig3.axes[0].patches) == 6  # impartial culture has no scores, so no PC1

    styles = analysis.electorate_styles(["human", *MODELS, "heterogeneous"])
    fewer = analysis.electorate_styles(["human", "heterogeneous"])
    assert styles["human"] == fewer["human"] and styles["heterogeneous"] == fewer["heterogeneous"]
    colours = [styles[n]["color"] for n in ("human", *MODELS, "heterogeneous")]
    assert len(set(colours)) == len(colours)

    # two elections become two panels with a shared scale
    second = tables["metrics"].assign(election="another_election")
    both = pd.concat([tables["metrics"], second], ignore_index=True)
    fig4 = analysis.fig_sp_deletion(both)
    assert [a.get_title(loc="left") for a in fig4.axes] == ["anes_synthetic", "another_election"]
    plt.close("all")


def test_notebook_is_thin_and_runs_against_the_outputs(noisy, tables, root, monkeypatch):
    out = analysis.write_outputs(tables, root / "nb_metrics")
    nb = json.loads(Path("notebooks/01_results.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    source = "\n".join(code)
    for forbidden in ("rules.", "domains.", "analyze(", "bootstrap(", "elicit"):
        assert forbidden not in source  # it reads files and draws; it computes nothing
    assert "read_outputs" in source and source.count("analysis.fig_") == 3
    monkeypatch.setenv("SCV_RUN_DIRS", os.pathsep.join([str(out)]))
    scope: dict = {}
    for cell in code:
        exec(compile(cell, "<notebook>", "exec"), scope)  # noqa: S102
    assert all(scope[f].axes for f in ("fig1", "fig2", "fig3"))
    assert len(scope["metrics"]) == len(tables["metrics"])


# --------------------------------------------------------------------------------------
# Weights reach the resampling, and the ranking ballot is the greedy one
# --------------------------------------------------------------------------------------


def lopsided() -> analysis.Electorate:
    """Three cyclic blocs; with weights, one bloc outweighs the other two together."""
    p = profile_from([(1, "A>B>C")] * 20 + [(1, "B>C>A")] * 20 + [(1, "C>A>B")] * 20, "ABC")
    weights = np.array([50.0] * 20 + [1.0] * 40)
    return analysis.Electorate("lopsided", "human", "thermometer", np.array(p.ranks), weights)


def test_bootstrap_uses_survey_weights_when_weighted():
    e = lopsided()
    consistency = domains.sp_consistency(e.profile(list("ABC"), weighted=False))
    weighted = analysis.bootstrap(e, True, 300, np.random.default_rng(0), consistency)
    unweighted = analysis.bootstrap(e, False, 300, np.random.default_rng(0), consistency)
    # weighted: the heavy bloc is a majority on its own, so A always wins and nothing cycles
    assert weighted["has_cycle"].mean() == 0.0 and weighted["condorcet_winner_exists"].mean() == 1.0
    assert weighted["sp_deletion_share"].max() < 0.1
    # unweighted: three equal blocs, so cycles are common
    assert unweighted["has_cycle"].mean() > 0.2
    # the smallest of three resampled blocs has to go: a little under one third on average
    assert 0.2 < unweighted["sp_deletion_share"].mean() < 1 / 3
    assert set(weighted) == {"has_cycle", "condorcet_winner_exists", "sp_deletion_share"}


def test_sub_electorates_are_drawn_in_proportion_to_weight_when_weighted():
    weights = np.array([50.0] * 20 + [1.0] * 40)
    rng = np.random.default_rng(1)
    w = analysis._sample_counts(weights, 11, 400, True, rng)
    u = analysis._sample_counts(weights, 11, 400, False, rng)
    for counts in (w, u):
        assert counts.shape == (400, 60) and set(np.unique(counts)) == {0.0, 1.0}
        assert (counts.sum(axis=1) == 11).all()  # distinct voters, each counted once
    assert w[:, :20].sum(axis=1).mean() > 9.5  # heavy voters fill almost every seat
    assert u[:, :20].sum(axis=1).mean() == pytest.approx(11 / 3, abs=0.3)

    e = lopsided()
    consistency = domains.sp_consistency(e.profile(list("ABC"), weighted=False))
    opts = AnalysisOptions(draws=200, sizes=(11,))
    heavy = analysis.n_curve([e], [consistency], True, opts, np.random.default_rng(2))[0]
    even = analysis.n_curve([e], [consistency], False, opts, np.random.default_rng(2))[0]
    assert heavy["cycle_freq"] < 0.02 < even["cycle_freq"]


def test_the_ranking_ballot_is_the_greedy_ranking(noisy, table):
    from scv.elicit import GREEDY, ranking_table

    ranks = ranking_table(read_calls(noisy.results_dir, noisy.run_id))
    ranks = ranks[ranks["model_key"] == "qwen25_7b"].set_index(["respondent_id", "sample_idx"])
    col = {c: j for j, c in enumerate(table.candidates)}
    differs_from_a_sample = 0
    for i, rid in enumerate(table.respondent_ids):
        greedy = ranks.loc[(rid, GREEDY), "ballot"]
        expected = np.full(len(col), np.nan)
        for cand, r in greedy.rank_of().items():
            expected[col[cand]] = r
        assert np.array_equal(table.llm_ranks["qwen25_7b"][i], expected, equal_nan=True)
        differs_from_a_sample += ranks.loc[(rid, 0), "ballot"] != greedy
    assert differs_from_a_sample > 20  # so this test can tell the two apart


def test_h4_voter_coherence(exact, noisy, table):
    # no noise: every mock voter is perfectly transitive and agrees with its own ranking
    calls = read_calls(exact.results_dir, exact.run_id)
    t = table_for(exact)
    clean = analysis.voter_coherence(calls, t.candidates)
    assert len(clean) == t.n and set(clean["model"]) == {"tiny"}
    assert clean["strict_cycle_share"].eq(0).all() and clean["any_strict_cycle"].eq(0).all()
    assert clean["abs_position_bias"].max() < 1e-12
    # ties in the mock's utilities give p = .5 exactly, which counts against agreement
    assert clean["ranking_agreement"].dropna().mean() > 0.97
    rows = analysis.coherence_metrics(clean, t, "exact")
    by = {(r["metric"], r["weighting"]): r["value"] for r in rows}
    assert by[("h4_any_strict_cycle", "weighted")] == 0.0
    assert {r["source"] for r in rows} == {"pairwise"}

    # with noise in the pairwise logits some voters become incoherent
    noisy_calls = read_calls(noisy.results_dir, noisy.run_id)
    messy = analysis.voter_coherence(noisy_calls, table.candidates)
    assert len(messy) == 3 * table.n
    assert messy["any_wst_violation"].mean() > 0.02
    assert messy["ranking_agreement"].mean() < clean["ranking_agreement"].mean()
    one = messy[messy["model"] == "qwen25_7b"]
    w = pd.Series(table.weights, index=table.respondent_ids)[one["respondent_id"]].to_numpy()
    rows = analysis.coherence_metrics(messy, table, "noisy")
    got = next(
        r["value"]
        for r in rows
        if r["model"] == "qwen25_7b"
        and r["metric"] == "h4_wst_violation_share"
        and r["weighting"] == "weighted"
    )
    assert got == pytest.approx(np.average(one["wst_violation_share"], weights=w))

    thermometer_only = noisy_calls[noisy_calls["mode"] == "thermometer"]
    assert analysis.voter_coherence(thermometer_only, table.candidates).empty


def test_sensitivity_run_leaves_a_models_ranking_ballots_out(noisy, root):
    """`scv analyze --exclude-ranking M`: M keeps its thermometer electorate, loses its
    ranking electorate and its place in the mixed-model ranking electorate, and the
    output goes to its own folder so the full analysis is untouched."""
    small = AnalysisOptions(bootstrap=20, draws=20, replicates=4, sizes=(11, 25), seed=3)
    full = analysis.read_outputs(analysis.analyze_run(noisy, small, figures=False))
    out = analysis.analyze_run(noisy, small, figures=False, exclude_ranking=["mistral_7b"])
    assert out.name == f"{noisy.run_id}__ranking_without_mistral_7b"
    assert out != analysis.metrics_dir(noisy.results_dir, noisy.run_id)
    sens = analysis.read_outputs(out)
    m = sens["metrics"]
    assert set(m["run_id"]) == {out.name}

    def models(frame, source):
        f = frame[(frame["kind"] == "homogeneous") & (frame["source"] == source)]
        return set(f["model"])

    assert models(m, "thermometer") == set(MODELS)
    assert models(m, "ranking") == set(MODELS) - {"mistral_7b"}
    assert models(full["metrics"], "ranking") == set(MODELS)
    # the mixed-model ranking electorate still exists (two models are left)
    key = ["electorate", "source", "weighting", "metric"]
    mixed = lambda t: t[(t["kind"] == "heterogeneous") & (t["source"] == "ranking")]  # noqa: E731
    assert len(mixed(m))
    # everything that does not involve rankings is identical
    keep = lambda t: t[t["source"] == "thermometer"].set_index(key)["value"]  # noqa: E731
    pd.testing.assert_series_equal(keep(full["metrics"]), keep(m))
    ranked = sens["matched"][sens["matched"]["tau_b_ranking"].notna()]["model"]
    assert "mistral_7b" not in set(ranked)

    with pytest.raises(ValueError, match="cannot exclude"):
        analysis.analyze_run(noisy, small, figures=False, exclude_ranking=["nope"])
    # with one ranking model left there is no mixed ranking electorate, and no crash
    one = analysis.read_outputs(
        analysis.analyze_run(
            noisy, small, figures=False, exclude_ranking=["mistral_7b", "qwen25_7b"]
        )
    )["metrics"]
    assert mixed(one).empty and models(one, "ranking") == {"llama31_8b"}


def test_mixed_ranking_electorate_uses_only_models_that_have_ranking_ballots(table):
    # The mock's greedy rankings are the same for every model, so give one model
    # reversed rankings: the mixed ranking electorate must change when it is left out,
    # and must equal a single model's electorate when only identical models remain.
    import dataclasses

    spoiled = {**table.llm_ranks, "mistral_7b": -table.llm_ranks["mistral_7b"]}
    with_bad = dataclasses.replace(table, llm_ranks=spoiled)
    without = dataclasses.replace(
        table, llm_ranks={m: r for m, r in spoiled.items() if m != "mistral_7b"}
    )
    small = AnalysisOptions(bootstrap=10, draws=10, replicates=4, sizes=(11,), seed=3)

    def value(tab, electorate, source="ranking"):
        m = analysis.analyze(tab, "x", MODELS, small)["metrics"]
        f = m[(m["electorate"] == electorate) & (m["source"] == source)]
        return f[(f["metric"] == "sp_deletion_share") & (f["weighting"] == "weighted")]["value"]

    bad, clean = float(value(with_bad, "heterogeneous").iloc[0]), value(without, "heterogeneous")
    assert len(clean) == 1 and float(clean.iloc[0]) < bad - 0.02  # less to delete
    assert float(clean.iloc[0]) == pytest.approx(float(value(without, "llama31_8b").iloc[0]))
    # thermometer mixing still uses all three models
    a = value(with_bad, "heterogeneous", "thermometer")
    assert a.to_list() == value(without, "heterogeneous", "thermometer").to_list()


def test_combined_ranking_ballot_averages_all_of_a_voters_rankings(noisy, root):
    """`--ranking-ballot combined`: mean rank over the greedy and sampled rankings that
    parsed; equal means are ties; a voter with no parsed ranking has an empty ballot."""
    election = resolve_election(noisy)
    calls = read_calls(noisy.results_dir, noisy.run_id)
    # break one voter's greedy ranking and all of another voter's rankings
    model = MODELS[0]
    is_rank = (calls["mode"] == "ranking") & (calls["model_key"] == model)
    voters = calls.loc[is_rank, "respondent_id"].unique()
    one_bad = is_rank & (calls["respondent_id"] == voters[0]) & (calls["sample_idx"] == -1)
    all_bad = is_rank & (calls["respondent_id"] == voters[1])
    calls.loc[one_bad | all_bad, ["parsed", "parse_ok"]] = ["null", False]

    args = (election.id, election.candidates, election.human_scores, calls, election.personas)
    greedy = build_ballot_table(*args)
    combined = build_ballot_table(*args, ranking_ballot="combined")
    with pytest.raises(ValueError, match="ranking_ballot"):
        build_ballot_table(*args, ranking_ballot="median")

    cands = list(election.candidates)
    from scv.elicit import ranking_table

    rt = ranking_table(calls)
    rt = rt[(rt["model_key"] == model) & rt["parse_ok"]]
    ids = list(combined.respondent_ids)
    for rid in (voters[0], voters[2], voters[3]):
        mine = rt[rt["respondent_id"] == rid]["ballot"]
        want = np.mean([[b.rank_of()[c] for c in cands] for b in mine], axis=0)
        assert len(mine) == (3 if rid == voters[0] else 4)
        np.testing.assert_allclose(combined.llm_ranks[model][ids.index(rid)], want, atol=1e-8)
    assert np.isnan(combined.llm_ranks[model][ids.index(voters[1])]).all()
    assert np.isnan(greedy.llm_ranks[model][ids.index(voters[0])]).all()  # greedy one is gone
    # sampled rankings differ from the greedy one, so the two ballots are not the same
    a, b = greedy.llm_ranks[MODELS[1]], combined.llm_ranks[MODELS[1]]
    assert not np.allclose(a, b, equal_nan=True)
    # thermometers are untouched
    np.testing.assert_array_equal(greedy.llm_scores[model], combined.llm_scores[model])

    small = AnalysisOptions(bootstrap=10, draws=10, replicates=3, sizes=(11,), seed=3)
    out = analysis.analyze_run(noisy, small, figures=False, ranking_ballot="combined")
    assert out.name == f"{noisy.run_id}__ranking_combined"
    both = analysis.variant_id("r", ["m2", "m1"], "combined")
    assert both == "r__ranking_combined__ranking_without_m1_m2"
    assert set(analysis.read_outputs(out)["metrics"]["run_id"]) == {out.name}


def test_analysis_records_the_settings_that_produced_each_folder(noisy):
    import json

    small = AnalysisOptions(bootstrap=11, draws=12, replicates=3, sizes=(11,), seed=9)
    out = analysis.analyze_run(noisy, small, figures=False, ranking_ballot="combined")
    s = json.loads((out / analysis.SETTINGS_FILE).read_text())
    assert (s["bootstrap"], s["draws"], s["replicates"], s["seed"]) == (11, 12, 3, 9)
    assert s["ranking_ballot"] == "combined" and s["exclude_ranking"] == []
    assert s["run_id"] == noisy.run_id and s["variant"] == out.name
    assert s["n_respondents"] == 150 and set(s["models"]) == set(MODELS)
    assert s["sizes"] == [11] and s["written"].startswith("20")
    analysis.read_outputs(out)  # the extra file does not disturb reading the tables


def test_paired_differences_from_the_human_electorate(table, tables):
    """`<metric>_minus_human` rows: the same resampled voters are used for the human and
    the LLM electorate, so identical ballots give a difference of exactly zero in every
    resample (unpaired resampling would not)."""
    small = AnalysisOptions(bootstrap=40, draws=10, replicates=2, sizes=(11,), seed=3)
    import dataclasses

    twin = dataclasses.replace(table, llm_scores={"tiny": table.human.copy()}, llm_ranks={})
    m = analysis.analyze(twin, "x", None, small)["metrics"]
    d = m[m["metric"].str.endswith(analysis.MINUS_HUMAN) & (m["source"] == "thermometer")]
    assert set(d["metric"]) == {"sp_deletion_share_minus_human", "pc1_share_minus_human"}
    assert set(d["electorate"]) == {"tiny"} and set(d["interval"]) == {"paired_bootstrap"}
    np.testing.assert_allclose(d[["value", "ci_low", "ci_high", "sd"]].to_numpy(), 0, atol=1e-12)

    # with noise: the point difference is the difference of the two point values, and the
    # paired interval is tighter than the two separate intervals would suggest
    t = tables["metrics"]
    key = (t["weighting"] == "weighted") & (t["source"] == "thermometer")
    one = lambda e, name: t[key & (t["electorate"] == e) & (t["metric"] == name)].iloc[0]  # noqa: E731
    for model in MODELS:
        diff = one(model, "sp_deletion_share_minus_human")
        a, h = one(model, "sp_deletion_share"), one("human", "sp_deletion_share")
        assert diff["value"] == pytest.approx(a["value"] - h["value"])
        assert diff["ci_low"] <= diff["ci_high"] and diff["interval"] == "paired_bootstrap"
        assert diff["sd"] < np.hypot(a["sd"], h["sd"])
    assert not t[t["electorate"] == "human"]["metric"].str.endswith("_minus_human").any()
    # ranking-source electorates are compared with the human thermometer ballots too
    r = t[(t["source"] == "ranking") & t["metric"].str.endswith("_minus_human")]
    assert set(r["metric"]) == {"sp_deletion_share_minus_human"}
