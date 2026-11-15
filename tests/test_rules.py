"""SPEC §5: social choice engine. Expected values are worked by hand in the comments."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from helpers import expand, profile_from, random_weak_profile, single_peaked_profile
from scv.ballots import WeakOrder, is_strict
from scv.rules import (
    RULES,
    Profile,
    borda,
    condorcet_loser,
    condorcet_order,
    condorcet_winner,
    copeland,
    has_majority_cycle,
    irv,
    irv_monotonicity_probe,
    kemeny,
    majority_cycles,
    minimax,
    pairwise_matrix,
    plurality,
    rule_disagreement,
    run_all_rules,
    smith_set,
    three_cycles,
)


def matrix(profile: Profile) -> dict[tuple[str, str], float]:
    n = pairwise_matrix(profile)
    c = profile.candidates
    return {(c[i], c[j]): n[i, j] for i in range(len(c)) for j in range(len(c)) if i != j}


# --------------------------------------------------------------------------------------
# Required: Condorcet paradox
# --------------------------------------------------------------------------------------


@pytest.fixture
def paradox() -> Profile:
    return profile_from([(1, "A>B>C"), (1, "B>C>A"), (1, "C>A>B")], "ABC")


def test_condorcet_paradox(paradox):
    # A beats B 2-1, B beats C 2-1, C beats A 2-1.
    assert matrix(paradox) == {
        ("A", "B"): 2, ("B", "A"): 1,
        ("B", "C"): 2, ("C", "B"): 1,
        ("C", "A"): 2, ("A", "C"): 1,
    }  # fmt: skip
    assert has_majority_cycle(paradox) is True
    assert condorcet_winner(paradox) is None
    assert condorcet_loser(paradox) is None
    assert smith_set(paradox) == {"A", "B", "C"}
    assert three_cycles(paradox) == [("A", "B", "C")]
    assert condorcet_order(paradox) is None

    report = majority_cycles(paradox)
    assert report.has_cycle is True
    assert report.smith_set == {"A", "B", "C"}
    assert report.smith_size == 3
    assert report.three_cycles == (("A", "B", "C"),)


def test_condorcet_paradox_kemeny_reports_every_optimum(paradox):
    # Scores: A>B>C = 2+1+2 = 5, likewise B>C>A and C>A>B; the three reversed orders score 4.
    k = kemeny(paradox)
    assert k.details["score"] == 5
    assert k.details["unique"] is False
    assert k.details["optimal_rankings"] == [("A", "B", "C"), ("B", "C", "A"), ("C", "A", "B")]
    assert k.winners == ("A", "B", "C")
    assert k.winner is None
    assert k.ranking == WeakOrder.parse("A>B>C")  # documented: first in candidate order


# --------------------------------------------------------------------------------------
# Required: Borda != Condorcet (2 voters R>G>B, 3 voters B>R>G)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("build", [profile_from, expand])
def test_borda_differs_from_condorcet(build):
    p = build([(2, "R>G>B"), (3, "B>R>G")], ["R", "G", "B"])
    # Pairwise: B beats R 3-2, R beats G 5-0, B beats G 3-2.
    assert matrix(p) == {
        ("B", "R"): 3, ("R", "B"): 2,
        ("R", "G"): 5, ("G", "R"): 0,
        ("B", "G"): 3, ("G", "B"): 2,
    }  # fmt: skip
    assert condorcet_winner(p) == "B"
    assert condorcet_loser(p) == "G"
    assert has_majority_cycle(p) is False
    assert smith_set(p) == {"B"}
    assert condorcet_order(p) == WeakOrder.parse("B>R>G")

    # Borda 2/1/0: R = 2*2 + 3*1 = 7, B = 2*0 + 3*2 = 6, G = 2*1 + 3*0 = 2.
    b = borda(p)
    assert b.scores == {"R": 7.0, "G": 2.0, "B": 6.0}
    assert b.winner == "R"
    assert b.ranking == WeakOrder.parse("R>B>G")

    results = run_all_rules(p)
    assert results["plurality"].winner == "B"  # B 3, R 2, G 0
    assert results["plurality"].scores == {"R": 2.0, "G": 0.0, "B": 3.0}
    for rule in ("copeland", "minimax", "kemeny", "irv"):
        assert results[rule].winner == "B", rule
    d = rule_disagreement(results)
    assert d.differs("borda", "copeland") and d.differs("borda", "kemeny")
    assert not d.differs("plurality", "irv")
    assert d.share == pytest.approx(5 / 15)  # Borda against each of the other five


# --------------------------------------------------------------------------------------
# One profile, every rule, all by hand
# --------------------------------------------------------------------------------------


@pytest.fixture
def lopsided_cycle() -> Profile:
    # 4: A>B>C, 3: B>C>A, 2: C>A>B.
    # A beats B 6-3, B beats C 7-2, C beats A 5-4: a cycle with unequal margins 3, 5, 1.
    return profile_from([(4, "A>B>C"), (3, "B>C>A"), (2, "C>A>B")], "ABC")


def test_every_rule_on_a_lopsided_cycle(lopsided_cycle):
    p = lopsided_cycle
    assert matrix(p) == {
        ("A", "B"): 6, ("B", "A"): 3,
        ("B", "C"): 7, ("C", "B"): 2,
        ("C", "A"): 5, ("A", "C"): 4,
    }  # fmt: skip
    assert condorcet_winner(p) is None and condorcet_loser(p) is None
    assert smith_set(p) == {"A", "B", "C"}

    pl = plurality(p)
    assert pl.scores == {"A": 4.0, "B": 3.0, "C": 2.0}
    assert pl.winner == "A" and pl.ranking == WeakOrder.parse("A>B>C")

    # Borda: A = 4*2 + 2*1 = 10, B = 4*1 + 3*2 = 10, C = 3*1 + 2*2 = 7. A and B tie.
    bo = borda(p)
    assert bo.scores == {"A": 10.0, "B": 10.0, "C": 7.0}
    assert bo.winners == ("A", "B") and bo.winner is None
    assert bo.ranking == WeakOrder.parse("A=B>C")

    # Copeland: everyone has one win and one loss.
    co = copeland(p)
    assert co.scores == {"A": 1.0, "B": 1.0, "C": 1.0}
    assert co.winners == ("A", "B", "C") and co.winner is None

    # Minimax, worst margin: A loses to C by 1, B loses to A by 3, C loses to B by 5.
    mm = minimax(p)
    assert mm.scores == {"A": -1.0, "B": -3.0, "C": -5.0}
    assert mm.winner == "A" and mm.ranking == WeakOrder.parse("A>B>C")

    # Kemeny: A>B>C = 6+4+7 = 17; B>C>A = 7+3+5 = 15; B>A>C = 3+7+4 = 14;
    #         C>A>B = 5+2+6 = 13; A>C>B = 4+6+2 = 12; C>B>A = 2+5+3 = 10.
    ke = kemeny(p)
    assert ke.details["score"] == 17 and ke.details["unique"] is True
    assert ke.winner == "A" and ke.ranking == WeakOrder.parse("A>B>C")

    # IRV: round 1 A 4, B 3, C 2 -> C out; its 2 ballots go to A; round 2 A 6, B 3.
    ir = irv(p)
    assert ir.winner == "A" and ir.ranking == WeakOrder.parse("A>B>C")
    assert ir.details["rounds"] == [
        {"round": 1, "tallies": {"A": 4.0, "B": 3.0, "C": 2.0}, "eliminated": "C"},
        {"round": 2, "tallies": {"A": 6.0, "B": 3.0}, "eliminated": "B"},
    ]
    assert ir.details["tie_breaks"] == [] and ir.details["winner_by_tiebreak"] is False


def test_rule_disagreement_matrix(lopsided_cycle):
    results = run_all_rules(lopsided_cycle)
    assert list(results) == list(RULES) == [
        "plurality", "borda", "copeland", "minimax", "kemeny", "irv",
    ]  # fmt: skip
    d = rule_disagreement(results)
    assert d.rules == tuple(RULES)
    assert d.matrix.dtype == bool and d.matrix.shape == (6, 6)
    assert np.array_equal(d.matrix, d.matrix.T)
    assert not d.matrix.diagonal().any()
    # Winner sets: {A} for plurality/minimax/kemeny/irv, {A,B} for Borda, {A,B,C} for Copeland.
    agree = {"plurality", "minimax", "kemeny", "irv"}
    for a in d.rules:
        for b in d.rules:
            assert d.differs(a, b) == (a != b and not {a, b} <= agree), (a, b)
    assert d.share == pytest.approx(9 / 15)


# --------------------------------------------------------------------------------------
# Majority relation: Smith set, cycles, ties
# --------------------------------------------------------------------------------------


def test_smith_set_can_be_a_proper_subset_and_exclude_the_condorcet_loser():
    p = profile_from([(1, "A>B>C>D"), (1, "B>C>A>D"), (1, "C>A>B>D")], "ABCD")
    assert smith_set(p) == {"A", "B", "C"}
    assert condorcet_loser(p) == "D"
    assert condorcet_winner(p) is None
    assert three_cycles(p) == [("A", "B", "C")]


def test_cycle_below_a_condorcet_winner():
    # A tops every ballot. Below it: B beats C (v1, v3), C beats D (v1, v2), D beats B (v2, v3).
    p = profile_from([(1, "A>B>C>D"), (1, "A>C>D>B"), (1, "A>D>B>C")], "ABCD")
    assert condorcet_winner(p) == "A"
    assert smith_set(p) == {"A"}
    assert has_majority_cycle(p) is True
    assert three_cycles(p) == [("B", "C", "D")]
    assert condorcet_order(p) is None


def test_three_cycle_orientation_against_candidate_order():
    # Same cycle as the paradox but running A -> C -> B -> A.
    p = profile_from([(1, "A>C>B"), (1, "C>B>A"), (1, "B>A>C")], "ABC")
    assert three_cycles(p) == [("A", "C", "B")]


def test_four_cycle_with_pairwise_ties_has_no_three_cycle():
    # The four rotations of A>B>C>D. A beats B, B beats C, C beats D, D beats A (3-1 each);
    # A vs C and B vs D are 2-2 ties. A cycle exists, but no 3-cycle does.
    p = profile_from([(1, "A>B>C>D"), (1, "B>C>D>A"), (1, "C>D>A>B"), (1, "D>A>B>C")], "ABCD")
    m = matrix(p)
    assert (m[("A", "B")], m[("B", "A")]) == (3, 1)
    assert (m[("D", "A")], m[("A", "D")]) == (3, 1)
    assert (m[("A", "C")], m[("C", "A")]) == (2, 2)
    assert (m[("B", "D")], m[("D", "B")]) == (2, 2)
    assert has_majority_cycle(p) is True
    assert three_cycles(p) == []
    assert smith_set(p) == {"A", "B", "C", "D"}
    assert condorcet_order(p) is None


def test_transitive_majority_with_a_tie_gives_a_weak_condorcet_order():
    p = profile_from([(1, "A>B>C"), (1, "B>A>C")], "ABC")  # A ties B 1-1; both beat C 2-0
    assert condorcet_order(p) == WeakOrder.parse("A=B>C")
    assert condorcet_winner(p) is None
    assert condorcet_loser(p) == "C"
    assert smith_set(p) == {"A", "B"}
    assert has_majority_cycle(p) is False
    assert copeland(p).scores == {"A": 1.5, "B": 1.5, "C": 0.0}
    assert copeland(p).winners == ("A", "B")


def test_intransitive_indifference_is_not_a_condorcet_order_but_not_a_cycle():
    # A beats C 2-0, but A ties B and B ties C: the weak relation is not transitive.
    p = profile_from([(1, "A>C>B"), (1, "B>A>C")], "ABC")
    assert condorcet_order(p) is None
    assert has_majority_cycle(p) is False
    assert three_cycles(p) == []
    assert smith_set(p) == {"A", "B", "C"}  # no smaller set beats all outsiders strictly


def test_float_noise_in_weights_does_not_invent_a_majority():
    # 0.1 + 0.2 = 0.30000000000000004 in floating point; this is still a tie with 0.3.
    p = profile_from([(0.1, "A>B"), (0.2, "A>B"), (0.3, "B>A")], "AB")
    assert pairwise_matrix(p)[0, 1] != pairwise_matrix(p)[1, 0]
    assert condorcet_winner(p) is None
    assert copeland(p).winners == ("A", "B")
    assert minimax(p).winners == ("A", "B")
    assert borda(p).winners == ("A", "B")


# --------------------------------------------------------------------------------------
# Ties and missing candidates inside ballots
# --------------------------------------------------------------------------------------


def test_missing_and_tied_comparisons_count_for_neither_side():
    p = Profile(
        [WeakOrder.parse("A>B"), WeakOrder.parse("C>A=B"), WeakOrder((), missing={"A"})],
        candidates=["A", "B", "C"],
    )
    # Ballot 1 never mentions C (treated as missing); ballot 2 ties A and B; ballot 3 is empty.
    assert matrix(p) == {
        ("A", "B"): 1, ("B", "A"): 0,
        ("A", "C"): 0, ("C", "A"): 1,
        ("B", "C"): 0, ("C", "B"): 1,
    }  # fmt: skip
    assert np.isnan(p.ranks[0, 2]) and np.isnan(p.ranks[2]).all()


def test_plurality_splits_tied_tops_and_ignores_empty_ballots():
    p = Profile(
        [
            WeakOrder.parse("A=B>C"),
            WeakOrder.parse("C>A>B"),
            WeakOrder.parse("A>B>C"),
            WeakOrder((), missing={"A", "B", "C"}),
        ],
        candidates=["A", "B", "C"],
    )
    assert plurality(p).scores == {"A": 1.5, "B": 0.5, "C": 1.0}


def test_borda_weak_order_scoring():
    p = profile_from([(1, "A>B=C")], "ABC")
    assert borda(p).scores == {"A": 2.0, "B": 0.5, "C": 0.5}


def test_borda_treats_a_missing_comparison_as_a_tie():
    # A>B with C missing: A = 1 (beats B) + 0.5 (C), B = 0 + 0.5, C = 0.5 + 0.5.
    p = Profile([WeakOrder.parse("A>B", missing={"C"})], candidates=["A", "B", "C"])
    assert borda(p).scores == {"A": 1.5, "B": 0.5, "C": 1.0}


def test_irv_tied_tops_split_then_transfer_in_full():
    # Round 1: A 3, B 1 (half of two B=C ballots), C 3 -> B out.
    # Round 2: the B=C ballots now count in full for C: A 3, C 4.
    p = profile_from([(3, "A>B>C"), (2, "B=C>A"), (2, "C>B>A")], "ABC")
    r = irv(p)
    assert r.details["rounds"][0]["tallies"] == {"A": 3.0, "B": 1.0, "C": 3.0}
    assert r.details["rounds"][1]["tallies"] == {"A": 3.0, "C": 4.0}
    assert r.winner == "C" and r.ranking == WeakOrder.parse("C>A>B")
    assert r.details["tie_breaks"] == []


def test_irv_exhausted_ballots_drop_out():
    p = Profile(
        [WeakOrder.parse("A", missing={"B", "C"})] * 2
        + [WeakOrder.parse("B>C", missing={"A"})] * 3
        + [WeakOrder.parse("C>A>B")] * 4,
        candidates=["A", "B", "C"],
    )
    # Round 1: A 2, B 3, C 4 -> A out; the two A-only ballots are exhausted.
    # Round 2: B 3, C 4 -> C wins.
    r = irv(p)
    assert r.details["rounds"] == [
        {"round": 1, "tallies": {"A": 2.0, "B": 3.0, "C": 4.0}, "eliminated": "A"},
        {"round": 2, "tallies": {"B": 3.0, "C": 4.0}, "eliminated": "B"},
    ]
    assert r.winner == "C"


def test_irv_ties_are_broken_by_seed_deterministically_and_logged(caplog):
    # Round 1: A 2, B 1, C 1 -> B and C tie for elimination.
    p = profile_from([(2, "A>B>C"), (1, "B>C>A"), (1, "C>B>A")], "ABC")
    with caplog.at_level(logging.INFO, logger="scv.rules"):
        r = irv(p, seed=7)
    first = r.details["tie_breaks"][0]
    assert first["round"] == 1 and first["tied"] == ("B", "C")
    assert first["eliminated"] in {"B", "C"}
    assert any("IRV tie" in rec.getMessage() for rec in caplog.records)
    # The survivor then ties A 2-2 in the final round, so the winner is a tie-break too.
    assert r.details["winner_by_tiebreak"] is True
    assert len(r.details["tie_breaks"]) == 2

    assert irv(p, seed=7) == r  # same seed, same everything
    eliminated_first = {irv(p, seed=s).details["tie_breaks"][0]["eliminated"] for s in range(20)}
    assert eliminated_first == {"B", "C"}  # the seed, not candidate order, decides


# --------------------------------------------------------------------------------------
# Required: IRV monotonicity failure
# --------------------------------------------------------------------------------------
#
# Source: constructed for this test and verified by hand below. It is the standard
# three-candidate form of the "more-is-less" (non-monotonicity) paradox of the
# alternative vote; see Doron & Kronick (1977), "Single Transferrable Vote: An Example
# of a Perverse Social Choice Function", American Journal of Political Science 21(2),
# and Fishburn & Brams (1983), "Paradoxes of Preferential Voting", Mathematics
# Magazine 56(4). The numbers here are ours, not copied from either paper.
#
# Before:  39 A>B>C, 35 B>C>A, 26 C>A>B
#   round 1: A 39, B 35, C 26 -> C out, its 26 go to A -> A 65, B 35.  A wins.
# After 10 of the B>C>A voters move A to the top (now A>B>C; nothing else changes):
#          49 A>B>C, 25 B>C>A, 26 C>A>B
#   round 1: A 49, B 25, C 26 -> B out, its 25 go to C -> C 51, A 49.  C wins.
# A gained support and lost.


def test_irv_monotonicity_failure_textbook_form():
    before = profile_from([(39, "A>B>C"), (35, "B>C>A"), (26, "C>A>B")], "ABC")
    after = profile_from([(49, "A>B>C"), (25, "B>C>A"), (26, "C>A>B")], "ABC")
    r0, r1 = irv(before), irv(after)
    assert r0.winner == "A"
    assert r0.details["rounds"][1]["tallies"] == {"A": 65.0, "B": 35.0}
    assert r1.winner == "C"
    assert r1.details["rounds"][0]["eliminated"] == "B"
    assert r1.details["rounds"][1]["tallies"] == {"A": 49.0, "C": 51.0}
    assert not r0.details["tie_breaks"] and not r1.details["tie_breaks"]


def test_irv_monotonicity_probe_detects_the_failure():
    # Same electorate, with the ten pivotal voters as their own row holding B>A>C, so that
    # raising A by ONE position on that row reproduces the "after" profile above.
    #   row 0: 39 A>B>C   A already first: not eligible
    #   row 1: 25 B>C>A   -> B>A>C: round 1 unchanged (39/35/26), C out, A 65-35. A still wins.
    #   row 2: 10 B>A>C   -> A>B>C: A 49, B 25, C 26 -> B out -> C 51, A 49. C wins: FAILURE.
    #   row 3: 26 C>A>B   -> A>C>B: A 65, B 35, C 0 -> A wins.
    p = profile_from([(39, "A>B>C"), (25, "B>C>A"), (10, "B>A>C"), (26, "C>A>B")], "ABC")
    assert irv(p).winner == "A"
    probe = irv_monotonicity_probe(p)
    assert probe.winner == "A"
    assert probe.n_ballots == 4
    assert probe.n_eligible == 3
    assert probe.failures == (2,)
    assert probe.n_failures == 1
    assert probe.failure_fraction == pytest.approx(1 / 3)
    assert probe.failure_fraction_all == pytest.approx(1 / 4)


def test_irv_monotonicity_probe_moves_one_row_so_single_voters_here_are_not_pivotal():
    # The same electorate as 100 unit ballots: no single voter raising A by one position
    # can flip the result (it takes the block of ten), so the probe reports zero.
    p = expand([(39, "A>B>C"), (25, "B>C>A"), (10, "B>A>C"), (26, "C>A>B")], "ABC")
    probe = irv_monotonicity_probe(p)
    assert probe.winner == "A"
    assert probe.n_ballots == 100
    assert probe.n_eligible == 61
    assert probe.n_failures == 0 and probe.failure_fraction == 0.0


def test_irv_monotonicity_probe_eligibility():
    # IRV winner is A (5.5 first-place weight of 8). Eligible: A tied for first (row 1) and
    # A ranked second (row 2). Not eligible: A sole first (row 0), A missing (row 3).
    p = Profile(
        [
            WeakOrder.parse("A>B>C"),
            WeakOrder.parse("A=B>C"),
            WeakOrder.parse("B>A>C"),
            WeakOrder.parse("B>C", missing={"A"}),
        ],
        candidates=["A", "B", "C"],
        weights=[5, 1, 1, 1],
    )
    probe = irv_monotonicity_probe(p)
    assert probe.winner == "A"
    assert probe.n_eligible == 2
    assert probe.n_failures == 0

    unanimous = profile_from([(3, "A>B>C")], "ABC")
    probe = irv_monotonicity_probe(unanimous)
    assert probe.n_eligible == 0 and probe.failure_fraction == 0.0


# --------------------------------------------------------------------------------------
# Required: Kemeny == Condorcet ranking when the majority relation is transitive
# --------------------------------------------------------------------------------------


def test_kemeny_equals_condorcet_ranking_on_200_single_peaked_profiles():
    rng = np.random.default_rng(20261115)
    for trial in range(200):
        m = int(rng.integers(3, 8))  # 3..7 candidates
        n = int(rng.choice([5, 11, 25, 51]))  # odd, so no pairwise majority ties
        p = single_peaked_profile(rng, n, m)

        order = condorcet_order(p)
        assert order is not None, f"trial {trial}: single-peaked majority must be transitive"
        assert is_strict(order) and len(order.ranked) == m
        assert has_majority_cycle(p) is False

        k = kemeny(p)
        assert k.details["unique"] is True, f"trial {trial}"
        assert k.ranking == order, f"trial {trial}: {k.ranking} != {order}"
        assert k.winner == condorcet_winner(p) == order.ranked[0]
        assert smith_set(p) == {order.ranked[0]}
        assert copeland(p).ranking == order
        assert minimax(p).winner == order.ranked[0]


def test_kemeny_refuses_more_than_seven_candidates():
    p = Profile([WeakOrder.strict(list("ABCDEFGH"))], candidates=list("ABCDEFGH"))
    with pytest.raises(ValueError, match="limited to 7"):
        kemeny(p)


# --------------------------------------------------------------------------------------
# Required: weights. Duplicating a ballot == giving it weight 2.
# --------------------------------------------------------------------------------------


def assert_same_outcome(p: Profile, q: Profile, seed: int) -> None:
    assert np.array_equal(pairwise_matrix(p), pairwise_matrix(q))
    assert condorcet_winner(p) == condorcet_winner(q)
    assert condorcet_loser(p) == condorcet_loser(q)
    assert majority_cycles(p) == majority_cycles(q)
    assert condorcet_order(p) == condorcet_order(q)
    rp, rq = run_all_rules(p, seed=seed), run_all_rules(q, seed=seed)
    for rule in RULES:
        a, b = rp[rule], rq[rule]
        assert a.winners == b.winners, rule
        assert a.ranking == b.ranking, rule
        if a.scores is not None:
            assert a.scores == pytest.approx(b.scores), rule
    assert rp["kemeny"].details["optimal_rankings"] == rq["kemeny"].details["optimal_rankings"]
    assert [r["eliminated"] for r in rp["irv"].details["rounds"]] == [
        r["eliminated"] for r in rq["irv"].details["rounds"]
    ]
    assert np.array_equal(rule_disagreement(rp).matrix, rule_disagreement(rq).matrix)


def test_duplicating_a_ballot_equals_weight_two():
    rng = np.random.default_rng(5)
    for trial in range(60):
        m = int(rng.integers(3, 8))
        n = int(rng.integers(3, 30))
        ballots, candidates = random_weak_profile(rng, n, m)
        extra = rng.integers(0, 3, size=n)  # 0, 1 or 2 extra copies of each ballot
        if trial == 0:
            extra[:] = 0
            extra[0] = 1  # the literal statement: one ballot, duplicated once

        duplicated = Profile(
            ballots + [b for b, k in zip(ballots, extra, strict=True) for _ in range(k)],
            candidates=candidates,
        )
        weighted = Profile(ballots, candidates=candidates, weights=1 + extra)
        assert duplicated.total_weight == weighted.total_weight
        assert_same_outcome(duplicated, weighted, seed=trial)


def test_results_are_invariant_to_rescaling_all_weights():
    rng = np.random.default_rng(11)
    for trial in range(30):
        ballots, candidates = random_weak_profile(rng, int(rng.integers(3, 30)), 5)
        w = rng.uniform(0.2, 3.0, size=len(ballots))
        p = Profile(ballots, candidates=candidates, weights=w)
        q = Profile(ballots, candidates=candidates, weights=w * 2.0)  # exact in binary
        rp, rq = run_all_rules(p, seed=trial), run_all_rules(q, seed=trial)
        for rule in RULES:
            assert rp[rule].winners == rq[rule].winners, rule
            assert rp[rule].ranking == rq[rule].ranking, rule
        assert majority_cycles(p) == majority_cycles(q)


def test_zero_weight_ballots_do_not_count():
    with_ghost = profile_from([(1, "A>B>C"), (1, "A>C>B"), (0, "C>B>A")], "ABC")
    without = profile_from([(1, "A>B>C"), (1, "A>C>B")], "ABC")
    assert np.array_equal(pairwise_matrix(with_ghost), pairwise_matrix(without))
    assert plurality(with_ghost).scores == plurality(without).scores


# --------------------------------------------------------------------------------------
# Profile validation
# --------------------------------------------------------------------------------------


def test_profile_validation():
    ab = WeakOrder.parse("A>B")
    with pytest.raises(ValueError, match="unknown candidates"):
        Profile([WeakOrder.parse("A>Z")], candidates=["A", "B"])
    with pytest.raises(ValueError, match="at least one ballot"):
        Profile([], candidates=["A", "B"])
    with pytest.raises(ValueError, match="duplicate"):
        Profile([ab], candidates=["A", "A", "B"])
    with pytest.raises(ValueError, match="at least 2"):
        Profile([WeakOrder.parse("A")], candidates=["A"])
    with pytest.raises(TypeError):
        Profile([ab], candidates="AB")
    with pytest.raises(ValueError, match="weights"):
        Profile([ab], candidates=["A", "B"], weights=[1, 2])
    with pytest.raises(ValueError, match="non-negative"):
        Profile([ab], candidates=["A", "B"], weights=[-1])
    with pytest.raises(ValueError, match="non-negative"):
        Profile([ab], candidates=["A", "B"], weights=[float("nan")])
    with pytest.raises(ValueError, match="positive"):
        Profile([ab], candidates=["A", "B"], weights=[0])


def test_profile_arrays_are_read_only():
    p = profile_from([(1, "A>B>C")], "ABC")
    for arr in (p.weights, p.ranks, p.pairwise, p.majority):
        with pytest.raises(ValueError):
            arr[0] = 1
