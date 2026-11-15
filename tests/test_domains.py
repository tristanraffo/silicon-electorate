"""SPEC §6: domain diagnostics."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from sklearn.decomposition import PCA

from helpers import profile_from, random_weak_profile
from scv.ballots import WeakOrder
from scv.domains import (
    all_axes,
    canonical_axis,
    dimensionality,
    diversity,
    is_single_peaked,
    is_single_plateaued,
    kendall_tau_distance,
    single_peakedness,
    sp_axes,
    sp_consistency,
    sp_deletion_share,
    sp_max_consistent,
    value_restriction,
)
from scv.nulls import impartial_culture, spatial, spatial_1d
from scv.rules import Profile

P = WeakOrder.parse


@pytest.fixture
def paradox() -> Profile:
    return profile_from([(1, "A>B>C"), (1, "B>C>A"), (1, "C>A>B")], "ABC")


# --------------------------------------------------------------------------------------
# Axes
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("m", [3, 4, 5, 6, 7])
def test_all_axes_enumerates_m_factorial_over_two(m):
    cands = [chr(ord("A") + i) for i in range(m)]
    axes = all_axes(cands)
    assert len(axes) == math.factorial(m) // 2  # 2,520 at m = 7
    assert len(set(axes)) == len(axes)
    assert all(sorted(a) == cands for a in axes)
    assert not any(a[::-1] in set(axes) for a in axes)  # one orientation each


def test_canonical_axis():
    assert canonical_axis("CBA", "ABC") == ("A", "B", "C")
    assert canonical_axis("ABC", "ABC") == ("A", "B", "C")
    assert canonical_axis(("B", "A", "C"), "ABC") in all_axes("ABC")
    with pytest.raises(ValueError):
        canonical_axis("AB", "ABC")
    with pytest.raises(ValueError, match="limited to 7"):
        all_axes("ABCDEFGH")


# --------------------------------------------------------------------------------------
# One ballot on one axis
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ballot, axis, peaked, plateaued",
    [
        # strict ballots: the two notions coincide
        ("A>B>C", "ABC", True, True),
        ("B>A>C", "ABC", True, True),
        ("B>C>A", "ABC", True, True),
        ("A>C>B", "ABC", False, False),  # B sits between A and C but is ranked below both
        ("C>A>B", "ABC", False, False),
        ("B>C>A>D", "ABCD", True, True),
        ("B>D>C>A", "ABCD", False, False),  # jumps over C
        # weak ballots
        ("A>B=C", "BAC", True, True),  # peak A, one tied candidate on each side
        ("A>B=C", "ABC", True, False),  # extension A>B>C is single-peaked; B=C on one side
        ("A=B>C", "ABC", True, True),  # plateau {A,B} is contiguous
        ("A=C>B", "ABC", False, False),  # top set {A,C} is split by B: no extension works
        ("A=B=C", "ABC", True, True),  # indifferent: constrains nothing
        ("B>A=C>D", "ABCD", True, True),  # A=C tie spans the two sides of the peak: allowed
        ("B>A=C>D", "DABC", True, True),
        ("B>C=D>A", "ABCD", True, False),  # C=D tie on the same side of the peak
    ],
)
def test_single_peaked_and_plateaued_by_hand(ballot, axis, peaked, plateaued):
    b = P(ballot)
    assert is_single_peaked(b, axis) is peaked
    assert is_single_plateaued(b, axis) is plateaued
    # an axis and its reversal are the same axis
    assert is_single_peaked(b, axis[::-1]) is peaked
    assert is_single_plateaued(b, axis[::-1]) is plateaued


def test_missing_candidates_are_skipped_on_the_axis():
    b = P("A>C", missing={"B"})
    assert is_single_peaked(b, "ABC") and is_single_plateaued(b, "ABC")
    # With B unranked, A>D>C is judged on the axis A-C-D, where C is a valley.
    b = P("A>D>C", missing={"B"})
    assert not is_single_peaked(b, "ABCD")
    assert is_single_peaked(b, "ABDC")
    with pytest.raises(ValueError, match="does not contain"):
        is_single_peaked(P("A>Z"), "ABC")


def test_strict_ballots_are_plateaued_exactly_when_peaked():
    cands = list("ABCDE")
    for perm in itertools.permutations(cands):
        b = WeakOrder.strict(perm)
        for axis in all_axes(cands)[::7]:
            assert is_single_peaked(b, axis) == is_single_plateaued(b, axis)


def test_number_of_single_peaked_orders_on_an_axis_is_two_to_the_m_minus_one():
    # Classical count: exactly 2^(m-1) strict orders are single-peaked on a given axis.
    for m in (3, 4, 5, 6):
        cands = [chr(ord("A") + i) for i in range(m)]
        count = sum(
            is_single_peaked(WeakOrder.strict(p), cands) for p in itertools.permutations(cands)
        )
        assert count == 2 ** (m - 1)


# --------------------------------------------------------------------------------------
# Whole profiles: the vectorised path must agree with the definitions
# --------------------------------------------------------------------------------------


def test_consistency_matrix_matches_the_ballot_level_definitions():
    rng = np.random.default_rng(0)
    checked = disagreements_between_modes = 0
    for _ in range(25):
        m = int(rng.integers(3, 7))
        ballots, cands = random_weak_profile(rng, 12, m)
        p = Profile(ballots, cands)
        axes = all_axes(cands)
        peaked = sp_consistency(p, "peaked")
        plateaued = sp_consistency(p, "plateaued")
        assert peaked.shape == plateaued.shape == (12, len(axes))
        for i, b in enumerate(ballots):
            for j, axis in enumerate(axes):
                assert peaked[i, j] == is_single_peaked(b, axis), (str(b), axis)
                assert plateaued[i, j] == is_single_plateaued(b, axis), (str(b), axis)
                checked += 1
        assert not (plateaued & ~peaked).any()  # plateaued is the stricter check
        disagreements_between_modes += int((peaked & ~plateaued).sum())
    assert checked > 10_000 and disagreements_between_modes > 100


def test_consistency_matrix_on_strict_profiles_at_m7():
    p = impartial_culture(40, 7, rng=0)
    mat = sp_consistency(p)
    axes = all_axes(p.candidates)
    assert mat.shape == (40, 2520)
    assert np.array_equal(mat, sp_consistency(p, "plateaued"))
    # spot-check against the ballot-level definition
    for i in (0, 13, 39):
        for j in range(0, 2520, 97):
            assert mat[i, j] == is_single_peaked(p.ballots[i], axes[j])


def test_condorcet_paradox_is_not_single_peaked(paradox):
    assert sp_axes(paradox) == []
    # Each axis has one candidate in the middle, and the voter who ranks it last fails.
    axis, share = sp_max_consistent(paradox)
    assert share == pytest.approx(2 / 3)
    assert axis == ("A", "B", "C")  # first of the three tied axes
    assert sp_deletion_share(paradox) == pytest.approx(1 / 3)


def test_sp_shares_are_weight_shares():
    # 3 x A>B>C fails only on the axis with C in the middle; B>C>A fails with A in the
    # middle; C>A>B fails with B in the middle. Best axis keeps weight 3 + 1 of 5.
    p = profile_from([(3, "A>B>C"), (1, "B>C>A"), (1, "C>A>B")], "ABC")
    axis, share = sp_max_consistent(p)
    assert share == pytest.approx(4 / 5)
    assert axis in [("A", "B", "C"), ("B", "A", "C")]
    assert sp_deletion_share(p) == pytest.approx(1 / 5)
    # zero-weight ballots constrain nothing
    q = profile_from([(1, "A>B>C"), (1, "B>C>A"), (0, "C>A>B")], "ABC")
    assert sp_deletion_share(q) == 0.0 and sp_axes(q) == [("A", "B", "C")]


def test_report_gives_both_checks():
    # Everyone agrees on A first; B=C tie. Peaked on any axis via an extension, but
    # plateaued only where B and C are on opposite sides of A.
    p = profile_from([(2, "A>B=C"), (1, "A>B>C")], "ABC")
    rep = single_peakedness(p)
    assert rep.deletion_share == 0.0 and rep.share == 1.0
    assert rep.n_axes_all_ballots == 2  # A-B-C and B-A-C; A-C-B has C between A and B: fails
    assert rep.plateau_deletion_share == 0.0
    assert rep.plateau_axis == ("B", "A", "C") and rep.plateau_n_axes_all_ballots == 1
    with pytest.raises(ValueError, match="mode"):
        sp_axes(p, mode="strict")  # type: ignore[arg-type]


def test_impartial_culture_has_a_high_deletion_share():
    """SPEC §6 test: IC with m = 5, n = 1,000 -> deletion share > 0.3."""
    p = impartial_culture(1_000, 5, rng=0)
    share = sp_deletion_share(p)
    print(f"IC SP deletion share (m=5, n=1000, seed=0): {share:.4f}")
    assert share > 0.3
    # Only 2^(m-1) = 16 of 120 orders fit any one axis, so the expectation is near 1 - 16/120.
    assert share == pytest.approx(1 - 16 / 120, abs=0.06)


# --------------------------------------------------------------------------------------
# Sen's value restriction
# --------------------------------------------------------------------------------------


def test_condorcet_paradox_fails_value_restriction(paradox):
    """SPEC §6 test. Each candidate is best once, middle once and worst once."""
    vr = value_restriction(paradox)
    assert vr.triples == (("A", "B", "C"),)
    assert vr.satisfied == (False,)
    assert vr.share == 0.0
    assert vr.min_violation == (pytest.approx(1 / 3),)  # drop any one voter to restore it
    assert vr.mean_min_violation == pytest.approx(1 / 3)


def test_value_restriction_holds_on_single_peaked_profiles():
    # Single-peaked => the candidate in the middle of each triple is never worst.
    for seed in range(10):
        vr = value_restriction(spatial_1d(200, 3 + seed % 5, rng=seed).profile)
        assert vr.share == 1.0 and all(vr.satisfied)
        assert vr.mean_min_violation == 0.0


def test_value_restriction_counts_triples_and_uses_weights():
    # A,B,C form a cycle; D is last on every ballot, so each triple containing D
    # has D never best. 1 of 4 triples fails.
    p = profile_from([(5, "A>B>C>D"), (3, "B>C>A>D"), (2, "C>A>B>D")], "ABCD")
    vr = value_restriction(p)
    assert len(vr.triples) == 4
    assert dict(zip(vr.triples, vr.satisfied, strict=True)) == {
        ("A", "B", "C"): False,
        ("A", "B", "D"): True,
        ("A", "C", "D"): True,
        ("B", "C", "D"): True,
    }
    assert vr.share == pytest.approx(3 / 4)
    # Cheapest repair of {A,B,C}: remove the weight-2 ballot (then e.g. A is never middle).
    assert vr.min_violation[0] == pytest.approx(2 / 10)
    assert vr.mean_min_violation == pytest.approx(0.2 / 4)


def test_value_restriction_ignores_what_a_ballot_does_not_strictly_decide():
    # The third voter of the paradox now ties everything / omits a candidate: only
    # strictly determined positions count, so the triple is value-restricted again.
    tied = profile_from([(1, "A>B>C"), (1, "B>C>A"), (1, "A=B=C")], "ABC")
    assert value_restriction(tied).satisfied == (True,)
    partial = Profile([P("A>B>C"), P("B>C>A"), P("C>A", missing={"B"})], candidates=["A", "B", "C"])
    assert value_restriction(partial).satisfied == (True,)
    # ...whereas C > A = B does put C best, and that completes the violation for "best"
    # only: A best (v1), B best (v2), C best (v3); but A is never middle -> still satisfied.
    top_only = profile_from([(1, "A>B>C"), (1, "B>C>A"), (1, "C>A=B")], "ABC")
    assert value_restriction(top_only).satisfied == (True,)


def test_value_restriction_matches_brute_force():
    def brute(p: Profile) -> list[bool]:
        out = []
        for trio in itertools.combinations(p.candidates, 3):
            seen = set()  # (candidate, position) cells occupied by some ballot
            for b, w in zip(p.ballots, p.weights, strict=True):
                rank = b.rank_of()
                if w <= 0:
                    continue
                for x in trio:
                    y, z = (c for c in trio if c != x)
                    if not all(c in rank for c in trio):
                        continue
                    if rank[x] < rank[y] and rank[x] < rank[z]:
                        seen.add((x, "best"))
                    if rank[x] > rank[y] and rank[x] > rank[z]:
                        seen.add((x, "worst"))
                    if rank[y] < rank[x] < rank[z] or rank[z] < rank[x] < rank[y]:
                        seen.add((x, "middle"))
            out.append(len(seen) < 9)
        return out

    rng = np.random.default_rng(3)
    n_true = n_false = 0
    for trial in range(300):
        n, m = int(rng.integers(2, 9)), int(rng.integers(3, 6))
        if trial % 2:  # strict complete ballots: where violations are common
            ic = impartial_culture(n, m, rng, candidates=[chr(ord("A") + i) for i in range(m)])
            ballots, cands = list(ic.ballots), list(ic.candidates)
        else:  # ties and missing candidates
            ballots, cands = random_weak_profile(rng, n, m)
        w = rng.integers(0, 3, size=len(ballots)).astype(float)
        if w.sum() == 0:
            continue
        p = Profile(ballots, cands, w)
        got = value_restriction(p)
        assert list(got.satisfied) == brute(p)
        n_true += sum(got.satisfied)
        n_false += len(got.satisfied) - sum(got.satisfied)
    assert n_true > 50 and n_false > 50


# --------------------------------------------------------------------------------------
# Dimensionality
# --------------------------------------------------------------------------------------


def test_dimensionality_matches_sklearn_pca_on_voter_centred_scores():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(300, 6)) @ rng.normal(size=(6, 6)) + 50
    d = dimensionality(scores)
    centred = scores - scores.mean(axis=1, keepdims=True)
    ref = PCA().fit(centred).explained_variance_ratio_
    assert d.explained == pytest.approx(tuple(ref), abs=1e-10)
    assert d.pc1_share == pytest.approx(ref[0]) and d.pc2_share == pytest.approx(ref[1])
    assert d.n_used == 300 and d.n_dropped == 0
    assert sum(d.explained) == pytest.approx(1.0)
    assert d.explained[-1] == pytest.approx(0.0, abs=1e-10)  # centring removes one dimension


def test_dimensionality_one_latent_dimension_gives_pc1_share_one():
    rng = np.random.default_rng(1)
    loadings = np.array([-2.0, -1.0, 0.5, 3.0])
    voters = rng.normal(size=(200, 1))
    level = rng.normal(size=(200, 1)) * 10  # "rates everyone high/low": removed by centring
    scores = voters * loadings + level
    d = dimensionality(scores)
    assert d.pc1_share == pytest.approx(1.0) and d.pc2_share == pytest.approx(0.0, abs=1e-9)

    two = scores + rng.normal(size=(200, 1)) * np.array([1.0, -1.0, 1.0, -1.0])
    assert dimensionality(two).pc1_share < 0.999
    assert sum(dimensionality(two).explained[:2]) == pytest.approx(1.0)


def test_dimensionality_is_lower_dimensional_for_1d_than_2d_spatial():
    one = dimensionality(spatial(2_000, 6, rng=0, dim=1).scores)
    two = dimensionality(spatial(2_000, 6, rng=0, dim=2).scores)
    assert one.pc1_share > two.pc1_share
    assert one.pc1_share > 0.8


def test_dimensionality_weights_and_missing():
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(50, 4))
    k = rng.integers(1, 4, size=50)
    weighted = dimensionality(scores, weights=k)
    duplicated = dimensionality(np.repeat(scores, k, axis=0))
    assert weighted.explained == pytest.approx(duplicated.explained)

    holes = scores.copy()
    holes[[3, 7], [0, 2]] = np.nan
    d = dimensionality(holes, weights=np.where(np.arange(50) == 10, 0.0, 1.0))
    assert d.n_used == 47 and d.n_dropped == 3  # two with a missing score, one zero-weight
    keep = np.ones(50, bool)
    keep[[3, 7, 10]] = False
    assert d.explained == pytest.approx(dimensionality(scores[keep]).explained)


def test_dimensionality_degenerate_inputs_give_nan():
    assert math.isnan(dimensionality(np.array([[1.0, 2.0, 3.0]])).pc1_share)  # one voter
    same = np.tile([10.0, 50.0, 90.0], (20, 1))
    assert math.isnan(dimensionality(same).pc1_share)  # no variance across voters
    shifted = same + np.arange(20)[:, None]  # differ only in level: still no variance
    assert math.isnan(dimensionality(shifted).pc1_share)
    with pytest.raises(ValueError):
        dimensionality(np.zeros(4))
    with pytest.raises(ValueError):
        dimensionality(np.zeros((4, 3)), weights=[1, 2])


# --------------------------------------------------------------------------------------
# Diversity
# --------------------------------------------------------------------------------------


def test_kendall_tau_distance_by_hand():
    c = "ABC"
    assert kendall_tau_distance(P("A>B>C"), P("A>B>C"), c) == 0.0
    assert kendall_tau_distance(P("A>B>C"), P("C>B>A"), c) == 1.0
    assert kendall_tau_distance(P("A>B>C"), P("B>A>C"), c) == pytest.approx(1 / 3)
    assert kendall_tau_distance(P("A>B>C"), P("A>B=C"), c) == pytest.approx(0.5 / 3)
    assert kendall_tau_distance(P("A=B=C"), P("A=B=C"), c) == 0.0
    # C missing on one ballot: pairs (A,C) and (B,C) are undecided there -> 0.5 each
    assert kendall_tau_distance(P("A>B>C"), P("A>B", missing={"C"}), c) == pytest.approx(1 / 3)


def test_diversity_by_hand():
    same = diversity(profile_from([(1, "A>B>C")] * 4, "ABC"))
    assert (same.n_ballots, same.n_distinct) == (4, 1)
    assert same.distinct_share == 0.25 and same.entropy == 0.0 and same.mean_kendall_tau == 0.0

    opposite = diversity(profile_from([(1, "A>B>C"), (1, "C>B>A")], "ABC"))
    assert opposite.distinct_share == 1.0
    assert opposite.entropy == pytest.approx(1.0)  # one bit
    assert opposite.mean_kendall_tau == pytest.approx(1.0)

    # 3 x A>B>C and 1 x B>A>C: entropy H(1/4); 6 voter pairs, 3 of them at distance 1/3.
    mixed = diversity(profile_from([(1, "A>B>C")] * 3 + [(1, "B>A>C")], "ABC"))
    assert mixed.n_distinct == 2 and mixed.distinct_share == 0.5
    assert mixed.entropy == pytest.approx(-(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25)))
    assert mixed.mean_kendall_tau == pytest.approx((3 * (1 / 3)) / 6)

    alone = diversity(profile_from([(1, "A>B>C")], "ABC"))
    assert math.isnan(alone.mean_kendall_tau)


def test_diversity_matches_brute_force_with_ties_missing_and_weights():
    rng = np.random.default_rng(4)
    for _ in range(40):
        n, m = int(rng.integers(2, 25)), int(rng.integers(3, 7))
        ballots, cands = random_weak_profile(rng, n, m)
        w = rng.uniform(0.2, 3.0, size=n)
        w[rng.random(n) < 0.1] = 0.0
        if (w > 0).sum() < 2:
            continue
        p = Profile(ballots, cands, w)
        d = diversity(p)

        live = [i for i in range(n) if w[i] > 0]
        num = sum(
            w[i] * w[j] * kendall_tau_distance(ballots[i], ballots[j], cands)
            for i in live
            for j in live
            if i != j
        )
        den = sum(w[i] * w[j] for i in live for j in live if i != j)
        assert d.mean_kendall_tau == pytest.approx(num / den)

        mass: dict[WeakOrder, float] = {}
        for i in live:
            # a candidate the ballot never mentions is the same as one it lists as missing
            key = WeakOrder(ballots[i].groups, set(cands) - set(ballots[i].ranked))
            mass[key] = mass.get(key, 0.0) + w[i]
        probs = np.array(list(mass.values())) / w.sum()
        assert d.n_ballots == len(live) and d.n_distinct == len(mass)
        assert d.entropy == pytest.approx(float(-(probs * np.log2(probs)).sum()))


def test_ic_is_more_diverse_than_a_single_peaked_electorate():
    ic = diversity(impartial_culture(2_000, 5, rng=0))
    sp = diversity(spatial_1d(2_000, 5, rng=0).profile)
    assert ic.mean_kendall_tau == pytest.approx(0.5, abs=0.01)  # random orders: half discordant
    assert ic.n_distinct == 120 and sp.n_distinct <= 5 * 4 // 2 + 1  # at most C(m,2)+1 on a line
    assert ic.entropy > sp.entropy
    assert ic.mean_kendall_tau > sp.mean_kendall_tau
