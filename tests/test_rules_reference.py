"""Cross-check the vectorised engine against slow, obviously-correct reference code.

Every reference below is written straight from the SPEC §5 definition, using
``scv.ballots.pairwise`` and plain loops, and shares no code with ``scv.rules``.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest

from helpers import random_weak_profile
from scv.ballots import WeakOrder, move_up, pairwise
from scv.rules import (
    Profile,
    borda,
    condorcet_loser,
    condorcet_winner,
    copeland,
    has_majority_cycle,
    irv,
    irv_monotonicity_probe,
    kemeny,
    minimax,
    pairwise_matrix,
    plurality,
    smith_set,
    three_cycles,
)


def ref_matrix(p: Profile) -> dict[tuple[str, str], Fraction]:
    n = {(a, b): Fraction(0) for a in p.candidates for b in p.candidates if a != b}
    for ballot, w in zip(p.ballots, p.weights, strict=True):
        for (a, b), sign in pairwise(ballot).items():
            if sign == 1:
                n[(a, b)] += Fraction(w)
    return n


def ref_beats(p: Profile) -> set[tuple[str, str]]:
    n = ref_matrix(p)
    return {(a, b) for (a, b) in n if n[(a, b)] > n[(b, a)]}


def ref_smith(p: Profile) -> frozenset[str]:
    beats = ref_beats(p)
    for size in range(1, p.m + 1):
        for subset in itertools.combinations(p.candidates, size):
            outside = [c for c in p.candidates if c not in subset]
            if all((s, t) in beats for s in subset for t in outside):
                return frozenset(subset)
    raise AssertionError("unreachable: the full set always qualifies")


def ref_has_cycle(p: Profile) -> bool:
    beats = ref_beats(p)
    for size in range(3, p.m + 1):
        for path in itertools.permutations(p.candidates, size):
            if all((path[i], path[(i + 1) % size]) in beats for i in range(size)):
                return True
    return False


def ref_plurality(p: Profile) -> dict[str, Fraction]:
    score = {c: Fraction(0) for c in p.candidates}
    for ballot, w in zip(p.ballots, p.weights, strict=True):
        if ballot.groups:
            top = ballot.groups[0]
            for c in top:
                score[c] += Fraction(w) / len(top)
    return score


def ref_borda(p: Profile) -> dict[str, Fraction]:
    """Per ballot: strictly beaten + 0.5 x tied, a missing comparison counting as a tie."""
    score = {c: Fraction(0) for c in p.candidates}
    for ballot, w in zip(p.ballots, p.weights, strict=True):
        rank = ballot.rank_of()
        for a in p.candidates:
            for b in p.candidates:
                if a == b:
                    continue
                if a in rank and b in rank and rank[a] < rank[b]:
                    score[a] += Fraction(w)
                elif a not in rank or b not in rank or rank[a] == rank[b]:
                    score[a] += Fraction(w) / 2
    return score


def ref_kemeny_score(p: Profile) -> Fraction:
    n = ref_matrix(p)
    return max(
        sum(n[(a, b)] for a, b in itertools.combinations(order, 2))
        for order in itertools.permutations(p.candidates)
    )


def ref_irv(p: Profile, seed: int) -> list[str]:
    """Elimination order (winner last), exact arithmetic, same documented tie-break."""
    seeded_order = [p.candidates[i] for i in np.random.default_rng(seed).permutation(p.m)]
    remaining = list(p.candidates)
    out: list[str] = []
    while len(remaining) > 1:
        tally = {c: Fraction(0) for c in remaining}
        for ballot, w in zip(p.ballots, p.weights, strict=True):
            for group in ballot.groups:
                top = [c for c in group if c in remaining]
                if top:
                    for c in top:
                        tally[c] += Fraction(w) / len(top)
                    break
        low = min(tally.values())
        tied = [c for c in remaining if tally[c] == low]
        loser = min(tied, key=seeded_order.index)
        remaining.remove(loser)
        out.append(loser)
    return out + remaining


def ref_probe(p: Profile, seed: int) -> tuple[int, list[int]]:
    w = irv(p, seed=seed).winner
    eligible, failures = 0, []
    for i, ballot in enumerate(p.ballots):
        if w in ballot.missing or w not in ballot.ranked or ballot.groups[0] == (w,):
            continue
        eligible += 1
        changed = list(p.ballots)
        changed[i] = move_up(ballot, w)
        if irv(Profile(changed, p.candidates, p.weights), seed=seed).winner != w:
            failures.append(i)
    return eligible, failures


def random_profiles(n_profiles: int, seed: int, strict_share: float = 0.3):
    rng = np.random.default_rng(seed)
    for _ in range(n_profiles):
        m = int(rng.integers(3, 7))
        n = int(rng.integers(1, 16))
        if rng.random() < strict_share:  # strict complete ballots: where cycles live
            candidates = [chr(ord("A") + i) for i in range(m)]
            ballots = [
                WeakOrder.strict([candidates[i] for i in rng.permutation(m)]) for _ in range(n)
            ]
        else:
            ballots, candidates = random_weak_profile(rng, n, m)
        if all(not b.groups for b in ballots):
            continue
        weights = rng.integers(1, 4, size=n) if rng.random() < 0.5 else None
        yield Profile(ballots, candidates, weights)


def test_majority_relation_matches_reference():
    seen_cycle = seen_no_cycle = seen_small_smith = 0
    for p in random_profiles(400, seed=1):
        c = p.candidates
        n = pairwise_matrix(p)
        ref_n = ref_matrix(p)
        assert all(n[c.index(a), c.index(b)] == v for (a, b), v in ref_n.items())

        beats = ref_beats(p)
        ref_cw = [a for a in c if all((a, b) in beats for b in c if b != a)]
        ref_cl = [a for a in c if all((b, a) in beats for b in c if b != a)]
        assert condorcet_winner(p) == (ref_cw[0] if ref_cw else None)
        assert condorcet_loser(p) == (ref_cl[0] if ref_cl else None)

        assert smith_set(p) == ref_smith(p)
        assert has_majority_cycle(p) == ref_has_cycle(p)
        ref_3 = {
            frozenset(t)
            for t in itertools.permutations(c, 3)
            if (t[0], t[1]) in beats and (t[1], t[2]) in beats and (t[2], t[0]) in beats
        }
        got_3 = three_cycles(p)
        assert {frozenset(t) for t in got_3} == ref_3 and len(got_3) == len(ref_3)
        for a, b, d in got_3:
            assert (a, b) in beats and (b, d) in beats and (d, a) in beats

        seen_cycle += has_majority_cycle(p)
        seen_no_cycle += not has_majority_cycle(p)
        seen_small_smith += 1 < len(smith_set(p)) < p.m
    # the sample must actually exercise each branch
    assert seen_cycle > 20 and seen_no_cycle > 20 and seen_small_smith > 5


def test_scoring_rules_match_reference():
    for p in random_profiles(300, seed=2):
        assert plurality(p).scores == pytest.approx(
            {k: float(v) for k, v in ref_plurality(p).items()}
        )
        assert borda(p).scores == pytest.approx({k: float(v) for k, v in ref_borda(p).items()})

        n, beats = ref_matrix(p), ref_beats(p)
        for a in p.candidates:
            rest = [b for b in p.candidates if b != a]
            wins = sum((a, b) in beats for b in rest)
            ties = sum((a, b) not in beats and (b, a) not in beats for b in rest)
            assert copeland(p).scores[a] == wins + 0.5 * ties
            assert minimax(p).scores[a] == float(min(n[(a, b)] - n[(b, a)] for b in rest))


def test_kemeny_matches_reference():
    for p in random_profiles(120, seed=3):
        k = kemeny(p)
        n = ref_matrix(p)
        best = ref_kemeny_score(p)
        assert k.details["score"] == float(best)
        for order in k.details["optimal_rankings"]:
            assert sum(n[(a, b)] for a, b in itertools.combinations(order, 2)) == best
        assert k.ranking.ranked == k.details["optimal_rankings"][0]
        assert set(k.winners) == {o[0] for o in k.details["optimal_rankings"]}
        cw = condorcet_winner(p)
        if cw is not None:  # Kemeny is Condorcet-consistent
            assert k.winners == (cw,)


def test_irv_matches_reference_including_tie_breaks():
    ties = 0
    for i, p in enumerate(random_profiles(300, seed=4)):
        r = irv(p, seed=i)
        assert list(reversed(r.ranking.ranked)) == ref_irv(p, seed=i)
        ties += bool(r.details["tie_breaks"])
    assert ties > 20  # the tie-break path is exercised, not just the easy case


def test_monotonicity_probe_matches_reference():
    total_failures = 0
    for i, p in enumerate(random_profiles(250, seed=6)):
        probe = irv_monotonicity_probe(p, seed=i)
        eligible, failures = ref_probe(p, seed=i)
        assert probe.winner == irv(p, seed=i).winner
        assert probe.n_eligible == eligible
        assert list(probe.failures) == failures
        total_failures += len(failures)
    assert total_failures > 0  # at least some genuine failures were found and agreed on
