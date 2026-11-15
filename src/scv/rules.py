"""Social choice engine (SPEC §5). Pure Python/numpy; no LLM dependency.

Conventions used throughout (each is also recorded in DECISIONS.md):

- A pairwise comparison in which either candidate is missing, or the two are tied,
  contributes 0 to both sides of the majority matrix.
- Two weighted totals are "equal" when they differ by at most ``profile.tol``
  (1e-9 x total weight), so float noise in survey weights never invents a majority.
- Rules never break ties at random. A rule reports every co-winner in ``winners``;
  ``winner`` is the unique winner or ``None``. IRV is the one exception: it needs an
  elimination order, so it breaks ties by a seeded candidate order and logs each one.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache, cached_property
from typing import Any

import numpy as np

from scv.ballots import MAX_CANDIDATES, WeakOrder, move_up

logger = logging.getLogger(__name__)

#: Relative tolerance for comparing weighted totals.
RTOL = 1e-9


# --------------------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------------------


class Profile:
    """A list of ballots, a weight per ballot, and the candidate list.

    A candidate in ``candidates`` that a ballot neither ranks nor lists as missing is
    treated as missing on that ballot. A ballot mentioning a candidate outside
    ``candidates`` is an error.

    For large or simulated electorates use :meth:`from_ranks` or :meth:`from_scores`,
    which take an (n, m) array and build ``WeakOrder`` objects only if ``ballots`` is read.
    """

    def __init__(
        self,
        ballots: Sequence[WeakOrder],
        candidates: Sequence[str],
        weights: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        self._set_candidates(candidates)
        ballots = tuple(ballots)
        if not ballots:
            raise ValueError("a profile needs at least one ballot")
        for i, b in enumerate(ballots):
            unknown = b.candidates - self.index.keys()
            if unknown:
                raise ValueError(f"ballot {i} mentions unknown candidates: {sorted(unknown)}")
        self._ballots: tuple[WeakOrder, ...] | None = ballots
        self._ranks: np.ndarray | None = None
        self._set_weights(weights, len(ballots))

    @classmethod
    def from_ranks(
        cls,
        ranks: np.ndarray,
        candidates: Sequence[str],
        weights: Sequence[float] | np.ndarray | None = None,
    ) -> Profile:
        """Build from an (n, m) array of rank values, columns in ``candidates`` order.

        Lower is better, equal values tie, NaN is missing. The values need not be dense
        (competition ranks or negated scores are fine); only their order within a row
        matters.
        """
        self = cls.__new__(cls)
        self._set_candidates(candidates)
        r = np.asarray(ranks, dtype=float)
        if r.ndim != 2 or r.shape[1] != len(self.candidates):
            raise ValueError(f"expected an (n, {len(self.candidates)}) array, got shape {r.shape}")
        if r.shape[0] == 0:
            raise ValueError("a profile needs at least one ballot")
        if np.isinf(r).any():
            raise ValueError("rank values must be finite or NaN")
        dense = _dense_ranks(r)
        dense.setflags(write=False)
        self._ranks = dense
        self._ballots = None
        self._set_weights(weights, r.shape[0])
        return self

    @classmethod
    def from_scores(
        cls,
        scores: np.ndarray,
        candidates: Sequence[str],
        weights: Sequence[float] | np.ndarray | None = None,
    ) -> Profile:
        """Build from an (n, m) score array: higher is better, equal ties, NaN is missing."""
        return cls.from_ranks(-np.asarray(scores, dtype=float), candidates, weights)

    def _set_candidates(self, candidates: Sequence[str]) -> None:
        if isinstance(candidates, str):
            raise TypeError("candidates must be a sequence of names, not a string")
        self.candidates: tuple[str, ...] = tuple(candidates)
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("duplicate candidate names")
        if len(self.candidates) < 2:
            raise ValueError("a profile needs at least 2 candidates")
        self.index: dict[str, int] = {c: i for i, c in enumerate(self.candidates)}

    def _set_weights(self, weights: Sequence[float] | np.ndarray | None, n: int) -> None:
        if weights is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(weights, dtype=float).copy()
            if w.shape != (n,):
                raise ValueError(f"expected {n} weights, got array of shape {w.shape}")
            if not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError("weights must be finite and non-negative")
        if w.sum() <= 0:
            raise ValueError("total weight must be positive")
        w.setflags(write=False)
        self.weights: np.ndarray = w

    @property
    def n(self) -> int:
        return len(self.weights)

    @property
    def m(self) -> int:
        return len(self.candidates)

    @property
    def total_weight(self) -> float:
        return float(self.weights.sum())

    @property
    def tol(self) -> float:
        """Absolute tolerance under which two weighted totals count as equal."""
        return RTOL * max(1.0, self.total_weight)

    @property
    def ballots(self) -> tuple[WeakOrder, ...]:
        """The ballots as ``WeakOrder`` objects (built on first use for array profiles)."""
        if self._ballots is None:
            self._ballots = tuple(_ballot_from_row(row, self.candidates) for row in self.ranks)
        return self._ballots

    @property
    def ranks(self) -> np.ndarray:
        """(n, m) tie-group index per ballot and candidate (0 = best); NaN = missing."""
        if self._ranks is None:
            r = np.full((self.n, self.m), np.nan)
            for i, b in enumerate(self.ballots):
                r[i] = _rank_row(b, self.index)
            r.setflags(write=False)
            self._ranks = r
        return self._ranks

    @cached_property
    def pairwise(self) -> np.ndarray:
        """Weighted pairwise majority matrix; see :func:`pairwise_matrix`."""
        r = self.ranks
        prefers = r[:, :, None] < r[:, None, :]  # NaN compares False: missing counts for nobody
        n_mat = np.tensordot(self.weights, prefers.astype(float), axes=1)
        n_mat.setflags(write=False)
        return n_mat

    @cached_property
    def majority(self) -> np.ndarray:
        """(m, m) int matrix: +1 if row strictly beats column, -1 if it loses, 0 if tied."""
        d = self.pairwise - self.pairwise.T
        s = np.where(d > self.tol, 1, np.where(d < -self.tol, -1, 0))
        s.setflags(write=False)
        return s


def _dense_ranks(x: np.ndarray) -> np.ndarray:
    """Row-wise dense ranks (0 = lowest value) of an (n, m) array; NaN stays NaN."""
    order = np.argsort(x, axis=1, kind="stable")  # NaN sorts last
    xs = np.take_along_axis(x, order, axis=1)
    is_new = np.ones(xs.shape, dtype=bool)
    is_new[:, 1:] = xs[:, 1:] != xs[:, :-1]
    dense_sorted = np.cumsum(is_new, axis=1) - 1.0
    dense_sorted[np.isnan(xs)] = np.nan
    out = np.empty_like(x)
    np.put_along_axis(out, order, dense_sorted, axis=1)
    return out


def _ballot_from_row(row: np.ndarray, candidates: Sequence[str]) -> WeakOrder:
    groups: dict[float, list[str]] = {}
    missing = []
    for c, r in zip(candidates, row.tolist(), strict=True):
        if r != r:  # NaN
            missing.append(c)
        else:
            groups.setdefault(r, []).append(c)
    return WeakOrder(tuple(tuple(groups[r]) for r in sorted(groups)), frozenset(missing))


def _rank_row(ballot: WeakOrder, index: dict[str, int]) -> np.ndarray:
    row = np.full(len(index), np.nan)
    for c, r in ballot.rank_of().items():
        row[index[c]] = r
    return row


# --------------------------------------------------------------------------------------
# Majority relation: matrix, Condorcet winner/loser, cycles, Smith set
# --------------------------------------------------------------------------------------


def pairwise_matrix(profile: Profile) -> np.ndarray:
    """``N[i][j]`` = total weight of voters strictly preferring candidate i to j.

    Indexed in ``profile.candidates`` order. Ties and missing contribute 0 to both sides.
    """
    return profile.pairwise


def condorcet_winner(profile: Profile) -> str | None:
    """The candidate who strictly beats every other candidate, or ``None``."""
    return _dominant(profile, sign=1)


def condorcet_loser(profile: Profile) -> str | None:
    """The candidate who strictly loses to every other candidate, or ``None``."""
    return _dominant(profile, sign=-1)


def _dominant(profile: Profile, sign: int) -> str | None:
    s = profile.majority
    for i, c in enumerate(profile.candidates):
        if np.all(np.delete(s[i], i) == sign):
            return c
    return None


def _closure(adj: np.ndarray) -> np.ndarray:
    """Transitive closure of a boolean adjacency matrix (Warshall; m is tiny)."""
    reach = adj.copy()
    for k in range(len(reach)):
        reach |= reach[:, [k]] & reach[[k], :]
    return reach


def has_majority_cycle(profile: Profile) -> bool:
    """True if the strict-majority relation contains a directed cycle of any length.

    With pairwise ties the relation is not a full tournament, so a cycle longer than 3
    can exist without any 3-cycle; this checks all lengths.
    """
    beats = profile.majority == 1
    return bool(np.any(np.diag(_closure(beats))))


def smith_set(profile: Profile) -> frozenset[str]:
    """Smallest non-empty set whose members each strictly beat every non-member."""
    s = profile.majority
    not_beaten = s >= 0
    np.fill_diagonal(not_beaten, False)
    reach = _closure(not_beaten)
    np.fill_diagonal(reach, True)
    return frozenset(c for i, c in enumerate(profile.candidates) if reach[i].all())


def three_cycles(profile: Profile) -> list[tuple[str, str, str]]:
    """All majority 3-cycles ``(a, b, c)`` with a beats b, b beats c, c beats a.

    Each cycle appears once, rotated to start at its earliest candidate in
    ``profile.candidates`` order.
    """
    beats = profile.majority == 1
    c = profile.candidates
    out = []
    for i, j, k in itertools.combinations(range(profile.m), 3):
        if beats[i, j] and beats[j, k] and beats[k, i]:
            out.append((c[i], c[j], c[k]))
        elif beats[i, k] and beats[k, j] and beats[j, i]:
            out.append((c[i], c[k], c[j]))
    return out


@dataclass(frozen=True)
class MajorityCycles:
    has_cycle: bool
    smith_set: frozenset[str]
    smith_size: int
    three_cycles: tuple[tuple[str, str, str], ...]


def majority_cycles(profile: Profile) -> MajorityCycles:
    """Cycle diagnostics of the strict-majority relation (SPEC §5.3)."""
    smith = smith_set(profile)
    return MajorityCycles(
        has_cycle=has_majority_cycle(profile),
        smith_set=smith,
        smith_size=len(smith),
        three_cycles=tuple(three_cycles(profile)),
    )


def condorcet_order(profile: Profile) -> WeakOrder | None:
    """The majority relation as a social ranking, if it is transitive; else ``None``.

    Uses the weak relation "a beats or ties b". When that is transitive it is a weak
    order (strict whenever there are no pairwise ties) and is returned as one.
    """
    r = profile.majority >= 0
    two_step = (r.astype(int) @ r.astype(int)) > 0
    if np.any(two_step & ~r):
        return None
    return _ranking(profile.candidates, _copeland_scores(profile), profile.tol)


# --------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleResult:
    """Outcome of one voting rule.

    ``winners`` holds every co-winner in candidate order (never empty). ``scores`` are
    oriented so that higher is always better. ``details`` is rule-specific.
    """

    rule: str
    winners: tuple[str, ...]
    ranking: WeakOrder
    scores: dict[str, float] | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def winner(self) -> str | None:
        """The unique winner, or ``None`` when the rule leaves a tie at the top."""
        return self.winners[0] if len(self.winners) == 1 else None


def _ranking(candidates: Sequence[str], scores: np.ndarray, tol: float) -> WeakOrder:
    """Weak order from scores, higher first; scores within ``tol`` of a neighbour tie."""
    order = np.argsort(-scores, kind="stable")
    groups: list[list[str]] = []
    prev = None
    for i in order:
        if prev is not None and prev - scores[i] <= tol:
            groups[-1].append(candidates[i])
        else:
            groups.append([candidates[i]])
        prev = scores[i]
    return WeakOrder(tuple(tuple(g) for g in groups))


def _score_result(rule: str, profile: Profile, scores: np.ndarray, **details: Any) -> RuleResult:
    ranking = _ranking(profile.candidates, scores, profile.tol)
    top = set(ranking.groups[0])
    return RuleResult(
        rule=rule,
        winners=tuple(c for c in profile.candidates if c in top),
        ranking=ranking,
        scores={c: float(s) for c, s in zip(profile.candidates, scores, strict=True)},
        details=details,
    )


def _top_tallies(ranks_inf: np.ndarray, weights: np.ndarray, remaining: np.ndarray) -> np.ndarray:
    """Weight each remaining candidate receives when every ballot counts for its highest-
    ranked remaining candidate, split equally among candidates tied at that top.

    ``ranks_inf`` is the rank matrix with missing as +inf. Ballots with no remaining
    ranked candidate are exhausted and count for nobody.
    """
    r = np.where(remaining[None, :], ranks_inf, np.inf)
    best = r.min(axis=1)
    top = (r == best[:, None]) & np.isfinite(best)[:, None]
    k = top.sum(axis=1)
    share = np.divide(weights, k, out=np.zeros_like(weights, dtype=float), where=k > 0)
    return (top * share[:, None]).sum(axis=0)


def _ranks_inf(ranks: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(ranks), np.inf, ranks)


def plurality(profile: Profile) -> RuleResult:
    """Plurality. A ballot with k candidates tied at the top gives each 1/k of its weight."""
    scores = _top_tallies(_ranks_inf(profile.ranks), profile.weights, np.ones(profile.m, bool))
    return _score_result("plurality", profile, scores)


def borda(profile: Profile) -> RuleResult:
    """Weak-order Borda: per ballot, candidates strictly beaten + 0.5 x candidates tied.

    A comparison involving a missing candidate counts as a tie (0.5 to each side), which
    keeps Borda consistent with the majority matrix: the score equals
    ``0.5 * (W * (m - 1) + sum_b (N[a][b] - N[b][a]))``.
    """
    n_mat = profile.pairwise
    scores = 0.5 * (profile.total_weight * (profile.m - 1) + (n_mat - n_mat.T).sum(axis=1))
    return _score_result("borda", profile, scores)


def _copeland_scores(profile: Profile) -> np.ndarray:
    s = profile.majority
    return (s == 1).sum(axis=1) + 0.5 * ((s == 0).sum(axis=1) - 1)  # -1 drops the diagonal


def copeland(profile: Profile) -> RuleResult:
    """Copeland: 1 per pairwise majority win, 0.5 per pairwise majority tie."""
    return _score_result("copeland", profile, _copeland_scores(profile))


def minimax(profile: Profile) -> RuleResult:
    """Minimax (Simpson-Kramer) by margins.

    A candidate's score is its worst pairwise margin, ``min_b (N[a][b] - N[b][a])``; the
    highest score wins. Equivalent to minimising the largest defeat margin.
    """
    d = np.array(profile.pairwise - profile.pairwise.T)
    np.fill_diagonal(d, np.inf)
    return _score_result("minimax", profile, d.min(axis=1))


@cache
def _permutations(m: int) -> np.ndarray:
    return np.array(list(itertools.permutations(range(m))), dtype=np.intp)


def kemeny(profile: Profile) -> RuleResult:
    """Kemeny-Young by brute force over all m! strict rankings (m <= 7).

    Maximises the total weight of pairwise preferences the ranking agrees with. When
    several rankings are optimal, ``ranking`` is the first in lexicographic order of
    ``profile.candidates``, ``details["optimal_rankings"]`` lists all of them, and
    ``winners`` holds every candidate that tops some optimal ranking.
    """
    m = profile.m
    if m > MAX_CANDIDATES:
        raise ValueError(f"Kemeny is brute force and limited to {MAX_CANDIDATES} candidates")
    perms = _permutations(m)
    hi, lo = np.triu_indices(m, 1)  # positions: hi is ranked above lo
    scores = profile.pairwise[perms[:, hi], perms[:, lo]].sum(axis=1)
    best = scores.max()
    optimal = perms[scores >= best - profile.tol]
    c = profile.candidates
    tops = set(optimal[:, 0].tolist())
    return RuleResult(
        rule="kemeny",
        winners=tuple(c[i] for i in range(m) if i in tops),
        ranking=WeakOrder.strict([c[i] for i in optimal[0]]),
        scores=None,
        details={
            "score": float(best),
            "unique": len(optimal) == 1,
            "optimal_rankings": [tuple(c[i] for i in p) for p in optimal],
        },
    )


def _tiebreak_priority(m: int, seed: int) -> np.ndarray:
    """``priority[i]`` = position of candidate i in the seeded order (lower goes first)."""
    order = np.random.default_rng(seed).permutation(m)
    priority = np.empty(m, dtype=int)
    priority[order] = np.arange(m)
    return priority


def _irv_run(
    tally_fn: Callable[[np.ndarray], np.ndarray], m: int, priority: np.ndarray, tol: float
) -> tuple[list[int], list[np.ndarray], list[tuple[int, list[int], int]]]:
    """Core IRV loop. ``tally_fn(remaining)`` gives each candidate's tally for a boolean
    mask of the candidates still standing.

    Returns the elimination order (the last entry is the winner), the tally vector of
    every round, and the tie-breaks as ``(round, tied candidate indices, eliminated)``.
    """
    remaining = np.ones(m, dtype=bool)
    eliminated: list[int] = []
    rounds: list[np.ndarray] = []
    ties: list[tuple[int, list[int], int]] = []
    while remaining.sum() > 1:
        tally = tally_fn(remaining)
        rounds.append(tally)
        low = tally[remaining].min()
        tied = [i for i in range(m) if remaining[i] and tally[i] <= low + tol]
        out = min(tied, key=lambda i: priority[i])
        if len(tied) > 1:
            ties.append((len(rounds), tied, out))
        remaining[out] = False
        eliminated.append(out)
    eliminated.append(int(np.flatnonzero(remaining)[0]))
    return eliminated, rounds, ties


def irv(profile: Profile, seed: int = 0) -> RuleResult:
    """Instant-runoff voting.

    Each ballot counts for its highest-ranked remaining candidate; candidates tied at
    that top split the ballot's weight equally. The lowest tally is eliminated each
    round, one candidate at a time, until one remains. Ties for lowest are broken by a
    seeded order: ``numpy.random.default_rng(seed).permutation(m)`` orders the
    candidates, and the tied candidate earliest in that order is eliminated. Every
    tie-break is logged and recorded in ``details["tie_breaks"]``.

    The social ranking is the reverse elimination order (strict).
    """
    c = profile.candidates
    priority = _tiebreak_priority(profile.m, seed)
    ranks_inf = _ranks_inf(profile.ranks)
    order, rounds, ties = _irv_run(
        lambda remaining: _top_tallies(ranks_inf, profile.weights, remaining),
        profile.m,
        priority,
        profile.tol,
    )
    tie_breaks = []
    for rnd, tied, out in ties:
        tb = {"round": rnd, "tied": tuple(c[i] for i in tied), "eliminated": c[out]}
        tie_breaks.append(tb)
        logger.info(
            "IRV tie for elimination in round %d among %s; eliminated %s (seed=%d)",
            rnd,
            list(tb["tied"]),
            tb["eliminated"],
            seed,
        )
    round_log = [
        {
            "round": r + 1,
            "tallies": {c[i]: float(t[i]) for i in range(profile.m) if i not in order[:r]},
            "eliminated": c[order[r]],
        }
        for r, t in enumerate(rounds)
    ]
    winner = c[order[-1]]
    return RuleResult(
        rule="irv",
        winners=(winner,),
        ranking=WeakOrder.strict([c[i] for i in reversed(order)]),
        scores=None,
        details={
            "seed": seed,
            "rounds": round_log,
            "tie_breaks": tie_breaks,
            "winner_by_tiebreak": any(tb["round"] == len(rounds) for tb in tie_breaks),
        },
    )


#: Rule name -> function. IRV additionally takes ``seed``.
RULES: dict[str, Callable[..., RuleResult]] = {
    "plurality": plurality,
    "borda": borda,
    "copeland": copeland,
    "minimax": minimax,
    "kemeny": kemeny,
    "irv": irv,
}


def run_all_rules(profile: Profile, seed: int = 0) -> dict[str, RuleResult]:
    """Run every rule in :data:`RULES`. ``seed`` only affects IRV tie-breaking."""
    return {
        name: (fn(profile, seed=seed) if name == "irv" else fn(profile))
        for name, fn in RULES.items()
    }


# --------------------------------------------------------------------------------------
# Rule disagreement
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleDisagreement:
    """``matrix[i][j]`` is True when rules i and j pick different winner sets."""

    rules: tuple[str, ...]
    matrix: np.ndarray

    def differs(self, rule_a: str, rule_b: str) -> bool:
        return bool(self.matrix[self.rules.index(rule_a), self.rules.index(rule_b)])

    @property
    def share(self) -> float:
        """Share of unordered rule pairs that disagree."""
        iu = np.triu_indices(len(self.rules), 1)
        return float(self.matrix[iu].mean()) if len(iu[0]) else 0.0


def rule_disagreement(results: dict[str, RuleResult]) -> RuleDisagreement:
    """Pairwise rule disagreement. Two rules disagree when their winner *sets* differ,
    so a rule that leaves a tie disagrees with one that picks a single co-winner."""
    names = tuple(results)
    sets = [frozenset(results[r].winners) for r in names]
    mat = np.array([[a != b for b in sets] for a in sets], dtype=bool)
    return RuleDisagreement(rules=names, matrix=mat)


# --------------------------------------------------------------------------------------
# IRV monotonicity probe
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MonotonicityProbe:
    """Result of :func:`irv_monotonicity_probe`.

    ``failure_fraction`` is failures / eligible ballots (0.0 when none are eligible);
    ``failure_fraction_all`` is failures / all ballots. Both count ballots (rows), not
    weight. ``failures`` holds the indices of the failing ballots.
    """

    winner: str
    n_ballots: int
    n_eligible: int
    n_failures: int
    failure_fraction: float
    failure_fraction_all: float
    failures: tuple[int, ...]


def irv_monotonicity_probe(profile: Profile, seed: int = 0) -> MonotonicityProbe:
    """Single-ballot IRV monotonicity probe (SPEC §5.6).

    For each ballot on which the IRV winner w is ranked but is not the sole top, raise w
    by one position (:func:`scv.ballots.move_up`), rerun IRV with the same seed, and
    record a failure if the winner is no longer w. Ballots on which w is missing are not
    eligible. The perturbation moves one ballot row with its full weight.

    Each rerun is a full IRV count, computed incrementally: the tallies for a given set
    of remaining candidates are the cached whole-electorate tallies, minus the ballot's
    old contribution, plus its new one.
    """
    c, m, tol = profile.candidates, profile.m, profile.tol
    priority = _tiebreak_priority(m, seed)
    base = _ranks_inf(profile.ranks)
    cache: dict[bytes, np.ndarray] = {}

    def total(remaining: np.ndarray) -> np.ndarray:
        key = remaining.tobytes()
        if key not in cache:
            cache[key] = _top_tallies(base, profile.weights, remaining)
        return cache[key]

    w = _irv_run(total, m, priority, tol)[0][-1]

    failures: list[int] = []
    n_eligible = 0
    memo: dict[tuple[WeakOrder, float], bool] = {}
    for i, ballot in enumerate(profile.ballots):
        r = profile.ranks[i, w]
        if np.isnan(r) or (r == 0 and len(ballot.groups[0]) == 1):
            continue
        n_eligible += 1
        key = (ballot, float(profile.weights[i]))
        if key not in memo:
            old = base[[i]]
            new = _ranks_inf(_rank_row(move_up(ballot, c[w]), profile.index))[None, :]
            wi = profile.weights[[i]]

            def tally(remaining: np.ndarray, old=old, new=new, wi=wi) -> np.ndarray:
                return (
                    total(remaining)
                    - _top_tallies(old, wi, remaining)
                    + _top_tallies(new, wi, remaining)
                )

            memo[key] = _irv_run(tally, m, priority, tol)[0][-1] != w
        if memo[key]:
            failures.append(i)

    return MonotonicityProbe(
        winner=c[w],
        n_ballots=profile.n,
        n_eligible=n_eligible,
        n_failures=len(failures),
        failure_fraction=len(failures) / n_eligible if n_eligible else 0.0,
        failure_fraction_all=len(failures) / profile.n,
        failures=tuple(failures),
    )
