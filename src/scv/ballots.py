"""Weak-order ballots (SPEC §4).

A ballot is a weak order over a candidate set: an ordered sequence of tie groups, best
first, plus a separate set of candidates the voter did not rank (``missing``). Ties are
kept, never broken at random.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

#: Brute-force routines (linear extensions, Kemeny, axis enumeration) assume m <= 7.
MAX_CANDIDATES = 7


@dataclass(frozen=True)
class WeakOrder:
    """An ordered list of tie groups, best first, e.g. ``[["A"], ["B", "C"], ["D"]]``.

    Stored as tuples (members sorted within a group) and a frozenset so that equal
    ballots compare and hash equal. ``missing`` holds unranked candidates; a missing
    candidate is neither above nor below anything.
    """

    groups: tuple[tuple[str, ...], ...] = ()
    missing: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if isinstance(self.groups, str):
            raise TypeError("groups must be a sequence of tie groups, not a string")
        groups = []
        for g in self.groups:
            if isinstance(g, str):
                raise TypeError(
                    f"tie group {g!r} is a bare string; write [{g!r}] for a singleton group"
                )
            g = tuple(sorted(g))
            if not g:
                raise ValueError("empty tie group")
            groups.append(g)
        if isinstance(self.missing, str):
            raise TypeError("missing must be a set of candidate names, not a string")
        missing = frozenset(self.missing)
        ranked = [c for g in groups for c in g]
        for c in (*ranked, *missing):
            if not isinstance(c, str) or not c:
                raise TypeError(f"candidate names must be non-empty strings, got {c!r}")
        if len(set(ranked)) != len(ranked):
            dupes = sorted({c for c in ranked if ranked.count(c) > 1})
            raise ValueError(f"candidates ranked more than once: {dupes}")
        both = missing.intersection(ranked)
        if both:
            raise ValueError(f"candidates both ranked and missing: {sorted(both)}")
        object.__setattr__(self, "groups", tuple(groups))
        object.__setattr__(self, "missing", missing)

    # -- constructors -----------------------------------------------------------------

    @classmethod
    def strict(cls, ranking: Iterable[str], missing: Iterable[str] = ()) -> WeakOrder:
        """A strict ranking, best first: ``WeakOrder.strict(["A", "B", "C"])``."""
        if isinstance(ranking, str):
            raise TypeError("ranking must be a sequence of candidate names, not a string")
        return cls(tuple((c,) for c in ranking), frozenset(missing))

    @classmethod
    def parse(cls, text: str, missing: Iterable[str] = ()) -> WeakOrder:
        """Parse ``"A>B=C>D"``: ``>`` separates ranks, ``=`` separates tied candidates."""
        groups = [[c.strip() for c in part.split("=")] for part in text.split(">")]
        return cls(tuple(tuple(g) for g in groups), frozenset(missing))

    # -- views ------------------------------------------------------------------------

    @property
    def ranked(self) -> tuple[str, ...]:
        """Ranked candidates, best first (alphabetical within a tie group)."""
        return tuple(c for g in self.groups for c in g)

    @property
    def candidates(self) -> frozenset[str]:
        """Every candidate this ballot mentions: ranked plus missing."""
        return frozenset(self.ranked) | self.missing

    def rank_of(self) -> dict[str, int]:
        """Tie-group index per ranked candidate (0 = best). Missing candidates are absent."""
        return {c: i for i, g in enumerate(self.groups) for c in g}

    def __str__(self) -> str:
        body = " > ".join(" = ".join(g) for g in self.groups)
        if self.missing:
            body += f"  [missing: {', '.join(sorted(self.missing))}]"
        return body


def from_scores(scores: Mapping[str, float | None]) -> WeakOrder:
    """Build a ballot from scores: higher ranks higher, equal scores tie.

    ``None`` and NaN mean missing (NaN is what a failed thermometer parse stores).
    Infinite scores are rejected.
    """
    missing: set[str] = set()
    by_score: dict[float, list[str]] = {}
    for cand, s in scores.items():
        if s is None:
            missing.add(cand)
            continue
        s = float(s)
        if math.isnan(s):
            missing.add(cand)
        elif math.isinf(s):
            raise ValueError(f"infinite score for {cand!r}")
        else:
            by_score.setdefault(s, []).append(cand)
    groups = tuple(tuple(by_score[s]) for s in sorted(by_score, reverse=True))
    return WeakOrder(groups, frozenset(missing))


def pairwise(ballot: WeakOrder) -> dict[tuple[str, str], int]:
    """Pairwise relation over every ordered pair of the ballot's candidates.

    ``+1`` if a is strictly above b, ``-1`` if strictly below, ``0`` if tied or if
    either candidate is missing.
    """
    rank = ballot.rank_of()
    cands = sorted(ballot.candidates)
    out: dict[tuple[str, str], int] = {}
    for a in cands:
        for b in cands:
            if a == b:
                continue
            ra, rb = rank.get(a), rank.get(b)
            if ra is None or rb is None or ra == rb:
                out[(a, b)] = 0
            else:
                out[(a, b)] = 1 if ra < rb else -1
    return out


def is_strict(ballot: WeakOrder) -> bool:
    """True if no two ranked candidates are tied. Says nothing about missing candidates."""
    return all(len(g) == 1 for g in ballot.groups)


def is_complete(ballot: WeakOrder) -> bool:
    """True if the ballot has no missing candidates."""
    return not ballot.missing


def linear_extensions(ballot: WeakOrder) -> Iterator[WeakOrder]:
    """Yield every strict order consistent with the ballot (brute force, m <= 7).

    Only ties are resolved: missing candidates stay missing. Order is deterministic.
    """
    if len(ballot.ranked) > MAX_CANDIDATES:
        raise ValueError(
            f"linear_extensions is brute force and limited to {MAX_CANDIDATES} ranked candidates"
        )
    per_group = (itertools.permutations(g) for g in ballot.groups)
    for combo in itertools.product(*per_group):
        yield WeakOrder(tuple((c,) for g in combo for c in g), ballot.missing)


def move_up(ballot: WeakOrder, candidate: str) -> WeakOrder:
    """Raise ``candidate`` by one position, leaving everything else in place.

    - Tied with others: the tie is broken in its favour (it gets its own group directly
      above its former tie-mates).
    - Alone in its group: it swaps places with the whole group directly above. For a
      strict ballot this is the usual adjacent swap.
    - Already the sole top: returned unchanged.
    """
    rank = ballot.rank_of()
    if candidate not in rank:
        raise ValueError(f"{candidate!r} is not ranked on this ballot")
    i = rank[candidate]
    groups = [list(g) for g in ballot.groups]
    if len(groups[i]) > 1:
        groups[i].remove(candidate)
        groups.insert(i, [candidate])
    elif i > 0:
        groups[i - 1], groups[i] = groups[i], groups[i - 1]
    else:
        return ballot
    return WeakOrder(tuple(tuple(g) for g in groups), ballot.missing)
