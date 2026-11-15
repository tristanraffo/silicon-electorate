"""Shared profile builders for tests."""

from __future__ import annotations

import numpy as np

from scv.ballots import WeakOrder, from_scores
from scv.nulls import spatial_1d
from scv.rules import Profile


def profile_from(spec: list[tuple[float, str]], candidates: str | list[str]) -> Profile:
    """``profile_from([(2, "A>B>C"), (3, "B>A=C")], "ABC")``: one weighted row per entry."""
    return Profile(
        [WeakOrder.parse(b) for _, b in spec],
        candidates=list(candidates),
        weights=[w for w, _ in spec],
    )


def expand(spec: list[tuple[int, str]], candidates: str | list[str]) -> Profile:
    """Same electorate as :func:`profile_from` but with one unit-weight ballot per voter."""
    ballots = [WeakOrder.parse(b) for w, b in spec for _ in range(w)]
    return Profile(ballots, candidates=list(candidates))


def random_weak_profile(
    rng: np.random.Generator, n: int, m: int, p_missing: float = 0.15
) -> tuple[list[WeakOrder], list[str]]:
    """Random weak-order ballots with ties (coarse scores) and missing candidates."""
    candidates = [chr(ord("A") + i) for i in range(m)]
    ballots = []
    for _ in range(n):
        scores = {
            c: (None if rng.random() < p_missing else float(rng.integers(0, 4))) for c in candidates
        }
        ballots.append(from_scores(scores))
    return ballots, candidates


def single_peaked_profile(rng: np.random.Generator, n: int, m: int) -> Profile:
    """1-D spatial electorate (strict ballots by distance on a line): single-peaked."""
    return spatial_1d(n, m, rng).profile
