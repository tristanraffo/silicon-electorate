"""Null and synthetic electorates (SPEC §7).

Every generator takes ``rng`` as a ``numpy.random.Generator`` or an integer seed, so a
result is fully determined by its arguments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from scv.rules import Profile


def as_rng(rng: np.random.Generator | int) -> np.random.Generator:
    """Accept a Generator or an integer seed. ``None`` is rejected: runs must be seeded."""
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)) and not isinstance(rng, bool):
        return np.random.default_rng(int(rng))
    raise TypeError(f"rng must be a numpy Generator or an int seed, got {type(rng).__name__}")


def default_candidates(m: int) -> tuple[str, ...]:
    """Placeholder names ``c1 .. cm`` for synthetic electorates."""
    return tuple(f"c{i + 1}" for i in range(m))


def _names(m: int, candidates: Sequence[str] | None) -> tuple[str, ...]:
    names = default_candidates(m) if candidates is None else tuple(candidates)
    if len(names) != m:
        raise ValueError(f"expected {m} candidate names, got {len(names)}")
    return names


def impartial_culture(
    n: int,
    m: int,
    rng: np.random.Generator | int,
    candidates: Sequence[str] | None = None,
) -> Profile:
    """Impartial culture: n independent strict orders, each uniform over the m! orders."""
    rng = as_rng(rng)
    ranks = rng.permuted(np.tile(np.arange(m), (n, 1)), axis=1)
    return Profile.from_ranks(ranks, _names(m, candidates))


@dataclass(frozen=True)
class SpatialElectorate:
    """A Euclidean spatial electorate and the latent positions that generated it.

    ``scores[i, j]`` is voter i's utility for candidate j: minus the distance between them.
    """

    profile: Profile
    scores: np.ndarray
    candidate_positions: np.ndarray  # (m, dim)
    voter_positions: np.ndarray  # (n, dim)

    @property
    def dim(self) -> int:
        return self.candidate_positions.shape[1]

    @property
    def axis(self) -> tuple[str, ...]:
        """1-D only: the candidates in left-to-right order (the true single-peaked axis)."""
        if self.dim != 1:
            raise ValueError("only a 1-D spatial electorate has a single axis")
        order = np.argsort(self.candidate_positions[:, 0], kind="stable")
        return tuple(self.profile.candidates[i] for i in order)


def spatial(
    n: int,
    m: int,
    rng: np.random.Generator | int,
    dim: int = 1,
    candidates: Sequence[str] | None = None,
    voter_sd: float = 1.0,
) -> SpatialElectorate:
    """Spatial model: candidates ~ N(0, I) and voters ~ N(0, voter_sd^2 I) in ``dim``
    dimensions; each voter ranks candidates by Euclidean distance, nearest first.

    With ``dim=1`` every ballot is single-peaked on the candidates' left-to-right order.
    """
    if dim < 1:
        raise ValueError("dim must be at least 1")
    rng = as_rng(rng)
    cand = rng.normal(size=(m, dim))
    voters = rng.normal(scale=voter_sd, size=(n, dim))
    dist = np.linalg.norm(voters[:, None, :] - cand[None, :, :], axis=2)
    scores = -dist
    return SpatialElectorate(
        profile=Profile.from_scores(scores, _names(m, candidates)),
        scores=scores,
        candidate_positions=cand,
        voter_positions=voters,
    )


def spatial_1d(n: int, m: int, rng: np.random.Generator | int, **kwargs) -> SpatialElectorate:
    """1-D spatial electorate: single-peaked by construction."""
    return spatial(n, m, rng, dim=1, **kwargs)


def spatial_2d(n: int, m: int, rng: np.random.Generator | int, **kwargs) -> SpatialElectorate:
    """2-D spatial electorate, for intuition and calibration curves."""
    return spatial(n, m, rng, dim=2, **kwargs)


def permutation_null(scores: np.ndarray, rng: np.random.Generator | int) -> np.ndarray:
    """Permutation null of a real score matrix (voters x candidates).

    Each candidate's column is permuted independently across voters. Every candidate
    keeps exactly its own score distribution (missing values included), but any
    correlation between candidates is destroyed. Returns a new array.
    """
    rng = as_rng(rng)
    s = np.asarray(scores, dtype=float)
    if s.ndim != 2:
        raise ValueError(f"expected a 2-D (voters x candidates) array, got shape {s.shape}")
    return rng.permuted(s, axis=0)
