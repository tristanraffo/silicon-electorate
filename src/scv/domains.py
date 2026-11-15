"""Domain diagnostics (SPEC §6): the core outcome variables.

Single-peakedness, Sen's value restriction, preference dimensionality and profile
diversity. Everything is computed on ``Profile.ranks`` so it stays fast at ANES scale.

Conventions (also in DECISIONS.md):

- Ballots with zero weight are ignored everywhere.
- A ballot is tested only on the candidates it ranks: a missing candidate can never
  make a ballot fail a domain condition.
- "Single-peaked" for a weak order means some linear extension is single-peaked
  (SPEC §6.2). "Single-plateaued" is the stricter check: ties are allowed only inside
  one contiguous top plateau, or between two candidates on opposite sides of it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np

from scv.ballots import MAX_CANDIDATES, WeakOrder, linear_extensions
from scv.rules import Profile

Mode = Literal["peaked", "plateaued"]
_MODES = ("peaked", "plateaued")

# --------------------------------------------------------------------------------------
# Axes
# --------------------------------------------------------------------------------------


@cache
def _axes_idx(m: int) -> np.ndarray:
    """All m!/2 axes as index arrays; an axis and its reversal are the same axis, and
    the orientation kept is the one whose first index is below its last."""
    if m > MAX_CANDIDATES:
        raise ValueError(f"axis enumeration is brute force and limited to {MAX_CANDIDATES}")
    axes = [p for p in itertools.permutations(range(m)) if p[0] < p[-1]]
    arr = np.array(axes, dtype=np.intp)
    arr.setflags(write=False)
    return arr


def all_axes(candidates: Sequence[str]) -> list[tuple[str, ...]]:
    """Every axis (ordering of the candidates up to reversal), in a fixed order."""
    c = tuple(candidates)
    return [tuple(c[i] for i in ax) for ax in _axes_idx(len(c))]


def canonical_axis(axis: Sequence[str], candidates: Sequence[str]) -> tuple[str, ...]:
    """The orientation of ``axis`` used by :func:`all_axes` (reverse it if needed)."""
    c = tuple(candidates)
    axis = tuple(axis)
    if sorted(axis) != sorted(c):
        raise ValueError("axis must be an ordering of exactly the candidates")
    return axis if c.index(axis[0]) < c.index(axis[-1]) else axis[::-1]


# --------------------------------------------------------------------------------------
# Single-peakedness: one ballot, by definition (slow, used as the reference in tests)
# --------------------------------------------------------------------------------------


def _ranks_along(ballot: WeakOrder, axis: Sequence[str]) -> list[int]:
    """The ballot's tie-group indices read along the axis, skipping unranked candidates."""
    rank = ballot.rank_of()
    unknown = set(rank) - set(axis)
    if unknown:
        raise ValueError(f"axis does not contain ranked candidates {sorted(unknown)}")
    return [rank[c] for c in axis if c in rank]


def _strictly_unimodal(seq: Sequence[int]) -> bool:
    """Strictly improving (rank number falling) up to a single peak, then strictly worsening."""
    if not seq:
        return True
    peak = seq.index(min(seq))
    left, right = seq[: peak + 1], seq[peak:]
    return all(a > b for a, b in itertools.pairwise(left)) and all(
        a < b for a, b in itertools.pairwise(right)
    )


def is_single_peaked(ballot: WeakOrder, axis: Sequence[str]) -> bool:
    """True if some linear extension of the ballot is single-peaked on ``axis``.

    For a strict ballot this is the usual definition: utility falls monotonically on
    both sides of the peak. The axis is restricted to the candidates the ballot ranks.
    """
    return any(_strictly_unimodal(_ranks_along(ext, axis)) for ext in linear_extensions(ballot))


def is_single_plateaued(ballot: WeakOrder, axis: Sequence[str]) -> bool:
    """True if the ballot is single-plateaued on ``axis``: strictly improving up to a
    contiguous plateau of top-ranked candidates, then strictly worsening.

    Stricter than :func:`is_single_peaked` for weak orders; identical for strict ones.
    """
    seq = _ranks_along(ballot, axis)
    if not seq:
        return True
    lo = min(seq)
    first = seq.index(lo)
    last = len(seq) - 1 - seq[::-1].index(lo)
    if any(r != lo for r in seq[first : last + 1]):
        return False
    return _strictly_unimodal(seq[: first + 1]) and _strictly_unimodal(seq[last:])


# --------------------------------------------------------------------------------------
# Single-peakedness: whole profile, vectorised
# --------------------------------------------------------------------------------------


def _unique_rows(profile: Profile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct ballots among positive-weight rows.

    Returns ``(rows, inverse, keep)``: ``rows`` is (k, m) with missing as +inf,
    ``keep`` is the boolean mask of positive-weight ballots, and ``inverse`` maps each
    kept ballot to its row.
    """
    keep = profile.weights > 0
    r = np.where(np.isnan(profile.ranks[keep]), np.inf, profile.ranks[keep])
    rows, inverse = np.unique(r, axis=0, return_inverse=True)
    return rows, np.asarray(inverse).reshape(-1), keep


def _consistency_unique(rows: np.ndarray, axes: np.ndarray, mode: Mode) -> np.ndarray:
    """(k, n_axes) bool: is distinct ballot k consistent with each axis?

    Reading a ballot's ranks x along an axis (missing = +inf), let pm[j] and sm[j] be the
    best rank strictly before and strictly after position j.

    - peaked: some linear extension is single-peaked iff no ranked candidate is a
      "valley", strictly worse than something on each side: not (x > pm and x > sm).
    - plateaued: every ranked candidate is either top-ranked, or strictly better than
      everything on one side: x == 0 or pm > x or sm > x.
    """
    k, m = rows.shape
    out = np.empty((k, len(axes)), dtype=bool)
    chunk = max(1, int(4_000_000 // max(1, k * m)))
    for start in range(0, len(axes), chunk):
        x = rows[:, axes[start : start + chunk]]  # (k, c, m)
        pad = np.full(x.shape[:2] + (1,), np.inf)
        pm = np.concatenate([pad, np.minimum.accumulate(x, axis=2)[:, :, :-1]], axis=2)
        sm = np.concatenate(
            [np.minimum.accumulate(x[:, :, ::-1], axis=2)[:, :, ::-1][:, :, 1:], pad], axis=2
        )
        ranked = np.isfinite(x)
        if mode == "peaked":
            bad = ranked & (x > pm) & (x > sm)
        else:
            bad = ranked & (x != 0) & ~(pm > x) & ~(sm > x)
        out[:, start : start + chunk] = ~bad.any(axis=2)
    return out


def _check_mode(mode: str) -> None:
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")


def sp_consistency(profile: Profile, mode: Mode = "peaked") -> np.ndarray:
    """(n, n_axes) boolean matrix: is ballot i consistent with axis j of :func:`all_axes`?

    Zero-weight ballots get an all-True row (they constrain nothing). Compute this once
    and reuse it for bootstraps and sub-electorates: the weight share on each axis is
    just ``weights @ matrix / weights.sum()``.
    """
    _check_mode(mode)
    rows, inverse, keep = _unique_rows(profile)
    out = np.ones((profile.n, len(_axes_idx(profile.m))), dtype=bool)
    out[keep] = _consistency_unique(rows, _axes_idx(profile.m), mode)[inverse]
    return out


def _axis_shares(profile: Profile, mode: Mode) -> np.ndarray:
    """Weight share of ballots consistent with each axis."""
    _check_mode(mode)
    rows, inverse, keep = _unique_rows(profile)
    w = np.bincount(inverse, weights=profile.weights[keep], minlength=len(rows))
    return (w @ _consistency_unique(rows, _axes_idx(profile.m), mode)) / profile.total_weight


_FULL = 1 - 1e-12  # a share at or above this is "every ballot"


def _summarise(profile: Profile, shares: np.ndarray):
    """(best axis, best share, deletion share, axes fitting every ballot) from axis shares."""
    c, axes = profile.candidates, _axes_idx(profile.m)
    best = float(shares.max())
    j = int(np.flatnonzero(shares >= best - 1e-12)[0])
    axis = tuple(c[i] for i in axes[j])
    full = [tuple(c[i] for i in axes[k]) for k in np.flatnonzero(shares >= _FULL)]
    return axis, min(best, 1.0), (0.0 if best >= _FULL else 1.0 - best), full


def sp_axes(profile: Profile, mode: Mode = "peaked") -> list[tuple[str, ...]]:
    """The axes on which **every** ballot is single-peaked (empty if there is none)."""
    return _summarise(profile, _axis_shares(profile, mode))[3]


def sp_max_consistent(profile: Profile, mode: Mode = "peaked") -> tuple[tuple[str, ...], float]:
    """The axis maximising the weight share of single-peaked ballots, and that share.

    If several axes tie, the first in :func:`all_axes` order is returned.
    """
    return _summarise(profile, _axis_shares(profile, mode))[:2]


def sp_deletion_share(profile: Profile, mode: Mode = "peaked") -> float:
    """**Primary metric.** ``1 - best_share``: the weight share of the electorate that
    must be removed for the rest to be single-peaked on a common axis."""
    return _summarise(profile, _axis_shares(profile, mode))[2]


@dataclass(frozen=True)
class SinglePeakedness:
    """Both single-peakedness checks for one profile (SPEC §6.2: "report both")."""

    axis: tuple[str, ...]
    share: float
    deletion_share: float
    n_axes_all_ballots: int
    plateau_axis: tuple[str, ...]
    plateau_share: float
    plateau_deletion_share: float
    plateau_n_axes_all_ballots: int


def single_peakedness(profile: Profile) -> SinglePeakedness:
    """Best axis, share and deletion share under both the peaked and plateaued checks."""
    fields: list = []
    for mode in _MODES:
        axis, share, deletion, full = _summarise(profile, _axis_shares(profile, mode))
        fields += [axis, share, deletion, len(full)]
    return SinglePeakedness(*fields)


# --------------------------------------------------------------------------------------
# Sen's value restriction
# --------------------------------------------------------------------------------------

_POSITIONS = ("best", "middle", "worst")


@dataclass(frozen=True)
class ValueRestriction:
    """Sen's value restriction, triple by triple.

    ``satisfied[t]``: in triple t some candidate is never best, never middle or never
    worst among the ballots. ``min_violation[t]``: the smallest weight share that has to
    be removed for the triple to satisfy it (0.0 when it already does). ``share`` is the
    share of triples satisfied; ``mean_min_violation`` averages ``min_violation``.
    """

    triples: tuple[tuple[str, str, str], ...]
    satisfied: tuple[bool, ...]
    min_violation: tuple[float, ...]
    share: float
    mean_min_violation: float


def value_restriction(profile: Profile) -> ValueRestriction:
    """Sen's value restriction for every triple of candidates (SPEC §6.3).

    Positions use strict relations within the triple: a candidate is *best* on a ballot
    if it is strictly above both others, *worst* if strictly below both, *middle* if
    strictly between them. A ballot that ties or omits members of the triple only
    contributes the positions it strictly determines (e.g. ``A > B = C`` puts A best and
    places nobody in the middle or last).
    """
    r, w, c = profile.ranks, profile.weights, profile.candidates
    total, tol = profile.total_weight, profile.tol
    triples, satisfied, min_violation = [], [], []
    for trio in itertools.combinations(range(profile.m), 3):
        weight_at = []  # weight with candidate x in position p, for the 9 (x, p) cells
        for x in trio:
            y, z = (i for i in trio if i != x)
            rx, ry, rz = r[:, x], r[:, y], r[:, z]
            best = (rx < ry) & (rx < rz)  # NaN compares False, so missing places nobody
            worst = (rx > ry) & (rx > rz)
            middle = ((ry < rx) & (rx < rz)) | ((rz < rx) & (rx < ry))
            weight_at.extend(float(w[mask].sum()) for mask in (best, middle, worst))
        lowest = min(weight_at)
        triples.append(tuple(c[i] for i in trio))
        satisfied.append(lowest <= tol)
        min_violation.append(0.0 if lowest <= tol else lowest / total)
    return ValueRestriction(
        triples=tuple(triples),
        satisfied=tuple(satisfied),
        min_violation=tuple(min_violation),
        share=float(np.mean(satisfied)) if satisfied else math.nan,
        mean_min_violation=float(np.mean(min_violation)) if min_violation else math.nan,
    )


# --------------------------------------------------------------------------------------
# Dimensionality
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dimensionality:
    """PCA of the voter-centred score matrix.

    ``explained`` are the explained-variance shares of all components, largest first.
    ``n_dropped`` voters were excluded for having a missing score or zero weight.
    Shares are NaN when fewer than 2 voters remain or the centred scores have no variance.
    """

    pc1_share: float
    pc2_share: float
    explained: tuple[float, ...]
    n_used: int
    n_dropped: int


def dimensionality(
    scores: np.ndarray, weights: Sequence[float] | np.ndarray | None = None
) -> Dimensionality:
    """Explained-variance shares of PC1 and PC2 (SPEC §6.4). PC1 share is the
    monoculture metric: 1.0 means one dimension accounts for all disagreement.

    ``scores`` is voters x candidates. Each voter's scores are centred on that voter's
    own mean (removing "rates everyone high/low"), then an ordinary PCA is run (columns
    centred, covariance eigenvalues). With ``weights`` the column means and covariance
    are weighted. Voters with any missing score are dropped (complete cases).
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 2:
        raise ValueError(f"expected a 2-D (voters x candidates) array, got shape {s.shape}")
    n, m = s.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"expected {n} weights, got array of shape {w.shape}")
    keep = ~np.isnan(s).any(axis=1) & (w > 0)
    s, w = s[keep], w[keep]
    n_used = int(keep.sum())
    nan = Dimensionality(math.nan, math.nan, (math.nan,) * m, n_used, n - n_used)
    if n_used < 2:
        return nan
    x = s - s.mean(axis=1, keepdims=True)
    x = x - np.average(x, axis=0, weights=w)
    cov = (x * w[:, None]).T @ x / w.sum()
    eig = np.clip(np.linalg.eigvalsh(cov)[::-1], 0.0, None)
    if eig.sum() <= 1e-12 * max(1.0, float(np.abs(s).max()) ** 2):
        return nan
    explained = eig / eig.sum()
    return Dimensionality(
        pc1_share=float(explained[0]),
        pc2_share=float(explained[1]) if m > 1 else 0.0,
        explained=tuple(float(e) for e in explained),
        n_used=n_used,
        n_dropped=n - n_used,
    )


# --------------------------------------------------------------------------------------
# Profile diversity
# --------------------------------------------------------------------------------------


def kendall_tau_distance(a: WeakOrder, b: WeakOrder, candidates: Sequence[str]) -> float:
    """Normalised Kendall-tau distance between two ballots, in [0, 1].

    Over all candidate pairs: 1 if the ballots order the pair in opposite ways, 0.5 if
    exactly one of them leaves the pair undecided (tied, or a candidate missing), else 0;
    divided by the number of pairs. For two complete strict rankings this is the usual
    share of discordant pairs.
    """
    ra, rb = a.rank_of(), b.rank_of()

    def sign(rank: dict[str, int], x: str, y: str) -> int:
        if x not in rank or y not in rank or rank[x] == rank[y]:
            return 0
        return 1 if rank[x] < rank[y] else -1

    pairs = list(itertools.combinations(candidates, 2))
    return sum(abs(sign(ra, x, y) - sign(rb, x, y)) for x, y in pairs) / (2 * len(pairs))


@dataclass(frozen=True)
class Diversity:
    """Profile diversity (SPEC §6.5), over ballots with positive weight.

    ``distinct_share`` = distinct ballots / n. ``entropy`` is the Shannon entropy (bits)
    of the weighted ballot distribution. ``mean_kendall_tau`` is the weighted mean of
    :func:`kendall_tau_distance` over pairs of different voters (NaN with one voter).
    """

    n_ballots: int
    n_distinct: int
    distinct_share: float
    entropy: float
    mean_kendall_tau: float


def diversity(profile: Profile) -> Diversity:
    rows, inverse, keep = _unique_rows(profile)
    w = profile.weights[keep]
    n, total = int(keep.sum()), float(w.sum())
    p = np.bincount(inverse, weights=w, minlength=len(rows)) / total
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    # Pair signs per distinct ballot: +1 a above b, -1 below, 0 tied or missing.
    ia, ib = np.triu_indices(profile.m, 1)
    a, b = rows[:, ia], rows[:, ib]
    decided = np.isfinite(a) & np.isfinite(b)
    plus = (decided & (a < b)).astype(float)
    minus = (decided & (a > b)).astype(float)
    p_plus, p_minus = p @ plus, p @ minus
    p_zero = 1.0 - p_plus - p_minus
    # Expected distance for two voters drawn with replacement, then drop self-pairs.
    with_replacement = float((2 * p_plus * p_minus + p_zero * (p_plus + p_minus)).mean())
    other_pairs = total**2 - float((w**2).sum())
    tau = with_replacement * total**2 / other_pairs if other_pairs > 0 else math.nan

    return Diversity(
        n_ballots=n,
        n_distinct=len(rows),
        distinct_share=len(rows) / n,
        entropy=max(entropy, 0.0),
        mean_kendall_tau=tau,
    )
