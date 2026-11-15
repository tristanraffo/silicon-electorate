"""SPEC §7: null and synthetic electorates."""

from __future__ import annotations

import numpy as np
import pytest

from scv.ballots import is_complete, is_strict
from scv.domains import canonical_axis, sp_axes, sp_deletion_share, sp_max_consistent
from scv.nulls import (
    as_rng,
    impartial_culture,
    permutation_null,
    spatial,
    spatial_1d,
    spatial_2d,
)
from scv.rules import Profile, has_majority_cycle

# --------------------------------------------------------------------------------------
# Impartial culture
# --------------------------------------------------------------------------------------


def test_ic_ballots_are_complete_strict_orders():
    p = impartial_culture(200, 5, rng=0)
    assert p.n == 200 and p.m == 5
    assert np.array_equal(np.sort(p.ranks, axis=1), np.tile(np.arange(5.0), (200, 1)))
    assert all(is_strict(b) and is_complete(b) for b in p.ballots)
    assert p.candidates == ("c1", "c2", "c3", "c4", "c5")


def test_ic_is_seeded():
    a, b = impartial_culture(50, 4, rng=3), impartial_culture(50, 4, rng=3)
    assert np.array_equal(a.ranks, b.ranks)
    assert not np.array_equal(a.ranks, impartial_culture(50, 4, rng=4).ranks)
    rng = np.random.default_rng(3)
    assert np.array_equal(impartial_culture(50, 4, rng=rng).ranks, a.ranks)


def test_ic_is_uniform_over_orders():
    # 6 orders, 60,000 voters: each expected 10,000 with sd ~ 91. Allow 5 sd.
    p = impartial_culture(60_000, 3, rng=1)
    _, counts = np.unique(p.ranks, axis=0, return_counts=True)
    assert len(counts) == 6
    assert np.all(np.abs(counts - 10_000) < 5 * 91.3)


def test_ic_cycle_frequency_matches_guilbaud():
    """SPEC §7 calibration: m = 3, n = 1,001, 20,000 draws -> 8.77% +/- 0.5 pp.

    Guilbaud's limit for three candidates is 1/4 - (3 / (2 pi)) asin(1/3) = 0.08774.
    """
    assert 0.25 - 3 / (2 * np.pi) * np.arcsin(1 / 3) == pytest.approx(0.0877, abs=5e-5)
    rng = np.random.default_rng(1785)  # Condorcet's Essai
    draws = 20_000
    cycles = sum(has_majority_cycle(impartial_culture(1_001, 3, rng)) for _ in range(draws))
    freq = cycles / draws
    print(f"IC cycle frequency (m=3, n=1001, {draws} draws): {freq:.4%}")
    assert abs(freq - 0.0877) < 0.005


# --------------------------------------------------------------------------------------
# Spatial models
# --------------------------------------------------------------------------------------


def test_spatial_1d_is_single_peaked_on_the_true_axis():
    """SPEC §6 test: deletion share is exactly 0 and the true axis is recovered."""
    identified = 0
    for seed in range(40):
        m = 3 + seed % 5  # 3..7 candidates
        e = spatial_1d(300, m, rng=seed)
        true_axis = canonical_axis(e.axis, e.profile.candidates)

        assert sp_deletion_share(e.profile) == 0.0
        assert sp_deletion_share(e.profile, mode="plateaued") == 0.0
        axes = sp_axes(e.profile)
        assert true_axis in axes
        axis, share = sp_max_consistent(e.profile)
        assert share == 1.0 and axis in axes

        # The axis is pinned down (up to reversal) as soon as some voter ranks the
        # candidates in axis order and another in the reverse order, i.e. there is a
        # voter beyond each end. Then the recovered axis must be exactly the true one.
        left_to_right = np.argsort(e.candidate_positions[:, 0])
        order = np.argsort(e.profile.ranks, axis=1)
        has_left = (order == left_to_right).all(axis=1).any()
        has_right = (order == left_to_right[::-1]).all(axis=1).any()
        if has_left and has_right:
            identified += 1
            assert axes == [true_axis]
            assert axis == true_axis
    assert identified >= 30  # the identified case is the norm, not a fluke


def test_spatial_axis_is_left_to_right_and_scores_are_minus_distance():
    e = spatial_1d(10, 4, rng=2, candidates=["W", "X", "Y", "Z"])
    pos = dict(zip(e.profile.candidates, e.candidate_positions[:, 0], strict=True))
    assert [pos[c] for c in e.axis] == sorted(pos.values())
    expected = -np.abs(e.voter_positions - e.candidate_positions.T)
    assert np.allclose(e.scores, expected)
    assert np.array_equal(e.profile.ranks, Profile.from_scores(e.scores, list("WXYZ")).ranks)


def test_spatial_2d_is_not_single_peaked_and_has_no_single_axis():
    e = spatial_2d(1_000, 5, rng=0)
    assert e.dim == 2 and e.candidate_positions.shape == (5, 2)
    assert sp_deletion_share(e.profile) > 0.05
    assert sp_axes(e.profile) == []
    with pytest.raises(ValueError, match="1-D"):
        _ = e.axis


def test_spatial_is_seeded_and_validates():
    a, b = spatial(30, 4, rng=9, dim=2), spatial(30, 4, rng=9, dim=2)
    assert np.array_equal(a.scores, b.scores)
    with pytest.raises(ValueError):
        spatial(5, 3, rng=0, dim=0)
    with pytest.raises(ValueError, match="candidate names"):
        spatial(5, 3, rng=0, candidates=["A", "B"])


def test_rng_must_be_explicit():
    with pytest.raises(TypeError):
        as_rng(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        impartial_culture(5, 3, rng=None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Permutation null
# --------------------------------------------------------------------------------------


def test_permutation_null_keeps_each_candidates_distribution_and_kills_correlation():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=2_000)
    scores = np.column_stack(
        [
            latent + 0.1 * rng.normal(size=2_000),
            latent + 0.1 * rng.normal(size=2_000),
            -latent + 0.1 * rng.normal(size=2_000),
        ]
    )
    scores[::50, 1] = np.nan  # missing values travel with their column
    original = scores.copy()

    null = permutation_null(scores, rng=7)

    assert np.array_equal(scores, original, equal_nan=True)  # input untouched
    assert null.shape == scores.shape
    for j in range(3):  # same multiset of values per candidate, NaNs included
        assert np.array_equal(np.sort(null[:, j]), np.sort(scores[:, j]), equal_nan=True)
    assert not np.array_equal(null, scores, equal_nan=True)

    def corr(x, a, b):
        ok = ~np.isnan(x[:, [a, b]]).any(axis=1)
        return np.corrcoef(x[ok, a], x[ok, b])[0, 1]

    assert corr(scores, 0, 1) > 0.95 and corr(scores, 0, 2) < -0.95
    assert abs(corr(null, 0, 1)) < 0.08 and abs(corr(null, 0, 2)) < 0.08

    assert np.array_equal(null, permutation_null(scores, rng=7), equal_nan=True)  # seeded
    with pytest.raises(ValueError):
        permutation_null(np.zeros(5), rng=0)


def test_permutation_null_destroys_single_peakedness():
    e = spatial_1d(1_000, 5, rng=0)
    assert sp_deletion_share(e.profile) == 0.0
    null = Profile.from_scores(permutation_null(e.scores, rng=1), e.profile.candidates)
    assert sp_deletion_share(null) > 0.2
