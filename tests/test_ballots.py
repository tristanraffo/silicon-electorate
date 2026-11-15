"""SPEC §4: weak-order ballots."""

from __future__ import annotations

import math

import pytest

from scv.ballots import (
    WeakOrder,
    from_scores,
    is_complete,
    is_strict,
    linear_extensions,
    move_up,
    pairwise,
)


def test_spec_example_representation():
    b = WeakOrder([["A"], ["B", "C"], ["D"]])
    assert b.groups == (("A",), ("B", "C"), ("D",))
    assert b.ranked == ("A", "B", "C", "D")
    assert b.missing == frozenset()
    assert b.rank_of() == {"A": 0, "B": 1, "C": 1, "D": 2}


def test_tie_groups_are_canonical_so_equal_ballots_hash_equal():
    a = WeakOrder([["A"], ["C", "B"]], missing={"D"})
    b = WeakOrder((("A",), ("B", "C")), frozenset({"D"}))
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_parse_and_strict_constructors():
    assert WeakOrder.parse("A>B=C>D") == WeakOrder([["A"], ["B", "C"], ["D"]])
    assert WeakOrder.parse(" A > B ") == WeakOrder.strict(["A", "B"])
    assert WeakOrder.strict(["A", "B"], missing=["C"]).missing == {"C"}


@pytest.mark.parametrize(
    "groups, missing, exc",
    [
        ([["A"], ["A", "B"]], (), ValueError),  # ranked twice
        ([["A"], []], (), ValueError),  # empty tie group
        ([["A"], ["B"]], {"B"}, ValueError),  # ranked and missing
        (["A", "B"], (), TypeError),  # bare strings instead of groups
        ("AB", (), TypeError),
        ([["A"], [""]], (), TypeError),  # empty name
        ([["A"]], "B", TypeError),  # missing as a string
    ],
)
def test_invalid_ballots_are_rejected(groups, missing, exc):
    with pytest.raises(exc):
        WeakOrder(groups, missing)


def test_from_scores_orders_ties_and_missing():
    b = from_scores({"A": 85, "B": 60, "C": 60, "D": None, "E": 0})
    assert b == WeakOrder([["A"], ["B", "C"], ["E"]], missing={"D"})


def test_from_scores_nan_is_missing_and_inf_is_rejected():
    assert from_scores({"A": 1.0, "B": math.nan}) == WeakOrder([["A"]], missing={"B"})
    with pytest.raises(ValueError):
        from_scores({"A": math.inf, "B": 1.0})


def test_from_scores_all_missing_gives_empty_ballot():
    b = from_scores({"A": None, "B": None})
    assert b.groups == ()
    assert b.missing == {"A", "B"}


def test_pairwise_signs_ties_and_missing():
    p = pairwise(WeakOrder.parse("A>B=C", missing={"D"}))
    assert p[("A", "B")] == 1 and p[("B", "A")] == -1
    assert p[("A", "C")] == 1 and p[("C", "A")] == -1
    assert p[("B", "C")] == 0 and p[("C", "B")] == 0  # tie
    for x in "ABC":  # missing is neither above nor below anything
        assert p[(x, "D")] == 0 and p[("D", x)] == 0
    assert len(p) == 4 * 3
    assert all(p[(a, b)] == -p[(b, a)] for a, b in p)


def test_is_strict_and_is_complete():
    assert is_strict(WeakOrder.parse("A>B>C"))
    assert not is_strict(WeakOrder.parse("A>B=C"))
    strict_but_partial = WeakOrder.parse("A>B", missing={"C"})
    assert is_strict(strict_but_partial)
    assert not is_complete(strict_but_partial)
    assert is_complete(WeakOrder.parse("A>B=C"))


def test_linear_extensions_of_a_strict_ballot_is_itself():
    b = WeakOrder.parse("A>B>C")
    assert list(linear_extensions(b)) == [b]


def test_linear_extensions_enumerates_every_tie_resolution():
    b = WeakOrder.parse("A=B>C>D=E=F", missing={"G"})
    exts = list(linear_extensions(b))
    assert len(exts) == math.factorial(2) * math.factorial(3)
    assert len(set(exts)) == len(exts)
    original = pairwise(b)
    for e in exts:
        assert is_strict(e)
        assert e.missing == {"G"}
        pe = pairwise(e)
        # an extension only resolves ties; every strict preference is preserved
        assert all(pe[k] == v for k, v in original.items() if v != 0 or "G" in k)


def test_linear_extensions_full_tie_is_all_permutations():
    exts = {e.ranked for e in linear_extensions(WeakOrder([["A", "B", "C"]]))}
    assert len(exts) == 6


def test_linear_extensions_refuses_more_than_seven():
    with pytest.raises(ValueError):
        next(linear_extensions(WeakOrder([list("ABCDEFGH")])))


def test_move_up():
    assert move_up(WeakOrder.parse("A>B>C"), "C") == WeakOrder.parse("A>C>B")  # adjacent swap
    assert move_up(WeakOrder.parse("A>B>C"), "A") == WeakOrder.parse("A>B>C")  # already top
    assert move_up(WeakOrder.parse("A>B=C"), "C") == WeakOrder.parse("A>C>B")  # tie broken
    assert move_up(WeakOrder.parse("A=B>C"), "A") == WeakOrder.parse("A>B>C")  # shared top
    assert move_up(WeakOrder.parse("A=B>C"), "C") == WeakOrder.parse("C>A=B")  # passes a group
    with_missing = WeakOrder.parse("A>B", missing={"C"})
    assert move_up(with_missing, "B") == WeakOrder.parse("B>A", missing={"C"})
    with pytest.raises(ValueError):
        move_up(with_missing, "C")
