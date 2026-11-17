"""SPEC §8 / §10: prompt templates, hashing and persona rendering."""

from __future__ import annotations

import math

import pytest

from scv.prompts import FORBIDDEN_WORDS, load_template, load_templates, render_persona


def test_shipped_templates_load_and_carry_the_spec_wording():
    t = load_templates()
    assert (t.persona.id, t.thermometer.id, t.ranking.id, t.pairwise.id) == (
        "persona_v1",
        "thermometer_v1",
        "ranking_v4_strict",
        "pairwise_v1",
    )
    rank = t.ranking.render(
        "user", persona="P", candidate_lines="A: X\nB: Y\nC: Z", n="3", letters="A, B, C"
    )
    assert "Here are the 3 candidates, each with a letter:\nA: X\nB: Y\nC: Z\n" in rank
    assert "Rank all 3 candidates from most preferred to least preferred." in rank
    assert "write all 3 letters, one per line, most preferred first" in rank
    assert "using each letter exactly once" in rank and "=" not in rank  # strict order
    assert "all 2 letters (A, B)" in t.ranking.render("repair", letters="A, B", n="2")
    pair = t.pairwise.render("user", persona="P", first="X", second="Y")
    assert "Which do you prefer?\n1: X\n2: Y\nAnswer with 1 or 2." in pair
    assert t.pairwise["labels"] == ["1", "2"]
    assert len({t.persona.hash, t.thermometer.hash, t.ranking.hash, t.pairwise.hash}) == 4
    assert t.for_mode("ranking") is t.ranking


def test_template_hash_tracks_content(tmp_path):
    a = tmp_path / "a.yaml"
    a.write_text("id: x_v1\nuser: 'Hello {name}.'\n")
    b = tmp_path / "b.yaml"
    b.write_text("id: x_v1\nuser: 'Hello {name}.'\n")
    c = tmp_path / "c.yaml"
    c.write_text("id: x_v1\nuser: 'Hello, {name}.'\n")  # one comma, same id
    ha, hb, hc = (load_template(p).hash for p in (a, b, c))
    assert ha == hb != hc
    assert len(ha) == 16


def test_render_requires_exactly_the_placeholders(tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text("id: t\nuser: '{a} and {b}'\n")
    t = load_template(f)
    assert t.render("user", a="1", b="2") == "1 and 2"
    with pytest.raises(KeyError):
        t.render("user", a="1")
    with pytest.raises(KeyError):
        t.render("user", a="1", b="2", c="3")


@pytest.mark.parametrize("word", FORBIDDEN_WORDS)
def test_templates_with_forbidden_words_are_rejected(tmp_path, word):
    f = tmp_path / "bad.yaml"
    f.write_text(f"id: bad\nuser: 'Mind your {word.title()} here.'\n")
    with pytest.raises(ValueError, match="forbidden"):
        load_template(f)
    g = tmp_path / "noid.yaml"
    g.write_text("user: hi\n")
    with pytest.raises(ValueError, match="'id'"):
        load_template(g)


def test_persona_is_first_person_and_drops_missing_fields():
    t = load_templates().persona
    full = render_persona(
        t,
        {
            "age": 45,
            "gender": "a woman",
            "race_ethnicity": "Black",
            "education": "a bachelor's degree",
            "income": "$50,000 to $59,999",
            "state_or_region": "Ohio",
            "party_id_7": "a strong Democrat",
            "ideology_7": "slightly liberal",
            "religious_attendance": "attend religious services almost every week",
            "vote_prior": "voted for Joe Biden",
        },
    )
    assert full == (
        "I am 45 years old. I am a woman. My race or ethnicity is Black. My highest level of "
        "education is a bachelor's degree. My household income is $50,000 to $59,999 a year. "
        "I live in Ohio. Politically, I think of myself as a strong Democrat. My political "
        "views are slightly liberal. I attend religious services almost every week. In the "
        "last presidential election, I voted for Joe Biden."
    )
    sparse = render_persona(
        t,
        {"age": 30, "gender": None, "income": math.nan, "state_or_region": "  ", "ideology_7": ""},
    )
    assert sparse == "I am 30 years old."
    assert "unknown" not in sparse.lower() and "None" not in sparse and "nan" not in sparse
    assert render_persona(t, {}) == ""
    with pytest.raises(KeyError, match="shoe_size"):
        render_persona(t, {"shoe_size": 9})
