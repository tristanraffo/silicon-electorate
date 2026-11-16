"""SPEC §10: elicitation, tested end to end against the mock backend."""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest

from scv.backends.base import Backend, ChatPrompt
from scv.backends.mock import MockBackend
from scv.ballots import WeakOrder, from_scores
from scv.elicit import (
    GREEDY,
    ElicitConfig,
    Respondent,
    derive_seed,
    execute,
    pairwise_coherence,
    pairwise_table,
    parse_ranking,
    parse_thermometer,
    plan_calls,
    ranking_table,
    run_elicitation,
    shuffled,
    thermometer_table,
)
from scv.prompts import FORBIDDEN_WORDS

CANDS = ("Ada Lovelace", "Grace Hopper", "Alan Turing", "Edsger Dijkstra", "Barbara Liskov")
QUESTION = "How would you rate this person? (Test wording, not the ANES text.)"


def make_world(n: int, candidates=CANDS, seed: int = 0):
    """n voters with distinct integer utilities in 0..100, and their personas."""
    rng = np.random.default_rng(seed)
    utilities, respondents = {}, []
    for i in range(n):
        values = rng.choice(101, size=len(candidates), replace=False)
        rid = f"r{i:04d}"
        utilities[rid] = dict(zip(candidates, (float(v) for v in values), strict=True))
        # the persona names a candidate on purpose: the question's candidate must still win
        respondents.append(
            Respondent(rid, f"I am {20 + i % 60} years old. I once met {candidates[i % 2]}.")
        )
    return utilities, respondents


def config(**kw) -> ElicitConfig:
    return ElicitConfig(
        **{"election": "test", "candidates": CANDS, "question_text": QUESTION, **kw}
    )


def truth(utilities: dict) -> dict[str, WeakOrder]:
    return {rid: from_scores(u) for rid, u in utilities.items()}


class Recorder(Backend):
    """Wraps a backend and records every prompt it is sent."""

    def __init__(self, inner: Backend):
        self.inner, self.generated, self.scored = inner, [], []

    def info(self):
        return self.inner.info()

    def generate(self, prompts, **kw):
        self.generated += [(p, kw["temperature"], kw["n"]) for p in prompts]
        return self.inner.generate(prompts, **kw)

    def next_token_logprobs(self, prompts, candidates):
        self.scored += prompts
        return self.inner.next_token_logprobs(prompts, candidates)


# --------------------------------------------------------------------------------------
# M3 acceptance: the mock's hidden utilities are recovered
# --------------------------------------------------------------------------------------


def test_end_to_end_recovers_hidden_rankings_at_zero_noise():
    """SPEC §13 M3: > 95% of mock voters recovered at noise 0 (here: all of them, by all
    three elicitation modes)."""
    utilities, respondents = make_world(200)
    calls = run_elicitation(config(), respondents, MockBackend(utilities), model_key="mock")
    want = truth(utilities)

    thermo = thermometer_table(calls)
    by_thermometer = {
        rid: from_scores(dict(zip(g["candidate"], g["score"], strict=True)))
        for rid, g in thermo.groupby("respondent_id")
    }
    assert np.mean([by_thermometer[r] == want[r] for r in want]) > 0.95
    assert by_thermometer == want
    truth_scores = {(r, c): u for r, cu in utilities.items() for c, u in cu.items()}
    for row in thermo.itertuples():
        assert row.score == row.greedy == truth_scores[(row.respondent_id, row.candidate)]
        assert (row.n_ok, row.n_samples, row.sd) == (5, 5, 0.0)

    ranks = ranking_table(calls)
    assert len(ranks) == 200 * 4 and ranks["parse_ok"].all()
    assert all(row.ballot == want[row.respondent_id] for row in ranks.itertuples())

    pairs = pairwise_table(calls)
    assert len(pairs) == 200 * 10
    for rid, g in pairs.groupby("respondent_id"):
        p = {(a, b): v for a, b, v in zip(g["a"], g["b"], g["p_a_over_b"], strict=True)}
        u = utilities[rid]
        assert all((v > 0.5) == (u[a] > u[b]) for (a, b), v in p.items())
        greedy = ranks[(ranks["respondent_id"] == rid) & (ranks["sample_idx"] == GREEDY)]
        co = pairwise_coherence(p, CANDS, greedy["ballot"].iloc[0])
        assert co["wst_violations"] == 0 and co["strict_cycles"] == 0
        assert co["ranking_agreement"] == 1.0
    assert np.allclose(pairs["position_bias"], 0.0)


def test_call_plan_counts_keys_and_log_fields():
    utilities, respondents = make_world(3)
    cfg = config()
    calls = plan_calls(cfg, respondents, "m1")
    m = len(CANDS)
    per_respondent = m * 6 + 4 + m * (m - 1)  # thermometer 5+1 each, ranking 3+1, pairs x 2
    assert len(calls) == 3 * per_respondent
    assert len({c.key for c in calls}) == len(calls)
    assert [c.key for c in plan_calls(cfg, respondents, "m1")] == [c.key for c in calls]

    by_mode = pd.Series([c.mode for c in calls]).value_counts().to_dict()
    assert by_mode == {"thermometer": 90, "ranking": 12, "pairwise": 60}
    assert {c.temperature for c in calls if c.sample_idx == GREEDY} == {0.0}
    assert {c.temperature for c in calls if c.mode != "pairwise" and c.sample_idx >= 0} == {1.0}

    df = run_elicitation(cfg, respondents, MockBackend(utilities), model_key="m1")
    assert sorted(df["key"]) == sorted(c.key for c in calls)
    assert set(df.columns) == {
        "key", "election", "respondent_id", "model_key", "hf_id", "revision", "backend",
        "mode", "target", "candidates_shown", "prompt_hash", "temperature", "seed",
        "sample_idx", "raw_text", "parsed", "logprobs", "parse_ok", "retries",
        "latency_ms", "timestamp",
    }  # fmt: skip
    assert set(df["backend"]) == {"mock"} and df["parse_ok"].all() and (df["retries"] == 0).all()
    pw = df[df["mode"] == "pairwise"].iloc[0]
    lp = json.loads(pw["logprobs"])
    assert math.exp(lp["1"]) + math.exp(lp["2"]) == pytest.approx(1.0)
    assert json.loads(pw["parsed"]) == pytest.approx(math.exp(lp["1"]))


def test_samples_sharing_a_prompt_are_requested_together():
    utilities, respondents = make_world(2)
    rec = Recorder(MockBackend(utilities))
    run_elicitation(config(modes=("thermometer",)), respondents, rec, model_key="m")
    # per respondent and candidate: one n=5 sampled request and one greedy request
    assert len(rec.generated) == 2 * len(CANDS) * 2
    assert sorted({(t, n) for _, t, n in rec.generated}) == [(0.0, 1), (1.0, 5)]


# --------------------------------------------------------------------------------------
# Prompt hygiene (SPEC §10.5)
# --------------------------------------------------------------------------------------


def test_text_is_identical_across_models_and_keys_are_not():
    _, respondents = make_world(4)
    a = plan_calls(config(), respondents, "llama")
    b = plan_calls(config(), respondents, "qwen")
    assert [c.prompt.text for c in a] == [c.prompt.text for c in b]
    assert [c.candidates_shown for c in a] == [c.candidates_shown for c in b]
    assert not {c.key for c in a} & {c.key for c in b}


def test_no_prompt_mentions_the_research_question():
    _, respondents = make_world(5)
    calls = plan_calls(config(), respondents, "m") + plan_calls(
        config(sequential=True, modes=("thermometer",)), respondents, "m"
    )
    for c in calls:
        text = (c.prompt.text + " " + c.repair).lower()
        assert not any(w in text for w in FORBIDDEN_WORDS)
        assert "transitiv" not in text and "condorcet" not in text
    with pytest.raises(ValueError, match="forbidden"):
        config(question_text="For consistency, how do you rate this person?")
    with pytest.raises(ValueError, match="reserved"):
        config(candidates=("A=B", "C"))


def test_ranking_order_is_randomised_per_call_stored_and_seeded():
    _, respondents = make_world(30)
    calls = [c for c in plan_calls(config(), respondents, "m") if c.mode == "ranking"]
    for c in calls:
        assert sorted(c.candidates_shown) == sorted(CANDS)
        positions = [c.prompt.text.rindex(name) for name in c.candidates_shown]
        assert positions == sorted(positions)  # the stored order is the order in the text
    orders = {c.candidates_shown for c in calls}
    assert len(orders) > 40  # 120 calls over 120 possible orders: many distinct ones
    first_shown = pd.Series([c.candidates_shown[0] for c in calls]).value_counts()
    assert len(first_shown) == 5 and first_shown.min() >= 10  # no candidate is favoured
    per_respondent = [c.candidates_shown for c in calls if c.respondent_id == "r0000"]
    assert len(set(per_respondent)) > 1  # varies across samples of one respondent

    again = [c for c in plan_calls(config(), respondents, "m") if c.mode == "ranking"]
    assert [c.candidates_shown for c in again] == [c.candidates_shown for c in calls]
    other = [c for c in plan_calls(config(run_seed=1), respondents, "m") if c.mode == "ranking"]
    assert [c.candidates_shown for c in other] != [c.candidates_shown for c in calls]


def test_seed_helpers_are_stable():
    assert derive_seed("a", 1) == derive_seed("a", 1) != derive_seed("a", 2)
    assert 0 <= derive_seed("x") < 2**31
    assert shuffled(CANDS, 1, "r") == shuffled(CANDS, 1, "r")
    assert sorted(shuffled(CANDS, 1, "r")) == sorted(CANDS)


def test_pairwise_runs_both_presentation_orders():
    _, respondents = make_world(1)
    calls = [c for c in plan_calls(config(), respondents, "m") if c.mode == "pairwise"]
    shown = {c.candidates_shown for c in calls}
    assert shown == set(itertools.permutations(CANDS, 2))
    for c in calls:
        first, second = c.candidates_shown
        assert f"1: {first}\n2: {second}\nAnswer with 1 or 2." in c.prompt.text
        a, b = c.target.split("|")
        assert CANDS.index(a) < CANDS.index(b)  # the pair id does not depend on the order


# --------------------------------------------------------------------------------------
# Noise, position bias, failures
# --------------------------------------------------------------------------------------


def test_noise_shows_up_in_samples_but_not_in_greedy():
    utilities, respondents = make_world(40)
    calls = run_elicitation(
        config(modes=("thermometer",)), respondents, MockBackend(utilities, noise=8.0), "m"
    )
    t = thermometer_table(calls)
    u = np.array([utilities[r][c] for r, c in zip(t["respondent_id"], t["candidate"], strict=True)])
    assert np.array_equal(t["greedy"].to_numpy(), u)
    assert (t["sd"] > 0).mean() > 0.95
    assert 1.0 < np.abs(t["score"] - u).mean() < 8.0  # the median of 5 is noisy but centred
    assert abs((t["score"] - u).mean()) < 1.0
    assert t["score"].between(0, 100).all()


def test_position_bias_is_measured_and_cancelled():
    utilities = {"r0": {"Ann": 60.0, "Bob": 50.0, "Cat": 50.0}}
    cfg = ElicitConfig(election="t", candidates=("Ann", "Bob", "Cat"), question_text=QUESTION)
    backend = MockBackend(utilities, position_bias=5.0, pairwise_scale=10.0)
    pairs = pairwise_table(
        run_elicitation(cfg, [Respondent("r0", "I am 40 years old.")], backend, "m")
    )
    sig = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731
    ab = pairs[(pairs["a"] == "Ann") & (pairs["b"] == "Bob")].iloc[0]
    # P(1 | A first) = sigmoid((60 - 50 + 5) / 10); P(1 | B first) = sigmoid((50 - 60 + 5) / 10)
    assert ab["p1_a_first"] == pytest.approx(sig(1.5))
    assert ab["p1_b_first"] == pytest.approx(sig(-0.5))
    assert ab["p_a_over_b"] == pytest.approx(0.5 * (sig(1.5) + 1 - sig(-0.5)))
    assert ab["position_bias"] == pytest.approx(0.5 * (sig(1.5) + sig(-0.5)) - 0.5)
    assert ab["position_bias"] > 0
    # B and C have equal utility: each order prefers whoever is listed first, and the
    # two-order average lands exactly on indifference.
    bc = pairs[(pairs["a"] == "Bob") & (pairs["b"] == "Cat")].iloc[0]
    assert bc["p1_a_first"] == bc["p1_b_first"] == pytest.approx(sig(0.5))
    assert bc["p_a_over_b"] == pytest.approx(0.5)
    assert bc["position_bias"] == pytest.approx(sig(0.5) - 0.5)


def test_position_bias_shifts_mock_rankings_toward_the_listed_order():
    utilities = {"r0": {"Ann": 50.0, "Bob": 50.0, "Cat": 50.0}}
    cfg = ElicitConfig(election="t", candidates=("Ann", "Bob", "Cat"), question_text=QUESTION)
    r = [Respondent("r0", "I am 40 years old.")]
    tied = ranking_table(run_elicitation(cfg, r, MockBackend(utilities), "m"))
    assert all(b == WeakOrder([["Ann", "Bob", "Cat"]]) for b in tied["ballot"])  # ties are kept
    biased = ranking_table(run_elicitation(cfg, r, MockBackend(utilities, position_bias=1.0), "m"))
    for row in biased.itertuples():
        assert row.ballot == WeakOrder.strict(row.candidates_shown)


def test_parse_failures_are_retried_twice_then_flagged():
    utilities, respondents = make_world(3)
    rec = Recorder(MockBackend(utilities, fail_rate=1.0))
    cfg = config(modes=("thermometer", "ranking"))
    calls = run_elicitation(cfg, respondents, rec, "m")
    assert len(calls) == 3 * (5 * 6 + 4)
    assert not calls["parse_ok"].any()
    assert (calls["retries"] == 2).all()
    assert set(calls["parsed"]) == {"null"}
    assert thermometer_table(calls)["score"].isna().all()
    assert thermometer_table(calls)["n_ok"].eq(0).all()
    assert ranking_table(calls)["ballot"].isna().all()
    # a retry continues the conversation: bad answer, then the repair message
    retry_prompts = [p for p, _, _ in rec.generated if len(p.messages) > 1]
    assert len(retry_prompts) == 2 * len(calls)
    deepest = max(retry_prompts, key=lambda p: len(p.messages))
    assert [role for role, _ in deepest.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert deepest.messages[1][1] == "I'm not sure how to answer that."
    assert deepest.messages[2][1] in (
        cfg.templates.thermometer["repair"],
        cfg.templates.ranking["repair"],
    )

    no_retry = run_elicitation(
        cfg, respondents, MockBackend(utilities, fail_rate=1.0), "m", max_retries=0
    )
    assert (no_retry["retries"] == 0).all() and not no_retry["parse_ok"].any()


def test_occasional_failures_are_repaired():
    utilities, respondents = make_world(60)
    calls = run_elicitation(
        config(modes=("thermometer", "ranking")),
        respondents,
        MockBackend(utilities, fail_rate=0.3),
        "m",
    )
    assert (calls["retries"] > 0).mean() > 0.15
    assert calls["parse_ok"].mean() > 0.95  # 0.3^3 = 2.7% stay broken
    assert calls["key"].is_unique
    fixed = calls[(calls["retries"] > 0) & calls["parse_ok"]]
    assert len(fixed) > 0 and not fixed["raw_text"].str.contains("not sure").any()
    want = truth(utilities)
    ok = ranking_table(calls).dropna(subset=["ballot"])
    assert all(row.ballot == want[row.respondent_id] for row in ok.itertuples())


def test_a_subset_of_calls_can_be_executed_on_its_own():
    # what the runner does on resume: only the missing rows are requested
    utilities, respondents = make_world(2)
    calls = plan_calls(config(), respondents, "m")
    todo = [c for i, c in enumerate(calls) if i % 3 == 0]
    rows = [r for batch in execute(todo, MockBackend(utilities), batch_size=7) for r in batch]
    assert sorted(r["key"] for r in rows) == sorted(c.key for c in todo)
    full = run_elicitation(config(), respondents, MockBackend(utilities), "m").set_index("key")
    for r in rows:
        assert r["parsed"] == full.loc[r["key"], "parsed"]


# --------------------------------------------------------------------------------------
# Sequential thermometer and base models
# --------------------------------------------------------------------------------------


def test_sequential_thermometer_is_one_growing_conversation_per_sample():
    utilities, respondents = make_world(6)
    cfg = config(sequential=True, modes=("thermometer",))
    plan = plan_calls(cfg, respondents, "m")
    assert len(plan) == 6 * 6 * len(CANDS)
    chains = {c.chain for c in plan}
    assert len(chains) == 6 * 6 and None not in chains
    orders = {c.candidates_shown for c in plan if c.respondent_id == "r0000"}
    assert len(orders) > 1 and all(sorted(o) == sorted(CANDS) for o in orders)

    rec = Recorder(MockBackend(utilities))
    calls = pd.DataFrame([r for b in execute(plan, rec, batch_size=5) for r in b])
    assert sorted(calls["key"]) == sorted(c.key for c in plan)
    t = thermometer_table(calls)
    for row in t.itertuples():
        assert row.score == row.greedy == utilities[row.respondent_id][row.candidate]

    lengths = sorted({len(p.messages) for p, _, _ in rec.generated})
    assert lengths == [1, 3, 5, 7, 9]  # turn k carries k earlier question/answer pairs
    last = next(p for p, _, _ in rec.generated if len(p.messages) == 9)
    assert QUESTION in last.messages[0][1] and QUESTION not in last.messages[8][1]
    asked = [next(c for c in CANDS if c in msg) for role, msg in last.messages[2::2]]
    assert len(set(asked)) == 4

    with pytest.raises(ValueError, match="incomplete"):
        list(execute([c for c in plan if c.turn != 1], MockBackend(utilities)))
    assert cfg.prompt_hash("thermometer") != config().prompt_hash("thermometer")


def test_base_model_uses_a_completion_frame_and_skips_ranking():
    utilities, respondents = make_world(20)
    cfg = config(base_model=True, modes=("thermometer", "pairwise"))
    plan = plan_calls(cfg, respondents, "olmo2_base")
    assert all(c.prompt.is_completion for c in plan)
    example = next(c for c in plan if c.mode == "thermometer").prompt.completion
    assert example.startswith("Survey transcript.\n") and example.endswith("\nA:")
    assert "Q: " in example and respondents[0].persona in example

    calls = pd.DataFrame([r for b in execute(plan, MockBackend(utilities)) for r in b])
    t = thermometer_table(calls)
    assert all(r.score == utilities[r.respondent_id][r.candidate] for r in t.itertuples())
    pairs = pairwise_table(calls)
    assert all(
        (r.p_a_over_b > 0.5) == (utilities[r.respondent_id][r.a] > utilities[r.respondent_id][r.b])
        for r in pairs.itertuples()
    )
    with pytest.raises(ValueError, match="base models"):
        config(base_model=True)  # default modes include ranking
    with pytest.raises(ValueError, match="sequential"):
        config(base_model=True, modes=("thermometer",), sequential=True)
    assert cfg.prompt_hash("pairwise") != config().prompt_hash("pairwise")  # format is logged


def test_chat_prompt_helpers():
    p = ChatPrompt.user("q1", respondent_id="r")
    q = p.extended("a1", "q2")
    assert q.messages == (("user", "q1"), ("assistant", "a1"), ("user", "q2"))
    assert q.last_turn == "q2" and q.meta == {"respondent_id": "r"}
    c = ChatPrompt.plain("Survey.\nQ: one\nA:").extended(" 5 ", "two")
    assert c.completion == "Survey.\nQ: one\nA: 5\nQ: two\nA:" and c.last_turn == " two\nA:"
    assert ChatPrompt.user("x", a=1) == ChatPrompt.user("x", a=2)  # meta is not identity
    with pytest.raises(ValueError):
        ChatPrompt()
    with pytest.raises(ValueError):
        ChatPrompt(messages=(("user", "x"),), completion="y")


# --------------------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("72", 72),
        (" 0 ", 0),
        ("100", 100),
        ("I would say 85 degrees.", 85),
        ("85.", 85),
        ("Rating: 40/100", 40),
        ("150", None),  # out of range
        ("150, or maybe 60", 60),  # first integer that is in range
        ("warm", None),
        ("", None),
        ("-5", 5),  # the sign is not part of the integer; SPEC: first integer in 0-100
        ("3.5", 3),
        ("0.75", 0),
    ],
)
def test_parse_thermometer(text, expected):
    assert parse_thermometer(text) == expected


def test_parse_ranking():
    c = ["Ann Lee", "Bo Chen", "Cy Diaz"]
    assert parse_ranking("Ann Lee\nBo Chen\nCy Diaz", c) == WeakOrder.parse(
        "Ann Lee>Bo Chen>Cy Diaz"
    )
    assert parse_ranking("Bo Chen = Cy Diaz\nAnn Lee", c) == WeakOrder.parse(
        "Bo Chen=Cy Diaz>Ann Lee"
    )
    assert parse_ranking("1. Cy Diaz\n2. ann lee\n3) **Bo Chen**\n", c) == WeakOrder.parse(
        "Cy Diaz>Ann Lee>Bo Chen"
    )
    assert parse_ranking("- Ann Lee\n\n- Bo Chen=Cy Diaz", c) == WeakOrder.parse(
        "Ann Lee>Bo Chen=Cy Diaz"
    )
    assert parse_ranking("Ann Lee > Bo Chen > Cy Diaz", c) == WeakOrder.parse(
        "Ann Lee>Bo Chen>Cy Diaz"
    )
    assert parse_ranking("Ann Lee=Bo Chen=Cy Diaz", c) == WeakOrder([c])
    assert parse_ranking("Ann Lee\nBo Chen", c) is None  # one missing
    assert parse_ranking("Ann Lee\nBo Chen\nCy Diaz\nAnn Lee", c) is None  # repeated
    assert parse_ranking("Ann Lee\nBo Chen\nDee Fox", c) is None  # unknown name
    assert parse_ranking("Here is my ranking:\nAnn Lee\nBo Chen\nCy Diaz", c) is None  # chatter
    assert parse_ranking("", c) is None


# --------------------------------------------------------------------------------------
# H4 coherence metrics
# --------------------------------------------------------------------------------------


def test_pairwise_coherence_by_hand():
    cands = ["A", "B", "C"]
    cyc = {("A", "B"): 0.9, ("B", "C"): 0.8, ("C", "A"): 0.7}  # A > B > C > A
    out = pairwise_coherence(cyc, cands, WeakOrder.parse("A>B>C"))
    assert (out["n_triples"], out["wst_violations"], out["strict_cycles"]) == (1, 1, 1)
    assert out["wst_violation_share"] == 1.0 and out["strict_cycle_share"] == 1.0
    assert out["ranking_agreement"] == pytest.approx(2 / 3)  # A>B and B>C agree; A>C does not

    transitive = {("A", "B"): 0.9, ("B", "C"): 0.8, ("A", "C"): 0.95}
    out = pairwise_coherence(transitive, cands, WeakOrder.parse("C>B>A"))
    assert out["wst_violations"] == 0 and out["strict_cycles"] == 0
    assert out["ranking_agreement"] == 0.0

    # WST can fail without a strict cycle: p(A>B) = p(B>C) = .5 exactly but p(A>C) < .5
    knife = {("A", "B"): 0.5, ("B", "C"): 0.5, ("A", "C"): 0.4}
    out = pairwise_coherence(knife, cands)
    assert out["wst_violations"] == 1 and out["strict_cycles"] == 0
    assert math.isnan(out["ranking_agreement"])

    # ties in the ranking are left out of the agreement denominator
    out = pairwise_coherence(transitive, cands, WeakOrder.parse("A=B>C"))
    assert out["ranking_agreement"] == 1.0
    assert math.isnan(
        pairwise_coherence(transitive, cands, WeakOrder([cands]))["ranking_agreement"]
    )

    four = {(a, b): 0.6 for a, b in itertools.combinations("ABCD", 2)}
    four[("A", "C")] = 0.4  # only triple {A,B,C} is cyclic
    out = pairwise_coherence(four, list("ABCD"))
    assert (out["n_triples"], out["strict_cycles"], out["wst_violations"]) == (4, 1, 1)
    assert out["strict_cycle_share"] == 0.25


def test_pairwise_table_needs_both_orders():
    utilities, respondents = make_world(1)
    calls = run_elicitation(config(modes=("pairwise",)), respondents, MockBackend(utilities), "m")
    with pytest.raises(ValueError, match="both presentation orders"):
        pairwise_table(calls.iloc[::2])
