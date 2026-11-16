"""SPEC §13 M3: tiny-model smoke test on CPU, 5 respondents x 3 candidates.

Opt-in (it loads a ~1 GB model):  uv run pytest -m hf -s
"""

from __future__ import annotations

import json
import math

import pytest

from scv.elicit import (
    ElicitConfig,
    Respondent,
    pairwise_table,
    ranking_table,
    run_elicitation,
    thermometer_table,
)

pytestmark = pytest.mark.hf

TINY = "Qwen/Qwen2.5-0.5B-Instruct"
CANDIDATES = ("Alice Smith", "Bob Jones", "Carol White")  # fictional
PERSONAS = [
    "I am 67 years old. I am a man. I live in Texas. Politically, I think of myself as a "
    "strong Republican.",
    "I am 24 years old. I am a woman. I live in Oregon. Politically, I think of myself as a "
    "strong Democrat.",
    "I am 45 years old. I am a woman. I live in Ohio. My political views are moderate.",
    "I am 33 years old. I am a man. I live in Georgia.",
    "I am 58 years old. I am a woman. I live in Maine. I attend religious services every week.",
]


@pytest.fixture(scope="module")
def backend():
    from scv.backends.hf import HFBackend

    return HFBackend(TINY, device="cpu", dtype="float32")


def test_tiny_model_end_to_end_on_cpu(backend):
    config = ElicitConfig(
        election="smoke",
        candidates=CANDIDATES,
        question_text=(
            "How would you rate this person on a scale from 0 to 100, where 0 is very cold "
            "or unfavorable and 100 is very warm or favorable?"  # test wording, not ANES
        ),
        run_seed=1,
    )
    respondents = [Respondent(f"r{i}", p) for i, p in enumerate(PERSONAS)]
    calls = run_elicitation(config, respondents, backend, model_key="tiny")

    # 5 respondents x (3 candidates x 6 thermometer + 4 ranking + 3 pairs x 2 orders)
    assert len(calls) == 5 * (3 * 6 + 4 + 6) == 140
    assert calls["key"].is_unique
    assert set(calls["backend"]) == {"hf"} and set(calls["hf_id"]) == {TINY}
    assert calls["revision"].notna().all()  # the model revision hash is logged

    pw = calls[calls["mode"] == "pairwise"]
    for lp in pw["logprobs"].map(json.loads):
        assert set(lp) == {"1", "2"}
        assert math.exp(lp["1"]) + math.exp(lp["2"]) == pytest.approx(1.0, abs=1e-5)
    table = pairwise_table(calls)
    assert len(table) == 5 * 3 and table["p_a_over_b"].between(0, 1).all()

    thermo, ranks = thermometer_table(calls), ranking_table(calls)
    assert len(thermo) == 15 and len(ranks) == 20
    gen = calls[calls["mode"] != "pairwise"]
    print(
        f"\n[{TINY} on cpu] parse rate thermometer="
        f"{gen[gen['mode'] == 'thermometer']['parse_ok'].mean():.2f} "
        f"ranking={gen[gen['mode'] == 'ranking']['parse_ok'].mean():.2f} "
        f"mean |position bias|={table['position_bias'].abs().mean():.3f}"
    )
    assert gen[gen["mode"] == "thermometer"]["parse_ok"].mean() > 0.5


def test_sampling_is_reproducible_per_prompt_seed(backend):
    from scv.backends.base import ChatPrompt

    a = ChatPrompt.user("Name a colour. One word.")
    b = ChatPrompt.user("Name an animal. One word.")
    kw = dict(temperature=1.0, max_tokens=6, n=3)
    alone = backend.generate([a], seed=[11], **kw)
    together = backend.generate([b, a], seed=[5, 11], **kw)
    assert alone[0] == together[1]  # same prompt + same seed, regardless of batch-mates
    assert (
        backend.generate([a], temperature=0.0, max_tokens=6, n=2, seed=0)[0][0]
        == (backend.generate([a, b], temperature=0.0, max_tokens=6, n=1, seed=3)[0][0])
    )


def test_chat_template_is_applied_and_labels_are_single_tokens(backend):
    from scv.backends.base import BackendError, ChatPrompt

    text = backend.render(ChatPrompt.user("hello"))
    assert "hello" in text and text != "hello" and text.rstrip().endswith("assistant")
    backend.check_labels(["1", "2"])
    with pytest.raises(BackendError, match="single token"):
        backend.check_labels(["1", "twenty-two"])


def test_missing_model_fails_fast_with_a_clear_message():
    from scv.backends.base import BackendError
    from scv.backends.hf import HFBackend

    with pytest.raises(BackendError, match="could not load"):
        HFBackend("scv-tests/definitely-not-a-model", device="cpu", local_files_only=True)
