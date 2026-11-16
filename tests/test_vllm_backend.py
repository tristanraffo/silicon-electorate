"""SPEC §13 M6: the vLLM code path, unit-tested against a fake `vllm` module.

vLLM itself is Linux/CUDA only and is never imported here. The fake records exactly
what the backend asks of `vllm.LLM` so the tests can check it mirrors the `hf` backend.
"""

from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace

import pytest
import yaml

from scv.backends.base import BackendError, ChatPrompt
from scv.elicit import (
    ElicitConfig,
    Respondent,
    pairwise_table,
    ranking_table,
    run_elicitation,
    thermometer_table,
)
from scv.runner import RunConfig, make_backend, synthetic_respondents

BOS = 1
SPACE_DIGIT = 500_000


class CharTokenizer:
    """One token per character; chat template wraps turns in <role>...</role> markers."""

    chat_template = "fake"

    def __init__(self):
        self.template_kwargs = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False and add_generation_prompt is True
        self.template_kwargs.append(kwargs)
        body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return "\x02" + body + "<assistant>"  # \x02 plays the template's own BOS

    def __call__(self, text, add_special_tokens):
        ids, i = [], 0
        while i < len(text):
            if text[i] == " " and i + 1 < len(text) and text[i + 1].isdigit():
                ids.append(SPACE_DIGIT + int(text[i + 1]))  # " 7" is one token, as in real BPE
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return {"input_ids": ([BOS] if add_special_tokens else []) + ids}


def decode(ids):
    return "".join(
        f" {i - SPACE_DIGIT}" if SPACE_DIGIT <= i < SPACE_DIGIT + 10 else chr(i)
        for i in ids
        if i not in (BOS, 2)
    )


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__dict__.update(kwargs)


class FakeLLM:
    """Stands in for vllm.LLM. `answer(text, params, k)` supplies sample k's text."""

    instances: list[FakeLLM] = []
    fail_on_load: Exception | None = None
    fail_on_generate: Exception | None = None

    def __init__(self, **kwargs):
        if FakeLLM.fail_on_load:
            raise FakeLLM.fail_on_load
        self.kwargs = kwargs
        self.calls = []
        self.logprob_table = {ord("1"): -0.2, ord("2"): -1.8, ord("x"): -3.0}
        self.answer = lambda text, params, k: f" sample{k}-seed{params.seed} "
        FakeLLM.instances.append(self)

    def generate(self, prompts, sampling_params, use_tqdm=True):
        if FakeLLM.fail_on_generate:
            raise FakeLLM.fail_on_generate
        assert use_tqdm is False and len(prompts) == len(sampling_params)
        self.calls.append((prompts, sampling_params))
        outputs = []
        for prompt, params in zip(prompts, sampling_params, strict=True):
            text = decode(prompt["prompt_token_ids"])
            logprobs = None
            if getattr(params, "logprobs", None):
                logprobs = [{t: SimpleNamespace(logprob=v) for t, v in self.logprob_table.items()}]
            # completions arrive out of order on purpose: the backend must sort by index
            samples = [
                SimpleNamespace(index=k, text=self.answer(text, params, k), logprobs=logprobs)
                for k in reversed(range(params.n))
            ]
            outputs.append(SimpleNamespace(outputs=samples))
        return outputs


@pytest.fixture
def vllm_backend(monkeypatch):
    """Import scv.backends.vllm_backend against a fake `vllm` module."""
    fake = types.ModuleType("vllm")
    fake.LLM, fake.SamplingParams = FakeLLM, FakeSamplingParams
    FakeLLM.instances, FakeLLM.fail_on_load, FakeLLM.fail_on_generate = [], None, None
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.delitem(sys.modules, "scv.backends.vllm_backend", raising=False)
    import scv.backends.vllm_backend as module

    yield module
    sys.modules.pop("scv.backends.vllm_backend", None)


def make(module, **kwargs):
    kwargs.setdefault("tokenizer", CharTokenizer())
    return module.VLLMBackend("org/model", **kwargs)


def test_without_vllm_installed_the_import_fails_and_the_runner_explains(tmp_path):
    assert "vllm" not in sys.modules
    with pytest.raises(ImportError):
        import scv.backends.vllm_backend  # noqa: F401
    cfg = RunConfig.model_validate(
        {
            "run_id": "x",
            "backend": "vllm",
            "models": ["tiny"],
            "election": {
                "id": "s",
                "candidates": ["Ann Lee", "Bo Chen", "Cy Diaz"],
                "question_text": "Rate.",
            },
            "results_dir": str(tmp_path),
        }
    )
    with pytest.raises(BackendError, match="uv sync --extra gpu"):
        make_backend(cfg, "tiny", synthetic_respondents(2, 0))


def test_engine_is_loaded_with_model_dtype_revision_and_options(vllm_backend):
    b = make(vllm_backend, revision="abc123", gpu_memory_utilization=0.85, max_model_len=2048)
    llm = FakeLLM.instances[-1]
    assert llm.kwargs == {
        "model": "org/model",
        "revision": "abc123",
        "dtype": "bfloat16",  # SPEC §9 default
        "seed": 0,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 2048,
    }
    assert b.info() == {
        "model_id": "org/model",
        "revision": "abc123",
        "dtype": "bfloat16",
        "backend": "vllm",
    }
    assert make(vllm_backend, dtype="float16").info()["dtype"] == "float16"


def test_generate_uses_plain_temperature_sampling_with_one_seed_per_prompt(vllm_backend):
    b = make(vllm_backend)
    prompts = [ChatPrompt.user("first"), ChatPrompt.user("second")]
    out = b.generate(prompts, temperature=1.0, max_tokens=8, n=3, seed=[11, 22])
    assert out == [
        ["sample0-seed11", "sample1-seed11", "sample2-seed11"],  # sorted by index, stripped
        ["sample0-seed22", "sample1-seed22", "sample2-seed22"],
    ]
    sent, params = FakeLLM.instances[-1].calls[-1]
    assert [p.kwargs for p in params] == [
        {
            "n": 3,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "max_tokens": 8,
            "seed": s,
        }
        for s in (11, 22)
    ]  # the same sampling as hf.py (DECISIONS E6): no top-k / top-p / repetition penalty
    assert b.generate(prompts, temperature=1.0, max_tokens=8, n=1, seed=5) == [
        ["sample0-seed5"],
        ["sample0-seed6"],  # an int seed expands to consecutive per-prompt seeds
    ]


def test_greedy_asks_for_one_sample_and_repeats_it(vllm_backend):
    b = make(vllm_backend)
    out = b.generate([ChatPrompt.user("q")], temperature=0.0, max_tokens=4, n=3, seed=[7])
    assert out == [["sample0-seed7"] * 3]
    params = FakeLLM.instances[-1].calls[-1][1][0]
    assert params.n == 1 and params.temperature == 0.0


def test_prompts_go_in_as_token_ids_with_the_chat_template_and_no_double_bos(vllm_backend):
    tok = CharTokenizer()
    b = make(vllm_backend, tokenizer=tok)
    chat = ChatPrompt(messages=(("user", "hi"), ("assistant", "5"), ("user", "again")))
    b.generate([chat], temperature=0.0, max_tokens=4, n=1, seed=0)
    sent = FakeLLM.instances[-1].calls[-1][0][0]
    assert set(sent) == {"prompt_token_ids"}
    ids = sent["prompt_token_ids"]
    assert decode(ids) == "<user>hi</user><assistant>5</assistant><user>again</user><assistant>"
    assert ids[0] == 2 and BOS not in ids  # the template's own BOS only; nothing added
    assert b.render(chat) == "\x02" + decode(ids)
    assert all(kw == {"enable_thinking": False} for kw in tok.template_kwargs)  # thinking off

    base = make(vllm_backend, base=True)
    base.generate(
        [ChatPrompt.plain("Survey.\nQ: x\nA:")], temperature=0.0, max_tokens=4, n=1, seed=0
    )
    ids = FakeLLM.instances[-1].calls[-1][0][0]["prompt_token_ids"]
    assert ids[0] == BOS and decode(ids) == "Survey.\nQ: x\nA:"  # completions get the usual BOS
    with pytest.raises(BackendError, match="base model"):
        base.generate([ChatPrompt.user("chat")], temperature=0.0, max_tokens=4, n=1, seed=0)


def test_next_token_logprobs_are_renormalised_over_the_labels(vllm_backend):
    b = make(vllm_backend)
    out = b.next_token_logprobs([ChatPrompt.user("1 or 2?"), ChatPrompt.user("again")], ["1", "2"])
    params = FakeLLM.instances[-1].calls[-1][1][0]
    assert (params.logprobs, params.max_tokens, params.temperature, params.n) == (20, 1, 0.0, 1)
    z = math.log(math.exp(-0.2) + math.exp(-1.8))
    for lp in out:
        assert lp == pytest.approx({"1": -0.2 - z, "2": -1.8 - z})
        assert math.exp(lp["1"]) + math.exp(lp["2"]) == pytest.approx(1.0)
    assert b.missing_label_count == 0
    assert b.next_token_logprobs([], ["1", "2"]) == []


def test_a_label_outside_the_top_k_is_bounded_by_the_weakest_listed_token(vllm_backend):
    b = make(vllm_backend)
    FakeLLM.instances[-1].logprob_table = {ord("1"): -0.1, ord("x"): -4.0, ord("y"): -6.0}
    (lp,) = b.next_token_logprobs([ChatPrompt.user("q")], ["1", "2"])
    z = math.log(math.exp(-0.1) + math.exp(-6.0))  # "2" gets the floor, -6.0
    assert lp == pytest.approx({"1": -0.1 - z, "2": -6.0 - z})
    assert b.missing_label_count == 1 and math.exp(lp["1"]) > 0.99

    FakeLLM.instances[-1].logprob_table = {ord("x"): -1.0, ord("y"): -2.0}
    (lp,) = b.next_token_logprobs([ChatPrompt.user("q")], ["1", "2"])
    assert lp == pytest.approx({"1": math.log(0.5), "2": math.log(0.5)})  # no information
    assert b.missing_label_count == 3


def test_labels_must_be_single_tokens_and_chat_models_need_a_template(vllm_backend):
    b = make(vllm_backend)  # "1" / "2" checked at startup
    b.check_labels(["1", "2"])
    with pytest.raises(BackendError, match="single token"):
        b.check_labels(["1", "22"])

    class NoTemplate(CharTokenizer):
        chat_template = None

    with pytest.raises(BackendError, match="base: true"):
        make(vllm_backend, tokenizer=NoTemplate())
    make(vllm_backend, tokenizer=NoTemplate(), base=True)  # fine as a base model

    class SplitsLabels(CharTokenizer):
        def __call__(self, text, add_special_tokens):
            out = super().__call__(text, add_special_tokens)
            if text.endswith(" 1"):  # pretend " 1" is two tokens after "A:"
                out["input_ids"].append(0)
            return out

    with pytest.raises(BackendError, match="single token"):
        make(vllm_backend, tokenizer=SplitsLabels(), base=True)


def test_load_and_memory_failures_fail_fast_with_clear_messages(vllm_backend):
    FakeLLM.fail_on_load = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    with pytest.raises(BackendError, match="out of GPU memory"):
        make(vllm_backend)
    FakeLLM.fail_on_load = OSError("org/model is not a valid model identifier")
    with pytest.raises(BackendError, match="could not load 'org/model'.*licence"):
        make(vllm_backend)
    FakeLLM.fail_on_load = None

    b = make(vllm_backend)
    FakeLLM.fail_on_generate = RuntimeError("CUDA out of memory")
    with pytest.raises(BackendError, match="out of GPU memory running"):
        b.generate([ChatPrompt.user("q")], temperature=1.0, max_tokens=4, n=1, seed=0)
    with pytest.raises(BackendError, match="out of GPU memory running"):
        b.next_token_logprobs([ChatPrompt.user("q")], ["1", "2"])
    FakeLLM.fail_on_generate = ValueError("something else")
    with pytest.raises(ValueError, match="something else"):  # other errors are not disguised
        b.generate([ChatPrompt.user("q")], temperature=1.0, max_tokens=4, n=1, seed=0)


def test_full_elicitation_runs_through_the_vllm_path(vllm_backend):
    cands = ("Ann Lee", "Bo Chen", "Cy Diaz")
    b = make(vllm_backend)

    def answer(text, params, k):
        question = text.rsplit("<user>", 1)[-1]
        if "Rank these candidates" in question:
            listed = sorted(cands, key=question.index)
            return "\n".join(listed)
        return "64"

    FakeLLM.instances[-1].answer = answer
    cfg = ElicitConfig(election="e", candidates=cands, question_text="Rate this person.")
    people = [Respondent(f"r{i}", f"I am {30 + i} years old.") for i in range(4)]
    calls = run_elicitation(cfg, people, b, "fake")
    assert len(calls) == 4 * (3 * 6 + 4 + 6) and calls["parse_ok"].all()
    assert set(calls["backend"]) == {"vllm"} and set(calls["hf_id"]) == {"org/model"}
    assert (thermometer_table(calls)["score"] == 64).all()
    for row in ranking_table(calls).itertuples():
        assert row.ballot.ranked == tuple(row.candidates_shown)
    pairs = pairwise_table(calls)
    p1 = math.exp(-0.2) / (math.exp(-0.2) + math.exp(-1.8))
    assert pairs["p1_a_first"].to_numpy() == pytest.approx(p1)
    assert pairs["p_a_over_b"].to_numpy() == pytest.approx(0.5)  # pure position bias cancels
    assert pairs["position_bias"].to_numpy() == pytest.approx(p1 - 0.5)


def test_runner_builds_the_vllm_backend_from_models_yaml(vllm_backend, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: CharTokenizer()
    )
    models = tmp_path / "models.yaml"
    models.write_text(
        yaml.safe_dump(
            {
                "pool": {"chat": {"hf_id": "org/chat", "dtype": "bfloat16"}},
                "training_stage": {"raw": {"hf_id": "org/raw", "base": True}},
            }
        )
    )
    cfg = RunConfig.model_validate(
        {
            "run_id": "x",
            "backend": "vllm",
            "models": ["chat", "raw"],
            "models_file": str(models),
            "election": {
                "id": "s",
                "candidates": ["Ann Lee", "Bo Chen", "Cy Diaz"],
                "question_text": "Rate.",
            },
            "backend_options": {"gpu_memory_utilization": 0.9},
            "results_dir": str(tmp_path),
        }
    )
    people = synthetic_respondents(2, 0)
    chat = make_backend(cfg, "chat", people)
    assert isinstance(chat, vllm_backend.VLLMBackend) and not chat.base
    assert FakeLLM.instances[-1].kwargs["model"] == "org/chat"
    assert FakeLLM.instances[-1].kwargs["gpu_memory_utilization"] == 0.9
    raw = make_backend(cfg, "raw", people)
    assert raw.base and raw.dtype == "bfloat16"
