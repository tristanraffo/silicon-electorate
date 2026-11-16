"""vLLM backend (SPEC §9): prod, on the CUDA box. Mirrors `hf.py`.

Offline batched `LLM.generate`. Prompts are rendered and tokenized here, with the same
code as the `hf` backend, and handed to vLLM as token ids, so both backends feed a model
identical tokens (and a chat-rendered prompt never gets a second BOS). Sampling
parameters are always set explicitly: plain temperature sampling, one seed per prompt.

vLLM is Linux/CUDA only (`uv sync --extra gpu`); importing this module anywhere else
raises ImportError, which the runner turns into a readable message. Written against the
vLLM 0.33 docs; it has not been run on a GPU from this repo yet (see DECISIONS.md).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from vllm import LLM, SamplingParams

from scv.backends import _tokens
from scv.backends.base import Backend, BackendError, ChatPrompt, per_prompt_seeds

logger = logging.getLogger(__name__)

#: Top-k next-token log-probs requested for pairwise calls (SPEC §9).
LOGPROBS_K = 20


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


class VLLMBackend(Backend):
    """A causal LM served by vLLM's offline engine.

    ``dtype`` defaults to bfloat16. ``base=True`` marks a base model with no chat
    template (completion prompts only, SPEC §10.4). ``tokenizer`` can be injected for
    tests; otherwise the model's Hugging Face tokenizer is loaded. Any other keyword
    (e.g. ``gpu_memory_utilization``, ``max_model_len``) is passed to ``vllm.LLM``.
    """

    def __init__(
        self,
        hf_id: str,
        *,
        dtype: str | None = None,
        revision: str | None = None,
        base: bool = False,
        tokenizer: Any = None,
        **engine_kwargs: Any,
    ) -> None:
        self.hf_id = hf_id
        self.base = base
        self.dtype = dtype or "bfloat16"
        self.missing_label_count = 0  # pairwise calls where a label fell outside the top k
        try:
            if tokenizer is None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
            self.tokenizer = tokenizer
            self.llm = LLM(
                model=hf_id, revision=revision, dtype=self.dtype, seed=0, **engine_kwargs
            )
        except Exception as e:  # noqa: BLE001 - any load failure must fail fast and readably
            reason = "out of GPU memory" if _is_oom(e) else f"{type(e).__name__}: {e}"
            raise BackendError(
                f"could not load {hf_id!r} with vLLM ({self.dtype}): {reason}. "
                "Check the model id, that its licence is accepted and `hf auth login` was run, "
                "and that the model fits on the GPU (try a lower `gpu_memory_utilization` "
                "or `max_model_len` under backend_options)."
            ) from e
        _tokens.require_chat_template(self.tokenizer, base=base, model=hf_id)
        self.revision = _tokens.resolved_revision(hf_id, revision)
        self._label_cache: dict[tuple[tuple[str, ...], bool], list[int]] = {}
        self.check_labels(["1", "2"])  # SPEC §9: assert single-token labels at startup

    # -- prompt handling (same code path as the hf backend) ----------------------------

    def render(self, prompt: ChatPrompt) -> str:
        return _tokens.render(self.tokenizer, prompt, base=self.base, model=self.hf_id)

    def _inputs(self, prompts: Sequence[ChatPrompt]) -> list[dict[str, list[int]]]:
        return [
            {
                "prompt_token_ids": _tokens.encode(
                    self.tokenizer, p, base=self.base, model=self.hf_id
                )
            }
            for p in prompts
        ]

    def check_labels(self, candidates: Sequence[str]) -> None:
        """Raise unless every label is exactly one token right after the answer prefix."""
        self._label_ids(tuple(candidates), completion=self.base)

    def _label_ids(self, candidates: tuple[str, ...], completion: bool) -> list[int]:
        key = (candidates, completion)
        if key not in self._label_cache:
            self._label_cache[key] = _tokens.label_token_ids(
                self.tokenizer, candidates, completion=completion, base=self.base, model=self.hf_id
            )
        return self._label_cache[key]

    def _run(self, inputs: list[dict[str, list[int]]], params: list[Any]) -> list[Any]:
        try:
            return self.llm.generate(inputs, params, use_tqdm=False)
        except Exception as e:  # noqa: BLE001
            if _is_oom(e):
                raise BackendError(
                    f"out of GPU memory running {self.hf_id!r}; lower `batch_size`, "
                    "`gpu_memory_utilization` or `max_model_len`"
                ) from e
            raise

    # -- interface ---------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        return {
            "model_id": self.hf_id,
            "revision": self.revision,
            "dtype": self.dtype,
            "backend": "vllm",
        }

    def generate(
        self,
        prompts: list[ChatPrompt],
        *,
        temperature: float,
        max_tokens: int,
        n: int,
        seed: int | Sequence[int],
    ) -> list[list[str]]:
        seeds = per_prompt_seeds(seed, len(prompts))
        greedy = temperature <= 0
        params = [
            SamplingParams(
                n=1 if greedy else n,  # greedy is deterministic: one sample, repeated
                temperature=0.0 if greedy else temperature,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                max_tokens=max_tokens,
                seed=s,
            )
            for s in seeds
        ]
        outputs = self._run(self._inputs(prompts), params)
        result = []
        for out in outputs:
            texts = [c.text.strip() for c in sorted(out.outputs, key=lambda c: c.index)]
            result.append(texts * n if greedy else texts)
        return result

    def next_token_logprobs(
        self, prompts: list[ChatPrompt], candidates: list[str]
    ) -> list[dict[str, float]]:
        if not prompts:
            return []
        completion = prompts[0].is_completion
        if any(p.is_completion != completion for p in prompts):
            raise BackendError("cannot mix chat and completion prompts in one batch")
        label_ids = self._label_ids(tuple(candidates), completion=completion)
        params = [
            SamplingParams(
                n=1,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                max_tokens=1,
                logprobs=LOGPROBS_K,
                seed=0,
            )
            for _ in prompts
        ]
        outputs = self._run(self._inputs(prompts), params)
        result = []
        for out in outputs:
            top = out.outputs[0].logprobs[0]  # first generated token: {token_id: Logprob}
            found = {tid: _value(lp) for tid, lp in top.items()}
            floor = min(found.values()) if found else 0.0
            raw = []
            for tid in label_ids:
                if tid not in found:
                    # Outside the top k: its true log-prob is at most the k-th best.
                    self.missing_label_count += 1
                raw.append(found.get(tid, floor))
            if all(tid not in found for tid in label_ids):
                logger.warning(
                    "no answer label in the top %d tokens for a pairwise call", LOGPROBS_K
                )
            result.append(dict(zip(candidates, _log_normalise(raw), strict=True)))
        return result


def _value(logprob: Any) -> float:
    """vLLM returns Logprob objects (``.logprob``); accept plain floats too."""
    return float(getattr(logprob, "logprob", logprob))


def _log_normalise(values: list[float]) -> list[float]:
    import math

    peak = max(values)
    total = peak + math.log(sum(math.exp(v - peak) for v in values))
    return [v - total for v in values]
