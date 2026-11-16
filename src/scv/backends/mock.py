"""Deterministic mock backend (SPEC §9): a voter with a hidden utility vector, plus
position bias and noise. Every test of elicitation and the runner uses this.

The mock learns *who* it is from ``prompt.meta["respondent_id"]`` and *what kind* of
question it is from ``prompt.meta["mode"]``, but it reads the candidates and their
presentation order from the rendered prompt text, so a test fails if the text shown to
the model disagrees with the order the caller recorded.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from scv.backends.base import Backend, ChatPrompt, per_prompt_seeds


class MockBackend(Backend):
    """Answers as voters with known utilities.

    Parameters
    ----------
    utilities:
        ``respondent_id -> {candidate: utility}`` on the 0-100 thermometer scale.
    noise:
        Standard deviation (thermometer points) of response noise. Applied to sampled
        generations (``temperature > 0``) and to pairwise logits; greedy generations
        are noise-free.
    position_bias:
        Bonus, in thermometer points, for being shown earlier: the first-listed option
        in a pairwise choice gets ``+position_bias``; in a ranking, each step down the
        list costs ``position_bias``.
    fail_rate:
        Probability that a generation is unparseable text (to exercise retries).
    pairwise_scale:
        Utility difference that corresponds to one logit in pairwise choices.
    calls:
        Counters of prompts served, by kind (``generate`` / ``logprobs``).
    """

    def __init__(
        self,
        utilities: Mapping[str, Mapping[str, float]],
        *,
        noise: float = 0.0,
        position_bias: float = 0.0,
        fail_rate: float = 0.0,
        pairwise_scale: float = 10.0,
        model_id: str = "mock",
    ) -> None:
        self.utilities = {str(k): dict(v) for k, v in utilities.items()}
        self.noise = noise
        self.position_bias = position_bias
        self.fail_rate = fail_rate
        self.pairwise_scale = pairwise_scale
        self.model_id = model_id
        self.calls = {"generate": 0, "logprobs": 0}

    # -- interface --------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "revision": "mock", "dtype": "none", "backend": "mock"}

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
        self.calls["generate"] += len(prompts)
        out = []
        for prompt, s in zip(prompts, seeds, strict=True):
            samples = []
            for k in range(n):
                rng = self._rng(prompt, s, k)
                if rng.random() < self.fail_rate:
                    samples.append("I'm not sure how to answer that.")
                    continue
                noise = self.noise if temperature > 0 else 0.0
                samples.append(self._answer(prompt, rng, noise))
            out.append(samples)
        return out

    def next_token_logprobs(
        self, prompts: list[ChatPrompt], candidates: list[str]
    ) -> list[dict[str, float]]:
        if len(candidates) != 2:
            raise ValueError("the mock only answers two-way choices")
        self.calls["logprobs"] += len(prompts)
        out = []
        for prompt in prompts:
            u = self._utilities(prompt)
            first, second = self._shown(prompt, u)[-2:]  # the two options come last
            rng = self._rng(prompt, 0, 0)
            diff = u[first] - u[second] + self.position_bias + self.noise * rng.normal()
            logit = diff / self.pairwise_scale
            # log-sigmoid, written to stay finite for large |logit|
            lp_first = -math.log1p(math.exp(-logit)) if logit > -700 else logit
            lp_second = -math.log1p(math.exp(logit)) if logit < 700 else -logit
            out.append({candidates[0]: lp_first, candidates[1]: lp_second})
        return out

    # -- behaviour --------------------------------------------------------------------

    def _utilities(self, prompt: ChatPrompt) -> dict[str, float]:
        rid = str(prompt.meta.get("respondent_id"))
        if rid not in self.utilities:
            raise KeyError(f"mock backend has no utilities for respondent {rid!r}")
        return self.utilities[rid]

    @staticmethod
    def _shown(prompt: ChatPrompt, utilities: Mapping[str, float]) -> list[str]:
        """Candidates named in the question being answered, in the order they appear.

        The question is the most recent user turn that names a candidate (a repair
        message names none, so the turn before it is used). Each name's *last*
        occurrence counts, so a name inside the persona text does not.
        """
        if prompt.is_completion:
            turns = prompt.completion.split("Q:")
        else:
            turns = [content for role, content in prompt.messages if role == "user"]
        for text in reversed(turns):
            found = sorted((text.rfind(c), c) for c in utilities if c in text)
            if found:
                return [c for _, c in found]
        raise ValueError("mock backend found no known candidate in the prompt")

    def _answer(self, prompt: ChatPrompt, rng: np.random.Generator, noise: float) -> str:
        u = self._utilities(prompt)
        shown = self._shown(prompt, u)
        mode = prompt.meta.get("mode")
        if mode == "thermometer":
            value = u[shown[-1]] + noise * rng.normal()
            return str(int(min(100, max(0, round(value)))))
        if mode == "ranking":
            eff = {
                c: u[c] + noise * rng.normal() - self.position_bias * i for i, c in enumerate(shown)
            }
            lines: list[list[str]] = []
            last = None
            for c in sorted(shown, key=lambda c: -eff[c]):
                if last is not None and abs(eff[c] - last) < 1e-9:
                    lines[-1].append(c)
                else:
                    lines.append([c])
                last = eff[c]
            return "\n".join(" = ".join(group) for group in lines)
        raise ValueError(f"mock backend cannot generate for mode {mode!r}")

    @staticmethod
    def _rng(prompt: ChatPrompt, seed: int, sample: int) -> np.random.Generator:
        digest = hashlib.sha256(prompt.text.encode("utf-8")).digest()
        return np.random.default_rng([int.from_bytes(digest[:8], "big"), int(seed), sample])
