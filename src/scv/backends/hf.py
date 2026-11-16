"""Hugging Face `transformers` backend (SPEC §9): dev on the Mac, and the prod fallback.

Sampling is plain temperature sampling (no top-k / top-p / repetition penalty), whatever
the model's own `generation_config` says, so that every model is sampled the same way.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

from scv.backends.base import Backend, BackendError, ChatPrompt, per_prompt_seeds

_PROBE_CHAT = ChatPrompt.user("Answer with 1 or 2.")
_PROBE_COMPLETION = ChatPrompt.plain("Q: Answer with 1 or 2.\nA:")


def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolved_revision(hf_id: str, revision: str | None, model) -> str | None:
    """The commit hash of the snapshot that was actually loaded (SPEC §2: a run is
    determined by its config plus the model revision hash)."""
    commit = getattr(model.config, "_commit_hash", None)
    if commit:
        return commit
    cached = try_to_load_from_cache(hf_id, "config.json", revision=revision)
    if isinstance(cached, str):  # .../snapshots/<commit>/config.json
        return Path(cached).parent.name
    return revision


class HFBackend(Backend):
    """A causal LM loaded with `transformers`.

    ``dtype`` is a torch dtype name (``"bfloat16"``, ``"float16"``, ``"float32"``) or
    ``None`` for bfloat16 on CUDA and float32 elsewhere. ``base=True`` marks a base model
    with no chat template: it only accepts completion prompts (SPEC §10.4).
    """

    def __init__(
        self,
        hf_id: str,
        *,
        dtype: str | None = None,
        device: str = "auto",
        revision: str | None = None,
        base: bool = False,
        batch_size: int = 8,
        local_files_only: bool = False,
    ) -> None:
        self.hf_id = hf_id
        self.base = base
        self.batch_size = batch_size
        self.device = _pick_device(device)
        if dtype is None:
            dtype = "bfloat16" if self.device == "cuda" else "float32"
        if not hasattr(torch, dtype):
            raise BackendError(f"unknown torch dtype {dtype!r}")
        self.dtype = dtype
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                hf_id, revision=revision, local_files_only=local_files_only
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                revision=revision,
                dtype=getattr(torch, dtype),
                local_files_only=local_files_only,
            )
            self.model.to(self.device)
        except Exception as e:  # noqa: BLE001 - any load failure must fail fast and readably
            raise BackendError(
                f"could not load {hf_id!r} on {self.device} ({dtype}): {type(e).__name__}: {e}. "
                "Check the model id, that its licence is accepted and `hf auth login` was run, "
                "and that the model fits in memory."
            ) from e
        self.model.eval()
        self.revision = _resolved_revision(hf_id, revision, self.model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        if not base and not self.tokenizer.chat_template:
            raise BackendError(
                f"{hf_id!r} has no chat template; configure it with `base: true` to use the "
                "plain-completion prompt format"
            )
        self._label_cache: dict[tuple[tuple[str, ...], bool], list[int]] = {}
        self.check_labels(["1", "2"])  # SPEC §9: assert single-token labels at startup

    # -- prompt rendering --------------------------------------------------------------

    def render(self, prompt: ChatPrompt) -> str:
        """The exact string fed to the model: the model's own chat template with thinking
        turned off, or the completion text verbatim."""
        if prompt.is_completion:
            return prompt.completion
        if self.base:
            raise BackendError(f"{self.hf_id!r} is a base model and only takes completion prompts")
        return self.tokenizer.apply_chat_template(
            [{"role": role, "content": content} for role, content in prompt.messages],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # ignored by templates that have no thinking mode
        )

    def _encode(self, prompts: Sequence[ChatPrompt]):
        texts = [self.render(p) for p in prompts]
        # a rendered chat already carries its special tokens; a completion needs BOS added
        special = [p.is_completion for p in prompts]
        if len(set(special)) > 1:
            raise BackendError("cannot mix chat and completion prompts in one batch")
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=special[0]
        )
        return enc.to(self.device)

    def _guard(self, fn):
        try:
            return fn()
        except torch.OutOfMemoryError as e:
            raise BackendError(
                f"out of memory running {self.hf_id!r} on {self.device}; lower the batch size "
                "or use a smaller dtype"
            ) from e

    # -- interface ---------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        return {
            "model_id": self.hf_id,
            "revision": self.revision,
            "dtype": self.dtype,
            "backend": "hf",
            "device": self.device,
        }

    @torch.inference_mode()
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
        common = dict(max_new_tokens=max_tokens, pad_token_id=self.tokenizer.pad_token_id)
        out: list[list[str]] = []
        if temperature <= 0:  # greedy: deterministic, so prompts can share a batch
            for i in range(0, len(prompts), self.batch_size):
                enc = self._encode(prompts[i : i + self.batch_size])
                ids = self._guard(
                    lambda enc=enc: self.model.generate(**enc, do_sample=False, **common)
                )
                texts = self._decode(ids[:, enc["input_ids"].shape[1] :])
                out += [[t] * n for t in texts]
            return out
        for prompt, s in zip(prompts, seeds, strict=True):  # one seed per prompt
            enc = self._encode([prompt])
            torch.manual_seed(s)
            ids = self._guard(
                lambda enc=enc: self.model.generate(
                    **enc,
                    do_sample=True,
                    temperature=temperature,
                    top_k=0,
                    top_p=1.0,
                    repetition_penalty=1.0,
                    num_return_sequences=n,
                    **common,
                )
            )
            out.append(self._decode(ids[:, enc["input_ids"].shape[1] :]))
        return out

    def _decode(self, ids) -> list[str]:
        return [t.strip() for t in self.tokenizer.batch_decode(ids, skip_special_tokens=True)]

    def check_labels(self, candidates: Sequence[str]) -> None:
        """Raise unless every label is exactly one token right after the answer prefix."""
        self._label_ids(tuple(candidates), completion=self.base)

    def _label_ids(self, candidates: tuple[str, ...], completion: bool) -> list[int]:
        key = (candidates, completion)
        if key not in self._label_cache:
            probe = _PROBE_COMPLETION if completion else _PROBE_CHAT
            prefix = self.render(probe)
            add_special = completion
            base_ids = self.tokenizer(prefix, add_special_tokens=add_special)["input_ids"]
            ids = []
            for label in candidates:
                # after "A:" a completion model writes " 1"; after a chat prefix, "1"
                text = prefix + (" " + label if completion else label)
                full = self.tokenizer(text, add_special_tokens=add_special)["input_ids"]
                if len(full) != len(base_ids) + 1 or full[:-1] != base_ids:
                    raise BackendError(
                        f"answer label {label!r} is not a single token after the assistant "
                        f"prefix for {self.hf_id!r} (tokens: {full[len(base_ids) - 1 :]})"
                    )
                ids.append(full[-1])
            if len(set(ids)) != len(ids):
                raise BackendError(f"answer labels {candidates} share a token for {self.hf_id!r}")
            self._label_cache[key] = ids
        return self._label_cache[key]

    @torch.inference_mode()
    def next_token_logprobs(
        self, prompts: list[ChatPrompt], candidates: list[str]
    ) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            label_ids = self._label_ids(tuple(candidates), completion=batch[0].is_completion)
            enc = self._encode(batch)
            logits = self._guard(lambda enc=enc: self.model(**enc).logits[:, -1, :])
            # left padding puts each prompt's last real token in the final position
            picked = logits.float()[:, label_ids]
            logp = torch.log_softmax(picked, dim=-1).cpu().tolist()
            out += [dict(zip(candidates, row, strict=True)) for row in logp]
        return out
