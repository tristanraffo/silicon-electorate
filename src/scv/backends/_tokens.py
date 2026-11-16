"""Tokenizer-side logic shared by the `hf` and `vllm` backends, so both feed a model
exactly the same tokens: chat-template rendering, encoding, the single-token label
check (SPEC §9) and the resolved model revision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scv.backends.base import BackendError, ChatPrompt

PROBE_CHAT = ChatPrompt.user("Answer with 1 or 2.")
PROBE_COMPLETION = ChatPrompt.plain("Q: Answer with 1 or 2.\nA:")


def render(tokenizer: Any, prompt: ChatPrompt, *, base: bool, model: str) -> str:
    """The exact string fed to the model: the model's own chat template with thinking
    turned off, or the completion text verbatim."""
    if prompt.is_completion:
        return prompt.completion
    if base:
        raise BackendError(f"{model!r} is a base model and only takes completion prompts")
    return tokenizer.apply_chat_template(
        [{"role": role, "content": content} for role, content in prompt.messages],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # ignored by templates that have no thinking mode
    )


def encode(tokenizer: Any, prompt: ChatPrompt, *, base: bool, model: str) -> list[int]:
    """Token ids for one prompt. A rendered chat already carries its special tokens (so
    none are added again); a plain completion gets the tokenizer's usual BOS."""
    text = render(tokenizer, prompt, base=base, model=model)
    return list(tokenizer(text, add_special_tokens=prompt.is_completion)["input_ids"])


def require_chat_template(tokenizer: Any, *, base: bool, model: str) -> None:
    if not base and not getattr(tokenizer, "chat_template", None):
        raise BackendError(
            f"{model!r} has no chat template; configure it with `base: true` to use the "
            "plain-completion prompt format"
        )


def label_token_ids(
    tokenizer: Any, candidates: tuple[str, ...], *, completion: bool, base: bool, model: str
) -> list[int]:
    """Token id of each answer label, checked to be exactly one token right after the
    answer prefix (the assistant-turn prefix for chat, ``A:`` for a completion)."""
    probe = PROBE_COMPLETION if completion else PROBE_CHAT
    prefix = render(tokenizer, probe, base=base, model=model)
    base_ids = list(tokenizer(prefix, add_special_tokens=completion)["input_ids"])
    ids = []
    for label in candidates:
        # after "A:" a completion model writes " 1"; after a chat prefix, "1"
        text = prefix + (" " + label if completion else label)
        full = list(tokenizer(text, add_special_tokens=completion)["input_ids"])
        if len(full) != len(base_ids) + 1 or full[:-1] != base_ids:
            raise BackendError(
                f"answer label {label!r} is not a single token after the assistant "
                f"prefix for {model!r} (tokens: {full[len(base_ids) - 1 :]})"
            )
        ids.append(full[-1])
    if len(set(ids)) != len(ids):
        raise BackendError(f"answer labels {candidates} share a token for {model!r}")
    return ids


def resolved_revision(hf_id: str, revision: str | None, config: Any = None) -> str | None:
    """Commit hash of the snapshot actually loaded (SPEC §2), from the model config if
    it carries one, else from the Hugging Face cache path, else the requested revision."""
    commit = getattr(config, "_commit_hash", None)
    if commit:
        return commit
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(hf_id, "config.json", revision=revision)
    except Exception:  # noqa: BLE001 - a missing cache must not stop a run
        cached = None
    if isinstance(cached, str):  # .../snapshots/<commit>/config.json
        return Path(cached).parent.name
    return revision
