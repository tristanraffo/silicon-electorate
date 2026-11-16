"""Common backend interface (SPEC §9)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class BackendError(RuntimeError):
    """A backend could not load or run a model. Raised early, with a readable message."""


@dataclass(frozen=True)
class ChatPrompt:
    """What a model is asked.

    Either a chat (``messages``: ``(role, content)`` pairs, roles ``user`` / ``assistant``),
    rendered with the model's own chat template, or a plain ``completion`` string used
    verbatim for base models (SPEC §10.4). ``meta`` is bookkeeping that real backends
    ignore; the mock backend uses it to know which voter it is impersonating.
    """

    messages: tuple[tuple[str, str], ...] = ()
    completion: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if bool(self.messages) == (self.completion is not None):
            raise ValueError("a ChatPrompt needs either messages or a completion, not both")

    @classmethod
    def user(cls, text: str, **meta: Any) -> ChatPrompt:
        return cls(messages=(("user", text),), meta=meta)

    @classmethod
    def plain(cls, text: str, **meta: Any) -> ChatPrompt:
        return cls(completion=text, meta=meta)

    @property
    def is_completion(self) -> bool:
        return self.completion is not None

    def extended(self, assistant: str, user: str) -> ChatPrompt:
        """This prompt, plus the model's reply and a further user turn."""
        if self.is_completion:
            return ChatPrompt(
                completion=f"{self.completion} {assistant.strip()}\nQ: {user}\nA:", meta=self.meta
            )
        return ChatPrompt(
            messages=(*self.messages, ("assistant", assistant), ("user", user)), meta=self.meta
        )

    @property
    def last_turn(self) -> str:
        """The text of the question currently being asked (last user turn / last ``Q:``)."""
        if self.is_completion:
            return self.completion.rsplit("Q:", 1)[-1]
        return self.messages[-1][1]

    @property
    def text(self) -> str:
        """Flat text of the whole prompt, for hashing and debugging."""
        if self.is_completion:
            return self.completion
        return "\n".join(f"<{role}>\n{content}" for role, content in self.messages)


def per_prompt_seeds(seed: int | Sequence[int], k: int) -> list[int]:
    """One seed per prompt: a list is used as given, an int s becomes s, s+1, ..."""
    if isinstance(seed, int):
        return [seed + i for i in range(k)]
    seeds = [int(s) for s in seed]
    if len(seeds) != k:
        raise ValueError(f"expected {k} seeds, got {len(seeds)}")
    return seeds


class Backend(ABC):
    """A model that can sample text and report next-token log-probabilities."""

    @abstractmethod
    def generate(
        self,
        prompts: list[ChatPrompt],
        *,
        temperature: float,
        max_tokens: int,
        n: int,
        seed: int | Sequence[int],
    ) -> list[list[str]]:
        """``n`` completions per prompt. ``temperature=0`` is greedy.

        ``seed`` is one seed per prompt (or an int, expanded to consecutive seeds), so a
        prompt's samples do not depend on what else is in the batch.
        """

    @abstractmethod
    def next_token_logprobs(
        self, prompts: list[ChatPrompt], candidates: list[str]
    ) -> list[dict[str, float]]:
        """Log-probability of each candidate label as the next token, renormalised over
        the candidate labels only."""

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """``model_id``, ``revision``, ``dtype``, ``backend``."""
