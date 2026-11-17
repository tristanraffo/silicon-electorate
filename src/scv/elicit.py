"""Elicitation (SPEC §10): thermometer, full ranking and pairwise-by-log-probs.

Two stages, so the runner (SPEC §11) can skip finished work:

1. :func:`plan_calls` turns (config, respondents, model) into a deterministic list of
   :class:`Call` objects, one per row of the call log, each with a stable ``key``.
2. :func:`execute` runs any subset of those calls against a backend and yields batches
   of call-log records (plain dicts with the SPEC §11 fields, minus ``run_id``).

The ``*_table`` functions and :func:`pairwise_coherence` turn call-log rows back into
scores, ballots and the H4 coherence metrics.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import string
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scv.backends.base import Backend, ChatPrompt
from scv.ballots import WeakOrder
from scv.prompts import FORBIDDEN_WORDS, Templates, content_hash, load_templates

MODES = ("thermometer", "ranking", "pairwise")
GREEDY = -1  # sample_idx of the temperature-0 call
PAIR_SEP = "|"

# --------------------------------------------------------------------------------------
# Seeds and ordering
# --------------------------------------------------------------------------------------


def derive_seed(*parts: Any) -> int:
    """A 31-bit seed that depends only on the given parts (stable across runs/machines)."""
    blob = json.dumps([str(p) for p in parts], ensure_ascii=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "big") & 0x7FFFFFFF


def shuffled(items: Sequence[str], *seed_parts: Any) -> tuple[str, ...]:
    """``items`` in a pseudo-random order determined by ``seed_parts`` alone."""
    rng = np.random.default_rng(derive_seed(*seed_parts))
    return tuple(items[i] for i in rng.permutation(len(items)))


# --------------------------------------------------------------------------------------
# Configuration and planning
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Respondent:
    """One persona. ``persona`` is the rendered backstory text; ``stratum`` is not shown
    to the model."""

    respondent_id: str
    persona: str
    stratum: str | None = None  # party_id_7 label, for stratified subsampling


@dataclass(frozen=True)
class ElicitConfig:
    election: str
    candidates: tuple[str, ...]
    question_text: str
    run_seed: int = 0
    modes: tuple[str, ...] = MODES
    thermometer_n: int = 5
    ranking_n: int = 3
    sample_temperature: float = 1.0
    sequential: bool = False  # SPEC §14.1: independent thermometer calls by default
    base_model: bool = False  # SPEC §10.4: plain-completion frame, no chat template
    templates: Templates = field(default_factory=load_templates)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "modes", tuple(self.modes))
        if len(set(self.candidates)) != len(self.candidates) or len(self.candidates) < 2:
            raise ValueError("candidates must be at least 2 distinct names")
        for c in self.candidates:
            if any(ch in c for ch in (PAIR_SEP, "=", ">", "\n")) or c != c.strip() or not c:
                raise ValueError(f"candidate name {c!r} contains a reserved character")
        unknown = set(self.modes) - set(MODES)
        if unknown:
            raise ValueError(f"unknown elicitation modes: {sorted(unknown)}")
        for text in (self.question_text, *self.candidates):
            for word in FORBIDDEN_WORDS:
                if word in text.lower():
                    raise ValueError(f"prompt text contains the forbidden word {word!r}")
        if self.base_model:
            if "ranking" in self.modes:
                raise ValueError("base models support thermometer and pairwise modes only")
            if self.sequential:
                raise ValueError("sequential thermometer is not available for base models")

    def prompt_hash(self, mode: str) -> str:
        """Hash of everything that shapes the prompt text for ``mode``: the mode template,
        the persona template, the question wording and the prompt format."""
        return content_hash(
            [
                self.templates.for_mode(mode).hash,
                self.templates.persona.hash,
                self.question_text if mode == "thermometer" else "",
                "completion" if self.base_model else "chat",
                "sequential" if self.sequential and mode == "thermometer" else "independent",
            ]
        )


@dataclass(frozen=True)
class Call:
    """One planned model call = one row of the call log.

    ``target`` is the candidate (thermometer), ``"a|b"`` (pairwise, in candidate-list
    order) or ``""`` (ranking). ``candidates_shown`` is the presentation order.
    ``chain`` groups the turns of one sequential thermometer conversation; the turns of a
    chain must be executed together, in ``turn`` order.
    """

    election: str
    respondent_id: str
    model_key: str
    mode: str
    target: str
    candidates_shown: tuple[str, ...]
    sample_idx: int
    temperature: float
    max_tokens: int
    seed: int
    prompt_hash: str
    prompt: ChatPrompt
    repair: str = ""
    chain: str | None = None
    turn: int = 0

    @property
    def key(self) -> str:
        """SPEC §11 call key."""
        blob = json.dumps(
            [
                self.election,
                self.respondent_id,
                self.model_key,
                self.mode,
                self.target,
                list(self.candidates_shown),
                self.sample_idx,
                self.prompt_hash,
            ],
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def plan_calls(
    config: ElicitConfig, respondents: Iterable[Respondent], model_key: str
) -> list[Call]:
    """Every call for these respondents on one model, in a deterministic order.

    Presentation orders are seeded by ``(run_seed, respondent_id, candidate set,
    call type[, sample])`` and do not depend on the model, so every model sees the same
    text for the same persona (SPEC §10.5).
    """
    calls: list[Call] = []
    for r in respondents:
        if "thermometer" in config.modes:
            calls += _plan_thermometer(config, r, model_key)
        if "ranking" in config.modes:
            calls += _plan_ranking(config, r, model_key)
        if "pairwise" in config.modes:
            calls += _plan_pairwise(config, r, model_key)
    keys = [c.key for c in calls]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate call keys: respondent ids must be unique")
    return calls


def _samples(n: int) -> list[int]:
    return [*range(n), GREEDY]


def _candidate_set(config: ElicitConfig) -> str:
    return PAIR_SEP.join(sorted(config.candidates))


def _plan_thermometer(config: ElicitConfig, r: Respondent, model_key: str) -> list[Call]:
    t = config.templates.thermometer
    common = dict(
        election=config.election,
        respondent_id=r.respondent_id,
        model_key=model_key,
        mode="thermometer",
        max_tokens=int(t["max_tokens"]),
        prompt_hash=config.prompt_hash("thermometer"),
        repair=t["repair"],
    )
    meta = dict(respondent_id=r.respondent_id, mode="thermometer")
    calls = []
    if not config.sequential:
        for cand in config.candidates:
            values = dict(persona=r.persona, question_text=config.question_text, candidate=cand)
            if config.base_model:
                prompt = ChatPrompt.plain(t.render("completion", **values), **meta)
            else:
                prompt = ChatPrompt.user(t.render("user", **values), **meta)
            seed = derive_seed(
                config.run_seed, config.election, r.respondent_id, "thermometer", cand
            )
            for s in _samples(config.thermometer_n):
                calls.append(
                    Call(
                        target=cand,
                        candidates_shown=(cand,),
                        sample_idx=s,
                        temperature=0.0 if s == GREEDY else config.sample_temperature,
                        seed=seed,
                        prompt=prompt,
                        **common,
                    )
                )
        return calls

    # Sequential: one multi-turn conversation per sample, candidates in a random order.
    for s in _samples(config.thermometer_n):
        order = shuffled(
            config.candidates,
            config.run_seed,
            r.respondent_id,
            _candidate_set(config),
            "thermometer",
            s,
        )
        chain = f"{config.election}/{r.respondent_id}/{model_key}/thermometer/{s}"
        for turn, cand in enumerate(order):
            if turn == 0:
                text = t.render(
                    "user", persona=r.persona, question_text=config.question_text, candidate=cand
                )
            else:
                text = t.render("followup", candidate=cand)
            calls.append(
                Call(
                    target=cand,
                    candidates_shown=order,
                    sample_idx=s,
                    temperature=0.0 if s == GREEDY else config.sample_temperature,
                    seed=derive_seed(
                        config.run_seed, config.election, r.respondent_id, "thermometer", cand, s
                    ),
                    prompt=ChatPrompt.user(text, **meta),
                    chain=chain,
                    turn=turn,
                    **common,
                )
            )
    return calls


_ORDINALS = ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th")


def _render(template: Any, key: str, available: Mapping[str, str]) -> str:
    """Fill ``template[key]`` with the values it names (ranking templates differ in which
    of persona / n / letters / candidate_lines / slots they use). An unknown placeholder
    is a KeyError."""
    wanted = {name for _, name, _, _ in string.Formatter().parse(template[key]) if name}
    return template.render(key, **{name: available[name] for name in wanted})


def _plan_ranking(config: ElicitConfig, r: Respondent, model_key: str) -> list[Call]:
    t = config.templates.ranking
    calls = []
    for s in _samples(config.ranking_n):
        order = shuffled(
            config.candidates,
            config.run_seed,
            r.respondent_id,
            _candidate_set(config),
            "ranking",
            s,
        )
        letters = RANK_LABELS[: len(order)]
        fill = {
            "persona": r.persona,
            "n": str(len(order)),
            "letters": ", ".join(letters),
            "candidate_lines": "\n".join(f"{a}: {c}" for a, c in zip(letters, order, strict=True)),
            "slots": "\n".join(f"{o}: _" for o in _ORDINALS[: len(order)]),
        }
        text = _render(t, "user", fill)
        calls.append(
            Call(
                election=config.election,
                respondent_id=r.respondent_id,
                model_key=model_key,
                mode="ranking",
                target="",
                candidates_shown=order,
                sample_idx=s,
                temperature=0.0 if s == GREEDY else config.sample_temperature,
                max_tokens=int(t["max_tokens"]),
                seed=derive_seed(config.run_seed, config.election, r.respondent_id, "ranking", s),
                prompt_hash=config.prompt_hash("ranking"),
                prompt=ChatPrompt.user(text, respondent_id=r.respondent_id, mode="ranking"),
                repair=_render(t, "repair", fill),
            )
        )
    return calls


def _plan_pairwise(config: ElicitConfig, r: Respondent, model_key: str) -> list[Call]:
    t = config.templates.pairwise
    calls = []
    for a, b in itertools.combinations(config.candidates, 2):
        for first, second in ((a, b), (b, a)):  # both presentation orders
            values = dict(persona=r.persona, first=first, second=second)
            meta = dict(respondent_id=r.respondent_id, mode="pairwise")
            if config.base_model:
                prompt = ChatPrompt.plain(t.render("completion", **values), **meta)
            else:
                prompt = ChatPrompt.user(t.render("user", **values), **meta)
            calls.append(
                Call(
                    election=config.election,
                    respondent_id=r.respondent_id,
                    model_key=model_key,
                    mode="pairwise",
                    target=f"{a}{PAIR_SEP}{b}",
                    candidates_shown=(first, second),
                    sample_idx=0,
                    temperature=0.0,
                    max_tokens=1,
                    seed=0,
                    prompt_hash=config.prompt_hash("pairwise"),
                    prompt=prompt,
                )
            )
    return calls


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------

#: Letters shown next to the candidates in a ranking prompt, by position.
RANK_LABELS = tuple("ABCDEFG")

_INT = re.compile(r"(?<![\d.])\d+(?![\d])")
_LEADING_MARK = re.compile(r"^\s*(?:\(?\d+(?:st|nd|rd|th)?[.):]|[-*•])\s*")


def parse_thermometer(text: str) -> int | None:
    """The first integer in 0-100 (SPEC §10.1), or ``None``."""
    for match in _INT.finditer(text):
        value = int(match.group())
        if 0 <= value <= 100:
            return value
    return None


def parse_ranking(
    text: str, candidates: Sequence[str], labels: Sequence[str] | None = None
) -> WeakOrder | None:
    """Parse a ranking, one rank per line, ties joined by ``=`` (SPEC §10.2).

    Tolerates list markers ("1.", "-"), ``>`` as a rank separator and differences in
    case. Returns ``None`` unless every candidate appears exactly once and nothing else does.

    ``labels`` are the letters shown next to the candidates, in the same order. An item
    may then be the letter, the name, or the two together as shown ("B: Ann Lee").
    """
    lookup = {c.casefold(): c for c in candidates}
    for label, c in zip(labels or (), candidates, strict=bool(labels)):
        lookup[label.casefold()] = c
        for sep in (": ", ". ", ") ", " - ", " "):
            lookup[f"{label}{sep}{c}".casefold()] = c
    groups: list[list[str]] = []
    for line in re.split(r"[\n>]", text):
        line = _LEADING_MARK.sub("", line).strip()
        if not line:
            continue
        group = []
        for name in line.split("="):
            name = name.strip().strip("*_`\"'").strip().casefold()
            if name not in lookup:
                return None
            group.append(lookup[name])
        groups.append(group)
    flat = [c for g in groups for c in g]
    if len(flat) != len(candidates) or set(flat) != set(candidates):
        return None
    return WeakOrder(tuple(tuple(g) for g in groups))


def _parse(call: Call, text: str) -> Any:
    """Parsed value in its call-log form (JSON-serialisable), or ``None`` on failure."""
    if call.mode == "thermometer":
        return parse_thermometer(text)
    labels = RANK_LABELS[: len(call.candidates_shown)]
    ballot = parse_ranking(text, call.candidates_shown, labels)
    return None if ballot is None else [list(g) for g in ballot.groups]


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------


def _batched(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _record(
    call: Call,
    info: Mapping[str, Any],
    *,
    raw_text: str,
    parsed: Any,
    logprobs: Mapping[str, float] | None,
    retries: int,
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "key": call.key,
        "election": call.election,
        "respondent_id": call.respondent_id,
        "model_key": call.model_key,
        "hf_id": info.get("model_id"),
        "revision": info.get("revision"),
        "backend": info.get("backend"),
        "mode": call.mode,
        "target": call.target,
        "candidates_shown": list(call.candidates_shown),
        "prompt_hash": call.prompt_hash,
        "temperature": call.temperature,
        "seed": call.seed,
        "sample_idx": call.sample_idx,
        "raw_text": raw_text,
        "parsed": json.dumps(parsed),
        "logprobs": None if logprobs is None else json.dumps(dict(logprobs)),
        "parse_ok": parsed is not None,
        "retries": retries,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _generate_with_retries(
    backend: Backend,
    items: Sequence[tuple[Call, ChatPrompt]],
    first_texts: Sequence[str],
    max_retries: int,
) -> list[tuple[str, Any, int]]:
    """Parse each first answer; re-ask failures with the repair message up to
    ``max_retries`` times. Returns ``(final raw text, parsed or None, retries used)``."""
    state = [
        [text, _parse(call, text), 0] for (call, _), text in zip(items, first_texts, strict=True)
    ]
    prompts = [prompt for _, prompt in items]
    for attempt in range(1, max_retries + 1):
        failed = [i for i, s in enumerate(state) if s[1] is None]
        if not failed:
            break
        for temperature in sorted({items[i][0].temperature for i in failed}):
            idx = [i for i in failed if items[i][0].temperature == temperature]
            for i in idx:
                prompts[i] = prompts[i].extended(state[i][0], items[i][0].repair)
            outs = backend.generate(
                [prompts[i] for i in idx],
                temperature=temperature,
                max_tokens=max(c.max_tokens for c, _ in (items[i] for i in idx)),
                n=1,
                seed=[
                    derive_seed(items[i][0].seed, items[i][0].sample_idx, "retry", attempt)
                    for i in idx
                ],
            )
            for i, out in zip(idx, outs, strict=True):
                state[i] = [out[0], _parse(items[i][0], out[0]), attempt]
    return [tuple(s) for s in state]


def execute(
    calls: Iterable[Call], backend: Backend, *, max_retries: int = 2, batch_size: int = 32
) -> Iterator[list[dict[str, Any]]]:
    """Run ``calls`` and yield call-log records in batches (pairwise first, then
    single-turn generations, then sequential chains).

    Samples that share a prompt and temperature are requested together (``n > 1``).
    A parse failure is retried up to ``max_retries`` times with the template's repair
    message; after that the row is stored with ``parsed = null`` and ``parse_ok = False``.
    """
    calls = list(calls)
    info = backend.info()

    def ms_since(t0: float, k: int) -> float:
        return round(1000 * (time.perf_counter() - t0) / max(1, k), 3)

    # -- pairwise: next-token log-probs ------------------------------------------------
    pairwise = [c for c in calls if c.mode == "pairwise"]
    for chunk in _batched(pairwise, batch_size):
        t0 = time.perf_counter()
        logprobs = backend.next_token_logprobs([c.prompt for c in chunk], ["1", "2"])
        latency = ms_since(t0, len(chunk))
        yield [
            _record(
                c,
                info,
                raw_text="",
                parsed=None if lp is None else math.exp(lp["1"]),  # None = unreadable
                logprobs=lp,
                retries=0,
                latency_ms=latency,
            )
            for c, lp in zip(chunk, logprobs, strict=True)
        ]

    # -- single-turn generations, grouped so that shared prompts are sampled with n > 1 --
    groups: dict[tuple, list[Call]] = {}
    for c in calls:
        if c.mode != "pairwise" and c.chain is None:
            # keyed by respondent too: two people with the same persona text are still
            # separate requests with separate seeds
            key = (c.respondent_id, c.prompt, c.temperature, c.max_tokens)
            groups.setdefault(key, []).append(c)
    buckets: dict[tuple, list[list[Call]]] = {}
    for (_, _, temperature, max_tokens), group in groups.items():
        buckets.setdefault((temperature, max_tokens, len(group)), []).append(group)
    for (temperature, max_tokens, n), bucket in buckets.items():
        for chunk in _batched(bucket, max(1, batch_size // n)):
            t0 = time.perf_counter()
            outs = backend.generate(
                [g[0].prompt for g in chunk],
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                # the smallest sample index is part of the seed so that a resumed run,
                # which only asks for the missing samples, does not repeat earlier draws
                seed=[derive_seed(g[0].seed, min(c.sample_idx for c in g)) for g in chunk],
            )
            items = [(c, c.prompt) for g in chunk for c in g]
            texts = [text for out in outs for text in out]
            results = _generate_with_retries(backend, items, texts, max_retries)
            latency = ms_since(t0, len(items))
            yield [
                _record(
                    c,
                    info,
                    raw_text=raw,
                    parsed=parsed,
                    logprobs=None,
                    retries=r,
                    latency_ms=latency,
                )
                for (c, _), (raw, parsed, r) in zip(items, results, strict=True)
            ]

    # -- sequential chains: turn by turn, each chain carrying its own history ----------
    chains: dict[str, list[Call]] = {}
    for c in calls:
        if c.chain is not None:
            chains.setdefault(c.chain, []).append(c)
    for chain_calls in chains.values():
        chain_calls.sort(key=lambda c: c.turn)
        if [c.turn for c in chain_calls] != list(range(len(chain_calls))):
            raise ValueError(f"chain {chain_calls[0].chain!r} is incomplete; run whole chains")
    for chunk in _batched(list(chains.values()), batch_size):
        history: list[ChatPrompt | None] = [None] * len(chunk)
        for turn in range(max(len(ch) for ch in chunk)):
            live = [i for i, ch in enumerate(chunk) if turn < len(ch)]
            t0 = time.perf_counter()
            items = []
            for i in live:
                call = chunk[i][turn]
                if history[i] is None:
                    prompt = call.prompt
                else:
                    prompt = ChatPrompt(
                        messages=(*history[i].messages, *call.prompt.messages),
                        meta=call.prompt.meta,
                    )
                items.append((call, prompt))
            texts: list[str] = [""] * len(items)
            for temperature in sorted({c.temperature for c, _ in items}):
                idx = [j for j, (c, _) in enumerate(items) if c.temperature == temperature]
                outs = backend.generate(
                    [items[j][1] for j in idx],
                    temperature=temperature,
                    max_tokens=max(items[j][0].max_tokens for j in idx),
                    n=1,
                    seed=[items[j][0].seed for j in idx],
                )
                for j, out in zip(idx, outs, strict=True):
                    texts[j] = out[0]
            results = _generate_with_retries(backend, items, texts, max_retries)
            latency = ms_since(t0, len(items))
            records = []
            for i, (call, prompt), (raw, parsed, r) in zip(live, items, results, strict=True):
                # the conversation continues from the first question and the final answer
                history[i] = ChatPrompt(
                    messages=(*prompt.messages, ("assistant", raw)), meta=prompt.meta
                )
                records.append(
                    _record(
                        call,
                        info,
                        raw_text=raw,
                        parsed=parsed,
                        logprobs=None,
                        retries=r,
                        latency_ms=latency,
                    )
                )
            yield records


def run_elicitation(
    config: ElicitConfig,
    respondents: Iterable[Respondent],
    backend: Backend,
    model_key: str,
    **execute_kwargs: Any,
) -> pd.DataFrame:
    """Plan and execute everything in memory; returns the call log as a DataFrame.
    (The resumable, on-disk version is the runner.)"""
    calls = plan_calls(config, respondents, model_key)
    rows = [rec for batch in execute(calls, backend, **execute_kwargs) for rec in batch]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# From call-log rows back to scores, ballots and coherence metrics
# --------------------------------------------------------------------------------------

_ID = ["election", "model_key", "respondent_id"]


def _values(df: pd.DataFrame) -> pd.Series:
    return df["parsed"].map(json.loads)


def thermometer_table(calls: pd.DataFrame) -> pd.DataFrame:
    """One row per (election, model, respondent, candidate).

    ``score`` is the median of the successfully parsed sampled answers (SPEC §10.1; NaN
    if none parsed), ``greedy`` the temperature-0 answer, ``n_ok`` / ``n_samples`` the
    parse counts and ``sd`` the spread of the sampled answers.
    """
    df = calls[calls["mode"] == "thermometer"].copy()
    df["value"] = pd.to_numeric(_values(df), errors="coerce").astype(float)
    df["candidate"] = df["target"]
    sampled = df[df["sample_idx"] != GREEDY]
    out = (
        sampled.groupby([*_ID, "candidate"], sort=False)["value"]
        .agg(score="median", n_ok="count", n_samples="size", sd=lambda v: v.std(ddof=0))
        .reset_index()
    )
    greedy = df[df["sample_idx"] == GREEDY][[*_ID, "candidate", "value"]].rename(
        columns={"value": "greedy"}
    )
    return out.merge(greedy, on=[*_ID, "candidate"], how="outer")


def ranking_table(calls: pd.DataFrame) -> pd.DataFrame:
    """One row per ranking call, with ``ballot`` as a ``WeakOrder`` (``None`` on a parse
    failure) and ``candidates_shown`` as the presentation order."""
    df = calls[calls["mode"] == "ranking"].copy()
    df["ballot"] = [
        None if v is None else WeakOrder(tuple(tuple(g) for g in v)) for v in _values(df)
    ]
    cols = [*_ID, "sample_idx", "candidates_shown", "parse_ok", "ballot"]
    return df[cols].reset_index(drop=True)


def unreadable_pairwise(df: pd.DataFrame) -> pd.Series:
    """Pairwise calls whose answer could not be read: logged as failed (``parse_ok``
    False), or, in vLLM logs written before 2026-11-17, recorded as exactly 50/50 because
    neither label was among the top next tokens. Only the vLLM backend ever wrote such a
    placeholder, and a real vLLM answer of exactly 0.5 does not occur with float
    log-probs; the mock backend does give exact 50/50 answers for tied utilities, which
    are real (DECISIONS.md G19)."""
    p1 = _values(df).map(lambda v: np.nan if v is None else float(v))
    legacy = (df["backend"] == "vllm") & ((p1 - 0.5).abs() < 1e-12)
    return ~df["parse_ok"].astype(bool) | legacy


def pairwise_table(calls: pd.DataFrame) -> pd.DataFrame:
    """One row per (election, model, respondent, unordered pair a, b).

    ``p1_a_first`` = P("1" | a shown first), ``p1_b_first`` = P("1" | b shown first).
    ``p_a_over_b = 0.5 * (p1_a_first + (1 - p1_b_first))`` and
    ``position_bias = 0.5 * (p1_a_first + p1_b_first) - 0.5`` (SPEC §10.3).
    """
    df = calls[calls["mode"] == "pairwise"].copy()
    df["p1"] = _values(df).astype(float)
    df.loc[unreadable_pairwise(df), "p1"] = np.nan
    df[["a", "b"]] = df["target"].str.split(PAIR_SEP, expand=True, regex=False)
    df["a_first"] = [
        shown[0] == a for shown, a in zip(df["candidates_shown"], df["a"], strict=True)
    ]
    key = [*_ID, "a", "b"]
    orders = df.groupby(key, sort=False)["a_first"].agg(["nunique", "size"])
    if len(df) and ((orders["nunique"] != 2) | (orders["size"] != 2)).any():
        raise ValueError("pairwise calls must include both presentation orders for every pair")
    wide = df.set_index([*key, "a_first"])["p1"].unstack("a_first")
    wide = wide.reindex(columns=[True, False])
    out = wide.rename(columns={True: "p1_a_first", False: "p1_b_first"}).reset_index()
    out.columns.name = None
    out["p_a_over_b"] = 0.5 * (out["p1_a_first"] + (1.0 - out["p1_b_first"]))
    out["position_bias"] = 0.5 * (out["p1_a_first"] + out["p1_b_first"]) - 0.5
    return out


def pairwise_coherence(
    p: Mapping[tuple[str, str], float],
    candidates: Sequence[str],
    ranking: WeakOrder | None = None,
) -> dict[str, float]:
    """Per-voter H4 metrics from pairwise probabilities ``p[(a, b)] = p(a > b)``.

    Each unordered pair needs one entry (either orientation). A pair whose probability
    is NaN (the answer could not be read in at least one presentation order) is left
    out: triples containing it are not counted, nor is it in the ranking agreement.
    Returns

    - ``n_triples`` and ``wst_violations``: the number of candidate triples in which weak
      stochastic transitivity fails for some ordering (p(a>b) >= .5 and p(b>c) >= .5 but
      p(a>c) < .5), with ``wst_violation_share``;
    - ``strict_cycles``: triples that form a strict 3-cycle in the majority-of-probability
      relation (a beats b when p(a>b) > .5), with ``strict_cycle_share``;
    - ``ranking_agreement``: among pairs the voter's own full ranking orders strictly,
      the share where the sign of p - .5 points the same way (NaN without a ranking, or
      if the ranking orders no pair). A pair at exactly p = .5 counts as disagreement.
    """

    def prob(a: str, b: str) -> float:
        if (a, b) in p:
            return float(p[(a, b)])
        return 1.0 - float(p[(b, a)])

    def known(a: str, b: str) -> bool:
        return not math.isnan(prob(a, b))

    triples = [
        t
        for t in itertools.combinations(candidates, 3)
        if all(known(a, b) for a, b in itertools.combinations(t, 2))
    ]
    wst = cycles = 0
    for trio in triples:
        orderings = list(itertools.permutations(trio))
        wst += any(
            prob(a, b) >= 0.5 and prob(b, c) >= 0.5 and prob(a, c) < 0.5 for a, b, c in orderings
        )
        cycles += any(
            prob(a, b) > 0.5 and prob(b, c) > 0.5 and prob(c, a) > 0.5 for a, b, c in orderings
        )
    agreement = math.nan
    if ranking is not None:
        rank = ranking.rank_of()
        decided = [
            (a, b)
            for a, b in itertools.combinations(candidates, 2)
            if a in rank and b in rank and rank[a] != rank[b] and known(a, b)
        ]
        if decided:
            agree = sum(
                (prob(a, b) > 0.5) == (rank[a] < rank[b]) and prob(a, b) != 0.5 for a, b in decided
            )
            agreement = agree / len(decided)
    n = len(triples)
    return {
        "n_triples": n,
        "wst_violations": wst,
        "wst_violation_share": wst / n if n else math.nan,
        "strict_cycles": cycles,
        "strict_cycle_share": cycles / n if n else math.nan,
        "ranking_agreement": agreement,
    }
