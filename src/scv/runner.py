"""Resumable batch runner, call log and dry run (SPEC §11).

The call log is a directory of append-only Parquet shards,
``results/calls/<run_id>/part-XXXXX.parquet``. A shard is written to a temporary file
and renamed into place, so a kill at any moment leaves only whole shards. On start the
runner reads the keys already logged and runs only what is missing.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from scv.backends.base import Backend, BackendError
from scv.elicit import MODES, Call, ElicitConfig, Respondent, derive_seed, execute, plan_calls
from scv.prompts import load_templates, render_persona

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

CALL_SCHEMA = pa.schema(
    [
        ("key", pa.string()),
        ("run_id", pa.string()),
        ("election", pa.string()),
        ("respondent_id", pa.string()),
        ("model_key", pa.string()),
        ("hf_id", pa.string()),
        ("revision", pa.string()),
        ("backend", pa.string()),
        ("mode", pa.string()),
        ("target", pa.string()),
        ("candidates_shown", pa.list_(pa.string())),
        ("prompt_hash", pa.string()),
        ("temperature", pa.float64()),
        ("seed", pa.int64()),
        ("sample_idx", pa.int64()),
        ("raw_text", pa.string()),
        ("parsed", pa.string()),
        ("logprobs", pa.string()),
        ("parse_ok", pa.bool_()),
        ("retries", pa.int64()),
        ("latency_ms", pa.float64()),
        ("timestamp", pa.string()),
    ]
)

# --------------------------------------------------------------------------------------
# Call log
# --------------------------------------------------------------------------------------


class CallLog:
    """Append-only Parquet shards in one directory."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        for stale in self.dir.glob("*.tmp"):  # left by a kill in the middle of a write
            stale.unlink()

    def shards(self) -> list[Path]:
        return sorted(self.dir.glob("part-*.parquet"))

    def read(self, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """All logged rows (optionally only some columns)."""
        shards = self.shards()
        if not shards:
            names = list(columns) if columns else CALL_SCHEMA.names
            return CALL_SCHEMA.empty_table().select(names).to_pandas()
        cols = list(columns) if columns else None
        tables = [pq.read_table(p, columns=cols) for p in shards]
        return pa.concat_tables(tables).to_pandas()

    def keys(self) -> set[str]:
        return set(self.read(columns=["key"])["key"])

    def write(self, rows: Sequence[Mapping[str, Any]]) -> Path | None:
        """Write one shard atomically. Returns its path (``None`` if there are no rows)."""
        if not rows:
            return None
        existing = self.shards()
        nxt = int(existing[-1].stem.split("-")[1]) + 1 if existing else 0
        path = self.dir / f"part-{nxt:05d}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(list(rows), schema=CALL_SCHEMA)
        pq.write_table(table, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path


@dataclass(frozen=True)
class RunStats:
    planned: int
    skipped: int  # already in the log
    written: int


def run_calls(
    calls: Sequence[Call],
    backend: Backend,
    log: CallLog,
    run_id: str,
    *,
    flush_every: int = 500,
    batch_size: int = 32,
    max_retries: int = 2,
    progress: bool = False,
) -> RunStats:
    """Run every call whose key is not yet in ``log``, flushing a shard every
    ``flush_every`` rows. Safe to kill at any moment and call again.

    If the process is interrupted by an exception (including Ctrl-C), finished rows still
    in memory are flushed before it propagates. A hard kill loses at most the rows since
    the last flush; they are simply run again.
    """
    logged = log.read(columns=["key", "model_key", "revision"])
    done = set(logged["key"])
    info = backend.info()
    for model_key in {c.model_key for c in calls}:
        seen = set(logged.loc[logged["model_key"] == model_key, "revision"].dropna())
        if seen and seen != {info.get("revision")}:
            raise BackendError(
                f"run {run_id!r} already has rows for model {model_key!r} at revision "
                f"{sorted(seen)}, but the loaded model is at {info.get('revision')!r}. "
                "Use a new run_id rather than mixing revisions."
            )

    # A sequential chain is re-run whole if any of its turns is missing (its rows that
    # are already logged are dropped below).
    open_chains = {c.chain for c in calls if c.chain is not None and c.key not in done}
    todo = [c for c in calls if c.key not in done or c.chain in open_chains]
    n_new = sum(c.key not in done for c in todo)

    buffer: list[dict[str, Any]] = []
    written = 0
    bar = tqdm(total=n_new, desc=run_id, disable=not progress, unit="call")
    try:
        for batch in execute(todo, backend, max_retries=max_retries, batch_size=batch_size):
            fresh = [{**row, "run_id": run_id} for row in batch if row["key"] not in done]
            done.update(row["key"] for row in fresh)
            buffer += fresh
            bar.update(len(fresh))
            if len(buffer) >= flush_every:
                log.write(buffer)
                written += len(buffer)
                buffer = []
    finally:
        if buffer:
            log.write(buffer)
            written += len(buffer)
        bar.close()
    return RunStats(planned=len(calls), skipped=len(calls) - n_new, written=written)


# --------------------------------------------------------------------------------------
# Respondent subsampling
# --------------------------------------------------------------------------------------


def stratified_subsample(
    respondents: Sequence[Respondent], k: int | None, seed: int
) -> list[Respondent]:
    """Deterministic subsample of ``k`` respondents, stratified on ``Respondent.stratum``
    (party_id_7; SPEC §11).

    Each stratum gets its proportional share (largest remainders decide the leftovers).
    Within a stratum, respondents are taken in the order of a hash of ``(seed, id)``, so
    the choice does not depend on input order and a larger ``k`` contains a smaller one's
    picks from the same stratum. Returns respondents in their original order.
    """
    if k is None or k >= len(respondents):
        return list(respondents)
    if k <= 0:
        raise ValueError("limit must be positive")
    strata: dict[str, list[Respondent]] = {}
    for r in respondents:
        strata.setdefault("" if r.stratum is None else str(r.stratum), []).append(r)
    names = sorted(strata)
    exact = np.array([len(strata[s]) * k / len(respondents) for s in names])
    take = np.floor(exact).astype(int)
    # leftovers go to the largest remainders; ties broken by stratum name (stable sort)
    for i in np.argsort(-(exact - take), kind="stable")[: k - take.sum()]:
        take[i] += 1
    chosen: set[str] = set()
    for s, n in zip(names, take, strict=True):
        ordered = sorted(
            strata[s],
            key=lambda r: hashlib.sha256(f"{seed}|{r.respondent_id}".encode()).hexdigest(),
        )
        chosen.update(r.respondent_id for r in ordered[:n])
    return [r for r in respondents if r.respondent_id in chosen]


# --------------------------------------------------------------------------------------
# Run configuration
# --------------------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ElectionSpec(_Strict):
    """Where respondents and candidates come from. ``synthetic`` needs no data; the
    ANES source is added with the loader in M5."""

    source: Literal["synthetic"] = "synthetic"
    id: str
    candidates: list[str]
    question_text: str
    n_respondents: int = 20


class Throughput(_Strict):
    """Assumed backend speed, used only for the dry-run time estimate."""

    prompt_tokens_per_s: float = 20_000.0
    output_tokens_per_s: float = 2_500.0
    chars_per_token: float = 4.0


class RunConfig(_Strict):
    run_id: str
    seed: int = 0
    backend: Literal["mock", "hf", "vllm"]
    models: list[str]
    models_file: str = "configs/models/models.yaml"
    election: ElectionSpec
    modes: list[str] = Field(default_factory=lambda: list(MODES))
    thermometer_n: int = 5
    ranking_n: int = 3
    sequential: bool = False
    limit_respondents: int | None = None  # SPEC §14.5: pilot with 300 before a full run
    flush_every: int = 500
    batch_size: int = 32
    max_retries: int = 2
    results_dir: str = "results"
    backend_options: dict[str, Any] = Field(default_factory=dict)
    throughput: Throughput = Field(default_factory=Throughput)


def load_run_config(path: str | Path) -> RunConfig:
    with open(path, encoding="utf-8") as f:
        return RunConfig.model_validate(yaml.safe_load(f))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else REPO_ROOT / p


def load_models(path: str | Path) -> dict[str, dict[str, Any]]:
    """``model_key -> {hf_id, dtype?, base?}`` from every section of models.yaml."""
    with open(_resolve(path), encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    models: dict[str, dict[str, Any]] = {}
    for section, entries in raw.items():
        if not isinstance(entries, dict):
            continue  # e.g. homogeneous_default: <key>
        for key, spec in entries.items():
            if key in models:
                raise ValueError(f"model key {key!r} appears twice in {path}")
            models[key] = {**spec, "section": section}
    return models


# --------------------------------------------------------------------------------------
# Synthetic respondents (dev smoke runs and tests; the real ones come from ANES in M5)
# --------------------------------------------------------------------------------------

_PARTIES = (
    "a strong Democrat",
    "a not very strong Democrat",
    "an independent who leans Democratic",
    "an independent",
    "an independent who leans Republican",
    "a not very strong Republican",
    "a strong Republican",
)
_STATES = ("Ohio", "Texas", "Oregon", "Georgia", "Maine", "Arizona", "Michigan")


def synthetic_respondents(n: int, seed: int) -> list[Respondent]:
    """``n`` made-up personas with a party-id stratum. Not ANES data."""
    rng = np.random.default_rng(derive_seed("synthetic_respondents", seed))
    template = load_templates().persona
    out = []
    for i in range(n):
        party = _PARTIES[int(rng.integers(len(_PARTIES)))]
        fields = {
            "age": int(rng.integers(18, 90)),
            "gender": ("a man", "a woman")[int(rng.integers(2))],
            "state_or_region": _STATES[int(rng.integers(len(_STATES)))],
            "party_id_7": party,
        }
        out.append(Respondent(f"syn{i:05d}", render_persona(template, fields), stratum=party))
    return out


def synthetic_utilities(
    respondents: Sequence[Respondent], candidates: Sequence[str], seed: int
) -> dict[str, dict[str, float]]:
    """Random hidden utilities (0-100) for the mock backend."""
    rng = np.random.default_rng(derive_seed("synthetic_utilities", seed))
    return {
        r.respondent_id: {
            c: float(v)
            for c, v in zip(candidates, rng.integers(0, 101, len(candidates)), strict=True)
        }
        for r in respondents
    }


# --------------------------------------------------------------------------------------
# Planning, dry run, and running from a config
# --------------------------------------------------------------------------------------


def load_respondents(cfg: RunConfig, limit: int | None = None) -> list[Respondent]:
    """The run's respondents after ``--limit-respondents`` / ``limit_respondents``."""
    respondents = synthetic_respondents(cfg.election.n_respondents, cfg.seed)
    k = limit if limit is not None else cfg.limit_respondents
    return stratified_subsample(respondents, k, cfg.seed)


def elicit_config(cfg: RunConfig, model_spec: Mapping[str, Any]) -> ElicitConfig:
    """The elicitation settings for one model. A base model gets the completion frame
    and no ranking mode (SPEC §10.4)."""
    base = bool(model_spec.get("base", False))
    modes = [m for m in cfg.modes if not (base and m == "ranking")]
    if base and (cfg.sequential or "ranking" in cfg.modes):
        logger.warning(
            "base model %s: plain-completion prompts, thermometer and pairwise only "
            "(format differs from chat models; known confound)",
            model_spec.get("hf_id"),
        )
    return ElicitConfig(
        election=cfg.election.id,
        candidates=tuple(cfg.election.candidates),
        question_text=cfg.election.question_text,
        run_seed=cfg.seed,
        modes=tuple(modes),
        thermometer_n=cfg.thermometer_n,
        ranking_n=cfg.ranking_n,
        sequential=cfg.sequential and not base,
        base_model=base,
    )


def plan_run(cfg: RunConfig, limit: int | None = None) -> dict[str, list[Call]]:
    """``model_key -> planned calls`` for the whole run."""
    models = load_models(cfg.models_file)
    unknown = [m for m in cfg.models if m not in models]
    if unknown:
        raise ValueError(f"models not in {cfg.models_file}: {unknown}")
    respondents = load_respondents(cfg, limit)
    return {m: plan_calls(elicit_config(cfg, models[m]), respondents, m) for m in cfg.models}


def call_log_dir(cfg: RunConfig) -> Path:
    return Path(cfg.results_dir) / "calls" / cfg.run_id


def dry_run(cfg: RunConfig, limit: int | None = None) -> pd.DataFrame:
    """Call counts by model and mode, with a token and time estimate (SPEC §11).

    ``calls`` is the planned total, ``remaining`` what is not yet in the call log.
    Tokens and seconds are for the remaining calls: prompt tokens are characters /
    ``chars_per_token``, counted once per request (samples that share a prompt share
    one request); output tokens assume every call uses its full ``max_tokens``. Parse
    retries are not included.
    """
    plans = plan_run(cfg, limit)
    done = CallLog(call_log_dir(cfg)).keys() if call_log_dir(cfg).exists() else set()
    tp = cfg.throughput
    rows = []
    for model, calls in plans.items():
        for mode in MODES:
            mine = [c for c in calls if c.mode == mode]
            if not mine:
                continue
            left = [c for c in mine if c.key not in done]
            requests: dict[tuple, float] = {}
            history: dict[str, float] = {}
            for c in sorted(left, key=lambda c: c.turn):
                chars = float(len(c.prompt.text))
                if c.chain is not None:  # a later turn re-reads the conversation so far
                    history[c.chain] = history.get(c.chain, 0.0) + chars
                    chars = history[c.chain]
                    requests[(c.key,)] = chars
                else:
                    requests[(c.respondent_id, c.prompt, c.temperature)] = chars
            prompt_tokens = sum(requests.values()) / tp.chars_per_token
            output_tokens = float(sum(c.max_tokens for c in left))
            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "calls": len(mine),
                    "remaining": len(left),
                    "requests": len(requests),
                    "prompt_tokens": round(prompt_tokens),
                    "output_tokens": round(output_tokens),
                    "est_seconds": round(
                        prompt_tokens / tp.prompt_tokens_per_s
                        + output_tokens / tp.output_tokens_per_s,
                        1,
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_backend(cfg: RunConfig, model_key: str, respondents: Sequence[Respondent]) -> Backend:
    """Load the configured backend for one model. Fails fast with a clear message."""
    spec = load_models(cfg.models_file)[model_key]
    if cfg.backend == "mock":
        from scv.backends.mock import MockBackend

        utilities = synthetic_utilities(respondents, cfg.election.candidates, cfg.seed)
        return MockBackend(utilities, model_id=spec["hf_id"], **cfg.backend_options)
    if cfg.backend == "hf":
        from scv.backends.hf import HFBackend

        options = {"dtype": spec.get("dtype"), "base": spec.get("base", False)}
        return HFBackend(spec["hf_id"], **{**options, **cfg.backend_options})
    try:
        from scv.backends.vllm_backend import VLLMBackend
    except ImportError as e:
        raise BackendError(
            "the vllm backend is not available here; on the GPU box run "
            "`uv sync --extra gpu` (it is Linux/CUDA only)"
        ) from e
    options = {"dtype": spec.get("dtype"), "base": spec.get("base", False)}
    return VLLMBackend(spec["hf_id"], **{**options, **cfg.backend_options})


BackendFactory = Callable[[RunConfig, str, Sequence[Respondent]], Backend]


def run_from_config(
    cfg: RunConfig,
    limit: int | None = None,
    *,
    backend_factory: BackendFactory = make_backend,
    progress: bool = False,
) -> dict[str, RunStats]:
    """Run (or resume) every model in the config, one model at a time."""
    plans = plan_run(cfg, limit)
    respondents = load_respondents(cfg, limit)
    log = CallLog(call_log_dir(cfg))
    done = log.keys()
    stats: dict[str, RunStats] = {}
    for model, calls in plans.items():
        if all(c.key in done for c in calls):  # nothing to do: do not even load the model
            stats[model] = RunStats(planned=len(calls), skipped=len(calls), written=0)
            continue
        backend = backend_factory(cfg, model, respondents)
        stats[model] = run_calls(
            calls,
            backend,
            log,
            cfg.run_id,
            flush_every=cfg.flush_every,
            batch_size=cfg.batch_size,
            max_retries=cfg.max_retries,
            progress=progress,
        )
        del backend
    return stats


def read_calls(results_dir: str | Path, run_id: str) -> pd.DataFrame:
    """The whole call log of a run."""
    return CallLog(Path(results_dir) / "calls" / run_id).read()


__all__ = [
    "CALL_SCHEMA",
    "CallLog",
    "RunConfig",
    "RunStats",
    "call_log_dir",
    "dry_run",
    "load_models",
    "load_run_config",
    "plan_run",
    "read_calls",
    "run_calls",
    "run_from_config",
    "stratified_subsample",
]
